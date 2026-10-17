import torch
import torch.nn.functional as F
import pytest
from models.decoder import ViTReceiver
from models.jscc_system import DeepJSCCSystem, load_jscc_weights, MODEL_VERSION


def test_latent_pipeline_shapes_and_gradients():
    # VAE-free system (same trainable weights as the full system)
    system = DeepJSCCSystem(use_vae=False)
    latents = torch.randn(4, 4, 16, 16)

    for channel_type in ["awgn", "rician", "rayleigh"]:
        system.zero_grad()
        latents_hat = system.forward_latent(latents, snr_db=torch.tensor([0.0, 5.0, 10.0, 20.0]),
                                            channel_type=channel_type)
        assert latents_hat.shape == latents.shape

        F.mse_loss(latents_hat, latents).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.transmitter.parameters()), \
            "Transmitter parameters did not receive gradients"
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.receiver.parameters()), \
            "Receiver parameters did not receive gradients"


def test_receiver_equalizes_fading():
    # With no noise and perfect CSI, the receiver's equalized features must equal the sent
    # symbols for any fading draw, so the output cannot depend on h.
    torch.manual_seed(0)
    rx = ViTReceiver(img_size=16, patch_size=2, out_channels=4, use_sigmoid=False).eval()
    s = torch.randn(2, 2, 256)
    h1 = torch.randn(2, 2, 256)

    def apply(h):
        yr = h[:, 0] * s[:, 0] - h[:, 1] * s[:, 1]
        yi = h[:, 0] * s[:, 1] + h[:, 1] * s[:, 0]
        return torch.stack([yr, yi], dim=1)

    with torch.no_grad():
        # Equal |h| so the gain feature matches too; only the phase differs
        h2 = h1.flip(1) * torch.tensor([1.0, -1.0]).view(1, 2, 1)
        out1 = rx(apply(h1), h1, snr_db=200.0)
        out2 = rx(apply(h2), h2, snr_db=200.0)
    assert torch.allclose(out1, out2, atol=1e-4)


def test_checkpoint_round_trip_and_version_check(tmp_path):
    src = DeepJSCCSystem(use_vae=False)
    path = tmp_path / "ckpt.pth"
    torch.save({"model_state": src.state_dict(), "epoch": 3, "model_version": MODEL_VERSION}, path)

    dst = DeepJSCCSystem(use_vae=False)
    assert load_jscc_weights(dst, path, "cpu") == 3
    for k, v in src.state_dict().items():
        assert torch.equal(v, dst.state_dict()[k])

    torch.save({"model_state": src.state_dict(), "epoch": 3}, path)  # old, unversioned checkpoint
    with pytest.raises(RuntimeError, match="model version"):
        load_jscc_weights(dst, path, "cpu")

    bad = {k: v for k, v in src.state_dict().items() if not k.startswith("receiver.")}
    torch.save({"model_state": bad, "epoch": 3, "model_version": MODEL_VERSION}, path)
    with pytest.raises(RuntimeError, match="missing"):
        load_jscc_weights(dst, path, "cpu")


def test_config_round_trip_and_mismatch(tmp_path):
    from models.jscc_system import build_from_checkpoint
    src = DeepJSCCSystem(use_vae=False, num_symbols=512, receiver_mode="raw")
    path = tmp_path / "k512.pth"
    torch.save({"model_state": src.state_dict(), "epoch": 1, "model_version": MODEL_VERSION,
                "config": src.config}, path)

    model, epoch = build_from_checkpoint(path, "cpu", use_vae=False)
    assert epoch == 1 and model.config == src.config
    assert model.forward_latent(torch.randn(2, 4, 16, 16), snr_db=10.0, channel_type="rayleigh").shape == (2, 4, 16, 16)

    with pytest.raises(RuntimeError, match="trained with"):
        load_jscc_weights(DeepJSCCSystem(use_vae=False), path, "cpu")  # default K=256 mmse model
