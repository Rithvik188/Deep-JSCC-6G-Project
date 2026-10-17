Deep JSCC 6G Project - Architecture State
Project Identity: End-to-End Learned Communication System using a Vision Transformer (ViT) with Region-of-Interest (ROI) Attention, evaluated over realistic Rayleigh/Rician fading channels.

Core Constraints & Math:

Input: CIFAR-10 Images (B, 3, 32, 32) normalized to [0, 1].

Baseband Representation: Tensors of shape (B, 2, K) where index 0 is Real (I) and index 1 is Imaginary (Q). K = 256 symbols.

Physics Boundary: ChannelPowerConstraint layer enforces average transmit power to exactly 1.0 (0 dBW) to prevent the network from cheating the noise.

Completed Components:

data/dataset.py: CIFAR-10 data loaders.

models/power_constraint.py: Hard power normalization layer.

models/encoder.py: ViTTransmitter converting image patches to complex IQ symbols.

channel/fading.py: Differentiable WirelessChannel applying exact complex arithmetic for AWGN, Rayleigh, and Rician fading.

models/decoder.py: ViTReceiver with self-attention (ROI) and Sigmoid reconstruction.

models/jscc_system.py: DeepJSCCSystem wrapping the end-to-end pipeline.

utils/losses.py: JSCCLoss combining MSE and SSIM.

train.py: Robust training loop with dynamic SNR sampling (0 to 20 dB) and dynamic channel sampling.

Milestone Achieved: The 1-batch sanity check ran successfully on CPU with a starting loss of ~0.107. Gradients flow perfectly.

Immediate Next Step: The project is ready for full training on the RTX 3060. The next coding task is to build Phase 5 (evaluate_visuals.py) to plot PSNR curves and generate the side-by-side image reconstruction grids.
