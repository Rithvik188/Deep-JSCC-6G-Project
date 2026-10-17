import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os

def get_face_dataloaders(batch_size=32, root='./data', img_size=128):
    """
    Downloads Labeled Faces in the Wild (LFW) and resizes them to img_size.
    This simulates video conferencing frames.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Outputs in [0, 1] range, perfect for our pipeline
    ])

    cifar_path = os.path.join(root, 'cifar-10-batches-py')
    if os.path.exists(cifar_path):
        print(f"Loading local image dataset from {cifar_path} (resized to {img_size}x{img_size})...")
        dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform)
    else:
        try:
            dataset = torchvision.datasets.LFWPeople(root=root, split='train', image_set='funneled', download=True, transform=transform)
        except Exception as e:
            print(f"Falling back to CIFAR10: {e}")
            dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)

    # Split into train/test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader
