import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vggnet.vggnet import VGG
from trainer import train_model


if __name__ == '__main__':
    BATCH_SIZE = 128
    NUM_EPOCHS = 90
    LEARNING_RATE = 0.01
    NUM_CLASSES = 10
    CHECKPOINT_INTERVAL = 10

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Metal Performance Shaders) - Apple Silicon GPU")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    print(f"Device: {device}")

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = VGG(classes=NUM_CLASSES).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    best_acc, run_dir = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        save_dir='runs',
        model_name='vgg_cifar10',
        checkpoint_interval=CHECKPOINT_INTERVAL,
        lr_scheduler_step=30,
        lr_scheduler_gamma=0.1
    )
    
    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*60}")

