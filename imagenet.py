import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import csv
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Conv1: 227x227x3 -> 55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),  # LRN 대신 BatchNorm
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2: 27x27x96 -> 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),  # LRN 대신 BatchNorm
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3: 13x13x256 -> 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: 13x13x384 -> 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: 13x13x384 -> 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ==================== 공통 학습 함수 ====================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """1 epoch 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/len(train_loader):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def save_checkpoint(model, optimizer, epoch, val_acc, save_path):
    """체크포인트 저장"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """체크포인트 로드"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    print(f"Checkpoint loaded: epoch {epoch}, val_acc {val_acc:.2f}%")
    return epoch, val_acc


def plot_training_progress(history, save_path):
    """학습 진행 상황 그래프 생성"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
    
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train_loss'] for h in history]
    val_losses = [h['val_loss'] for h in history]
    train_accs = [h['train_acc'] for h in history]
    val_accs = [h['val_acc'] for h in history]
    lrs = [h['lr'] for h in history]
    
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss over Epochs', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, lrs, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    if len(epochs) >= 2:
        val_loss_diff = [val_losses[i] - val_losses[i-1] for i in range(1, len(val_losses))]
        axes[1, 1].plot(epochs[1:], val_loss_diff, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Val Loss Change', fontsize=12)
        axes[1, 1].set_title('Validation Loss Change', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Need more epochs...', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_model(model, train_loader, val_loader, num_epochs, lr, device,
                save_dir='checkpoints', model_name='alexnet'):
    """전체 학습 루프"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(save_dir, model_name, timestamp)
    
    checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    plots_dir = os.path.join(base_dir, 'plots')
    epochs_dir = os.path.join(base_dir, 'epochs')
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(epochs_dir, exist_ok=True)
    
    print(f"Run directory: {base_dir}")
    
    with open(os.path.join(base_dir, 'config.txt'), 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Batch Size: {train_loader.batch_size}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Num Epochs: {num_epochs}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Timestamp: {timestamp}\n")

    metrics_csv_path = os.path.join(base_dir, 'metrics.csv')
    with open(metrics_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'
        ])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_acc = 0.0
    history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # 학습
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 검증
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr}")

        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr,
        }
        history.append(epoch_metrics)
        
        epoch_dir = os.path.join(epochs_dir, f'epoch_{epoch+1:03d}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        with open(os.path.join(epoch_dir, 'metrics.json'), 'w') as jf:
            json.dump(epoch_metrics, jf, indent=2)
        
        with open(os.path.join(epoch_dir, 'metrics.txt'), 'w') as tf:
            tf.write(
                f"Epoch: {epoch+1}\n"
                f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%\n"
                f"Val   Loss: {val_loss:.6f}, Val   Acc: {val_acc:.2f}%\n"
                f"LR: {current_lr}\n"
            )
        
        with open(metrics_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{train_loss:.6f}", f"{train_acc:.4f}",
                f"{val_loss:.6f}", f"{val_acc:.4f}", current_lr
            ])
        
        plot_training_progress(history, os.path.join(plots_dir, 'training_progress.png'))
        print(f"Plot updated: {os.path.join(plots_dir, 'training_progress.png')}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                os.path.join(checkpoints_dir, f'{model_name}_best.pth')
            )

        save_checkpoint(
            model, optimizer, epoch, val_acc,
            os.path.join(checkpoints_dir, f'{model_name}_epoch_{epoch+1:03d}.pth')
        )

    with open(os.path.join(base_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    with open(os.path.join(base_dir, 'summary.txt'), 'w') as f:
        f.write("=== Training Summary ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Epochs: {num_epochs}\n")
        f.write(f"Best Val Accuracy: {best_acc:.2f}%\n")
        f.write(f"Final Train Loss: {history[-1]['train_loss']:.6f}\n")
        f.write(f"Final Train Acc: {history[-1]['train_acc']:.2f}%\n")
        f.write(f"Final Val Loss: {history[-1]['val_loss']:.6f}\n")
        f.write(f"Final Val Acc: {history[-1]['val_acc']:.2f}%\n")
    
    print(f"\nTraining completed! Best Val Acc: {best_acc:.2f}%")
    print(f"All logs and checkpoints are saved under: {base_dir}")
    return best_acc, base_dir


# ==================== 실행 예시 ====================
if __name__ == '__main__':
    # 하이퍼파라미터
    BATCH_SIZE = 128
    NUM_EPOCHS = 90  # 논문: 90 epochs
    LEARNING_RATE = 0.01  # 논문: 0.01
    NUM_CLASSES = 10  # CIFAR-10

    # Device 설정 - Mac용
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

    # 데이터 전처리 (CIFAR-10용, 227x227로 리사이즈)
    transform_train = transforms.Compose([
        transforms.Resize(227),
        transforms.RandomCrop(227, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 데이터셋 로드
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

    # 모델 생성
    model = AlexNet(num_classes=NUM_CLASSES).to(device)

    # 모델 파라미터 수 확인
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 학습 시작
    best_acc, run_dir = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        save_dir='runs',
        model_name='alexnet_cifar10'
    )
    
    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*60}")

    # ==================== 저장된 모델 로드 예시 ====================
    # model = AlexNet(num_classes=10).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # epoch, val_acc = load_checkpoint(model, optimizer, 'checkpoints/alexnet_cifar10_best.pth')