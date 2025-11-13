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


def train_model(model, train_loader, val_loader, num_epochs, lr, device,
                save_dir='checkpoints', model_name='alexnet'):
    """전체 학습 루프"""
    # 러닝 세션 디렉토리: save_dir/model_name/YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(save_dir, model_name, timestamp)
    os.makedirs(base_dir, exist_ok=True)
    print(f"Run directory: {base_dir}")

    # 에폭별 서브디렉토리 및 CSV 요약 파일 준비
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

        # Learning rate 조정
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr}")

        # 에폭 디렉토리 생성 및 메트릭 저장
        epoch_dir = os.path.join(base_dir, f'epoch_{epoch+1:03d}')
        os.makedirs(epoch_dir, exist_ok=True)
        # JSON로 에폭 메트릭 저장
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': float(f"{train_loss:.6f}"),
            'train_acc': float(f"{train_acc:.6f}"),
            'val_loss': float(f"{val_loss:.6f}"),
            'val_acc': float(f"{val_acc:.6f}"),
            'lr': current_lr,
        }
        with open(os.path.join(epoch_dir, 'metrics.json'), 'w') as jf:
            json.dump(epoch_metrics, jf, indent=2)
        # 텍스트 요약도 저장
        with open(os.path.join(epoch_dir, 'metrics.txt'), 'w') as tf:
            tf.write(
                f"Epoch: {epoch+1}\n"
                f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%\n"
                f"Val   Loss: {val_loss:.6f}, Val   Acc: {val_acc:.2f}%\n"
                f"LR: {current_lr}\n"
            )
        # CSV에 한 줄 추가
        with open(metrics_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{train_loss:.6f}", f"{train_acc:.4f}",
                f"{val_loss:.6f}", f"{val_acc:.4f}", current_lr
            ])

        # 최고 성능 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                os.path.join(base_dir, f'{model_name}_best.pth')
            )

        # 매 10 epoch마다 저장
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_acc,
                os.path.join(base_dir, f'{model_name}_epoch_{epoch+1}.pth')
            )

    print(f"\nTraining completed! Best Val Acc: {best_acc:.2f}%")
    print(f"All logs and checkpoints are saved under: {base_dir}")
    return best_acc


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
    best_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        save_dir='checkpoints',
        model_name='alexnet_cifar10'
    )

    # ==================== 저장된 모델 로드 예시 ====================
    # model = AlexNet(num_classes=10).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # epoch, val_acc = load_checkpoint(model, optimizer, 'checkpoints/alexnet_cifar10_best.pth')