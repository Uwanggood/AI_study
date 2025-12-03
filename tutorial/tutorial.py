import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ====== 설정 ======
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ====== 1. 데이터 로드 ======
print("\n=== 데이터 로드 중 ===")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ====== 2. 간단한 CNN 모델 (YOLO 대신 분류 모델 사용) ======
class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()

        # Backbone (ResNet 스타일)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # 32 -> 16

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)  # 16 -> 8

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)  # 8 -> 4

        # Global average pooling
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

# ====== 3. 모델 초기화 ======
print("\n=== 모델 초기화 ===")
model = SimpleModel(num_classes=10).to(DEVICE)
print(model)

# 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ====== 4. Loss, Optimizer, Scheduler ======
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ====== 5. 학습 함수 ======
def train_epoch(epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # 통계
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Progress bar 업데이트
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100 * correct / total
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})

    return total_loss / len(train_loader), 100 * correct / total

# ====== 6. 평가 함수 ======
def evaluate():
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(test_loader), 100 * correct / total

# ====== 7. 학습 루프 ======
print("\n=== 학습 시작 ===")
train_losses = []
train_accs = []
test_losses = []
test_accs = []

best_acc = 0.0
best_model_path = './best_model.pth'

for epoch in range(EPOCHS):
    # 학습
    train_loss, train_acc = train_epoch(epoch)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # 평가
    test_loss, test_acc = evaluate()
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    # Learning rate 업데이트
    scheduler.step()

    # 최고 모델 저장
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ Best model saved! (Test Acc: {test_acc:.2f}%)")

    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    print()

# ====== 8. 결과 시각화 ======
print("\n=== 학습 완료 ===")
print(f"최고 Test Accuracy: {best_acc:.2f}%")

plt.figure(figsize=(12, 4))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(test_losses, label='Test Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Test Loss')
plt.legend()
plt.grid(True)

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy', marker='o')
plt.plot(test_accs, label='Test Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training & Test Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('./training_results.png', dpi=100)
print("✓ 결과 그래프 저장: training_results.png")

# ====== 9. 최고 모델 로드 및 최종 평가 ======
print("\n=== 최고 모델 평가 ===")
model.load_state_dict(torch.load(best_model_path))
final_loss, final_acc = evaluate()
print(f"Final Test Loss: {final_loss:.4f}")
print(f"Final Test Accuracy: {final_acc:.2f}%")

# ====== 10. 모델 저장 ======
print(f"\n✓ 모델 저장: {best_model_path}")