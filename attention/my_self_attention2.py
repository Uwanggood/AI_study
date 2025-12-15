import math

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 64
epochs = 10
learning_rate = 0.001

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(32, 32), patch_size=16, in_chans=3, embed_dim=64):
        super().__init__()

        if isinstance(img_size, int):
            img_h = img_w = img_size
        else:
            assert len(img_size) == 2, "img_size must be either an int or a tuple of length 2."
            img_h, img_w = img_size

        self.img_size = (img_h, img_w)
        self.patch_size = patch_size

        self.n_patches_h = img_h // patch_size
        self.n_patches_w = img_w // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)

        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.w_q(k).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.w_q(v).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        out = attn_weights @ V

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=64, num_heads=8, d_ff=256, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x

        x = self.self_attn(x, x, x, mask=mask)
        x = self.dropout1(x)

        x = self.norm1(x + residual)

        residual = x

        x = self.ffn(x)
        x = self.dropout2(x)

        x = self.norm2(x + residual)
        return x


class SimpleViT(nn.Module):
    def __init__(self, img_size=(32, 32), patch_size=16, d_model=64, in_chans=3, num_heads=8, num_layers=6,
                 num_classes=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff=d_model * 4)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)

        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)

        return self.head(x)


model = SimpleViT().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train_epoch(epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Backword pass
        loss = criterion(outputs, labels)
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
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(test_loader), 100 * correct / total

# ====== 7. 학습 루프 ======
train_losses = []
train_accs = []
test_losses = []
test_accs = []

best_acc = 0.0
best_model_path = './best_model.pth'

if "__main__" in __name__:
    print("\n=== 학습 시작 ===")

    for epoch in range(epochs):
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

        print(f"Epoch {epoch+1}/{epochs} - "
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