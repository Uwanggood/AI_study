import numpy as np
import torch.cuda
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn as nn

from imagenet import alexnet
from imagenet_util import preprocess_image_with_rgb
from util import train, test

# https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html 참고
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
)

# 미리 섞여진 fashoin-mnist의 학습 데이터와 테스트 데이터 로드
# (학습 이미지, 학습 레이블), (테스트 이미지, 테스트 레이블)
x_before_preprocessing = training_data.data
x_label = training_data.targets
y_before_preprocessing = test_data.data
y_label = test_data.targets

x_label = x_label[:100]
y_label = y_label[:100]
x_train = preprocess_image_with_rgb(x_before_preprocessing[:100])
x_train_label = preprocess_image_with_rgb(y_before_preprocessing[:100])

# 레이블 정의
fashion_mnist_labels = ["T-shirt/top",  # 인덱스 0
                        "Trouser",  # 인덱스 1
                        "Pullover",  # 인덱스 2
                        "Dress",  # 인덱스 3
                        "Coat",  # 인덱스 4
                        "Sandal",  # 인덱스 5
                        "Shirt",  # 인덱스 6
                        "Sneaker",  # 인덱스 7
                        "Bag",  # 인덱스 8
                        "Ankle boot"]  # 인덱스 9

# 데이터 정규화
x_train = x_train.astype('float32') / 255
x_train_label = x_train_label.astype('float32') / 255

# 학습 데이터 셋을 학습 / 평가 셋으로 나눈다. (# 학습 셋: 55,000, 검증 셋 : 5000)
(x_train, x_test) = x_train[50:], x_train[:50]
(x_train_label, x_test_label) = x_train_label[50:], x_train_label[:50]

batch_size = 32
model = alexnet(num_classes=len(fashion_mnist_labels), pretrained=False)
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == 'cuda':
    model = model.cuda()

# 손실 함수와 옵티마이저를 정의한다.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
x_train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(np.array(x_train_label)))
x_test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(np.array(x_test_label)))
train_dataloader = DataLoader(x_train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(x_test_dataset, batch_size=batch_size)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)
print("Done!")
