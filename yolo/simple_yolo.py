import torch
import torch.nn as nn

import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights


class SimpleYolo(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 255, kernel_size=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.interpolate(x, size=(13, 13))
        x = self.head(x)

        x = x.permute(0, 2, 3, 1)
        return x


if __name__ == "__main__":
    fake_image = torch.randn(1, 3, 448, 448)
    model = SimpleYolo()
    output = model(fake_image)
    print("입력 이미지 크기:", fake_image.shape)
    print("모델 출력 크기:", output.shape)

    # 30개 숫자 중 앞의 5개가 첫 번째 박스 정보입니다.
    # 7x7 그리드 중 정중앙(3, 3) 셀의 첫 번째 박스 예측값을 봅니다.
    print("정중앙 그리드 셀의 첫 번째 박스 예측값:", output[0, 3, 3, :5])

# resnet으로 호출된 이미지의 피쳐를 추출한다
# 추출된 피쳐를 7*7로 다운샘플링한다.
# 피쳐를 다시 30개의 컨볼루션으로 나오게한다
