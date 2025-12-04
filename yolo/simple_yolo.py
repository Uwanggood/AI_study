import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights, ResNet101_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
# ==========================================
# 1. 모델 설계 (Body): ResNet + YOLO Head
# ==========================================
class SimpleYolo(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet18을 가져와서 뒤에 2개(AvgPool, FC) 자르기
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Head: 채널을 512 -> 255 (3앵커 * 85정보)로 변경
        self.head = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 255, kernel_size=1),
        )

    def forward(self, x):
        # 1. Backbone 통과
        x = self.backbone(x) # 결과: [Batch, 512, 14, 14]

        # 2. Grid Size인 13x13으로 강제 조절
        x = nn.functional.interpolate(x, size=(13, 13))

        # 3. Head 통과 (채널 255개로 만듦)
        x = self.head(x) # 결과: [Batch, 255, 13, 13]

        # 4. 보기 좋게 순서 변경 [Batch, 13, 13, 255]
        x = x.permute(0, 2, 3, 1)

        # 5. Loss 계산을 위해 [Batch, 13, 13, 3, 45]로 모양 변경 (Reshape)
        batch_size = x.size(0)
        x = x.view(batch_size, 13, 13, 3, 85)

        # 좌표(0~3)와 Confidence(4), Class(5~)는 Sigmoid를 걸어 0~1로 맞춤
        # (학습 안정성을 위해 여기선 단순하게 전체 Sigmoid 적용)
        return torch.sigmoid(x)

# ==========================================
# 2. Loss Function (채점표)
# ==========================================
class MyYoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, pred, target):
        # target[..., 4]는 Confidence (물체 있으면 1, 없으면 0)
        obj_mask = target[..., 4] == 1
        noobj_mask = target[..., 4] == 0

        # [배경 손실] 물체 없는 곳은 0이 되어야 함
        noobj_loss = self.mse(pred[..., 4][noobj_mask], target[..., 4][noobj_mask])

        # [물체 손실] 물체 있는 곳은 1이 되어야 함
        obj_loss = self.mse(pred[..., 4][obj_mask], target[..., 4][obj_mask])

        # [좌표 손실] x,y,w,h (0:4) 비교
        box_loss = self.mse(pred[..., 0:4][obj_mask], target[..., 0:4][obj_mask])

        # [클래스 손실] 클래스 확률 (5:) 비교
        class_loss = self.mse(pred[..., 5:][obj_mask], target[..., 5:][obj_mask])

        # 최종 합산
        total_loss = (self.lambda_coord * box_loss) + \
                     obj_loss + \
                     (self.lambda_noobj * noobj_loss) + \
                     class_loss
        return total_loss

# ==========================================
# 3. 실행 및 테스트 (Training Loop)
# ==========================================
if __name__ == "__main__":
    # 1. 모델과 손실함수 생성
    model = SimpleYolo()
    criterion = MyYoloLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.01, patience=5)

    print("🚀 학습 시작! (가짜 데이터 1개로 과적합 테스트)")

    # 2. 가짜 데이터 생성 (Input Image)
    # 이미지 1장, RGB 3채널, 448x448 크기
    fake_input = torch.randn(1, 3, 448, 448)

    # 3. 가짜 정답 생성 (Target Label)
    # [1, 13, 13, 3, 85] 모양의 0으로 가득 찬 텐서 (모두 배경)
    fake_target = torch.zeros(1, 13, 13, 3, 85)

    # 정답 설정: (6, 6) 그리드의 0번 앵커에 "물체(개)"가 있다고 가정
    grid_y, grid_x = 6, 6
    anchor_idx = 0

    # 해당 위치에 정답 데이터 입력
    fake_target[0, grid_y, grid_x, anchor_idx, 0:4] = torch.tensor([0.5, 0.5, 0.2, 0.3]) # x,y,w,h
    fake_target[0, grid_y, grid_x, anchor_idx, 4] = 1.0  # Confidence (물체 있음!)
    fake_target[0, grid_y, grid_x, anchor_idx, 5] = 1.0  # Class 0번 (개) 확률 100%

    # 4. 학습 루프 (Training Loop)
    model.train()
    for epoch in range(200): # 100번 반복 학습
        # (1) 예측 (Forward)
        prediction = model(fake_input)

        # (2) 손실 계산 (Loss)
        loss = criterion(prediction, fake_target)

        # (3) 학습 (Backward)
        optimizer.zero_grad() # 이전 기울기 초기화
        loss.backward()       # 기울기 계산 (어느 방향으로 수정할지)
        optimizer.step()
        scheduler.step(metrics=loss)      # 가중치 수정 (학습)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/100] Loss: {loss.item():.6f}")

    print("✅ 학습 완료!")

    # 5. 결과 확인
    # 학습된 모델이 (6,6) 위치의 값을 제대로 뱉어내는지 확인
    final_pred = model(fake_input)
    center_pred = final_pred[0, 6, 6, 0]

    print("\n[결과 검증] (6, 6) 그리드 0번 앵커 예측값:")
    print(f"Conf (목표:1.0) -> {center_pred[4]:.4f}")
    print(f"Class0 (목표:1.0) -> {center_pred[5]:.4f}")
    print(f"Box (목표:0.5, 0.5...) -> {center_pred[0:4].tolist()}")