import torch
import torch.nn as nn

# 간단한 YOLOX Decoupled Head 예제
class SimpleYOLOXHead(nn.Module):
    def __init__(self, num_classes=80, in_channels=256):
        super().__init__()
        self.num_classes = num_classes
        
        # 1개 feature level만 사용 (실제는 3개: P3, P4, P5)
        
        # stem: 입력 feature 정제
        self.stem = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        
        # cls branch: 클래스 분류용
        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1, 1, 0)
        
        # reg branch: bbox 회귀용
        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.reg_pred = nn.Conv2d(in_channels, 4, 1, 1, 0)  # x, y, w, h
        
        # obj branch: objectness (객체 존재 여부)
        self.obj_pred = nn.Conv2d(in_channels, 1, 1, 1, 0)
    
    def forward(self, x):
        # x shape: [batch, 256, H, W] (예: [1, 256, 80, 80])
        
        # stem 통과
        x = self.stem(x)  # [1, 256, 80, 80]
        
        # 분리된 두 경로
        cls_feat = self.cls_convs(x)  # [1, 256, 80, 80]
        reg_feat = self.reg_convs(x)  # [1, 256, 80, 80]
        
        # 최종 prediction
        cls_output = self.cls_pred(cls_feat)  # [1, 80, 80, 80] - 80개 클래스
        reg_output = self.reg_pred(reg_feat)  # [1, 4, 80, 80] - bbox 좌표
        obj_output = self.obj_pred(reg_feat)  # [1, 1, 80, 80] - objectness
        
        return cls_output, reg_output, obj_output


# 테스트
if __name__ == "__main__":
    # 가상의 feature map 생성 (Backbone에서 나온 것처럼)
    batch_size = 1
    feature = torch.randn(batch_size, 256, 80, 80)  # [B, C, H, W]
    
    # Head 생성
    head = SimpleYOLOXHead(num_classes=80)
    
    # Forward
    cls_out, reg_out, obj_out = head(feature)
    
    print(f"Input feature: {feature.shape}")
    print(f"Class output: {cls_out.shape}")  # [1, 80, 80, 80]
    print(f"Bbox output:  {reg_out.shape}")   # [1, 4, 80, 80]
    print(f"Obj output:   {obj_out.shape}")    # [1, 1, 80, 80]
    
    # 각 위치마다 예측값 확인
    print(f"\n한 위치(0,0)에서의 예측:")
    print(f"  클래스 점수 (80개): {cls_out[0, :, 0, 0].shape}")
    print(f"  bbox (x,y,w,h): {reg_out[0, :, 0, 0]}")
    print(f"  objectness: {obj_out[0, :, 0, 0].item():.3f}")