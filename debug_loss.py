import torch
import torch.nn as nn

# 데이터셋에서 하나의 샘플을 로드하여 loss 계산 과정을 디버그
from yolox.exp.large_object_exp import LargeObjectExp
from yolox.data import ValTransform

exp = LargeObjectExp()
dataset = exp.get_eval_dataset()

# 첫 번째 샘플 가져오기
img, target, img_info, img_id = dataset[0]

print(f"Image shape: {img.shape}")
print(f"Target shape: {target.shape}")
print(f"Target:\n{target}")
print(f"\nTarget stats:")
print(f"  Classes: {target[:, 0]}")
print(f"  BBox (cx, cy, w, h): {target[:, 1:5]}")
print(f"  cx range: [{target[:, 1].min():.4f}, {target[:, 1].max():.4f}]")
print(f"  cy range: [{target[:, 2].min():.4f}, {target[:, 2].max():.4f}]")
print(f"  w range: [{target[:, 3].min():.4f}, {target[:, 3].max():.4f}]")
print(f"  h range: [{target[:, 4].min():.4f}, {target[:, 4].max():.4f}]")

# 모델 생성 및 forward
model = exp.get_model()
model.eval()

# 배치로 만들기
imgs = torch.from_numpy(img).unsqueeze(0).float()
targets = target.unsqueeze(0).float()

print(f"\nBatch image shape: {imgs.shape}")
print(f"Batch target shape: {targets.shape}")

# Forward pass (training mode)
model.train()
try:
    outputs = model(imgs, targets)
    print(f"\nOutputs:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
except Exception as e:
    print(f"\nError during forward: {e}")
    import traceback
    traceback.print_exc()
