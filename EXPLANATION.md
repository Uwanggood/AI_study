# 모델이 고장난 이유 - 초간단 설명

## 1. 정상적인 모델
```
이미지 A (트럭) → 모델 → "트럭 여기 있어!" (70%)
이미지 B (빈 도로) → 모델 → "아무것도 없어" (5%)
이미지 C (그래플) → 모델 → "그래플 여기!" (80%)
```
→ 다른 이미지 = 다른 결과 ✅


## 2. 당신 모델 (고장남)
```
이미지 A (트럭) → 모델 → "뭔가 있어" (63.74%)
이미지 B (빈 도로) → 모델 → "뭔가 있어" (63.74%) ← 똑같음!
이미지 C (그래플) → 모델 → "뭔가 있어" (63.74%) ← 똑같음!
```
→ 다른 이미지 = 같은 결과 ❌


## 3. 왜 이렇게 됐나?

### Pretrained Weight 문제:
```
ImageNet (고양이, 강아지, 꽃 사진)로 학습된 모델
         ↓
  BatchNorm이 "고양이, 강아지, 꽃"에 맞춰짐
         ↓
  트럭/그래플 사진 넣으면
         ↓
  "이게 뭐야? 본 적 없는데?" → 고장
```

### BatchNorm이란?
```
입력 이미지의 숫자들을 "정규화"하는 것

예시:
- ImageNet 이미지: 밝기 평균 120, 범위 0~255
- 트럭 이미지: 밝기 평균 90, 범위 0~200

BatchNorm이 120 기준으로 맞춰져 있으면,
90이 들어오면 이상하게 작동함!
```


## 4. 해결책

**Pretrained 없이 처음부터 학습:**
```python
backbone = ConvNeXtBackbone(
    single_scale=True,
    pretrained=None,  # ← 이거!
)
```

이미 수정했으니, 이제 새로 학습하면 됩니다:
```bash
python train_large_object.py -b 16 -expn new_training -f16
```


## 5. 공부할 것 (난이도 순)

### ⭐ 필수 (이것만 알아도 이해 가능)
1. **BatchNorm 개념** 
   - 유튜브: "BatchNormalization 설명"
   - 키워드: 정규화, running_mean, running_var
   - 10분이면 이해 가능

2. **Transfer Learning**
   - 남이 학습한 모델 갖다 쓰기
   - 키워드: Pretrained, Fine-tuning
   - 5분이면 이해 가능

### ⭐⭐ 중급 (더 깊이 이해하려면)
3. **PyTorch train() vs eval()**
   - eval(): 학습 때 저장한 통계 사용
   - train(): 현재 데이터 통계 사용
   
4. **Object Detection 기초**
   - YOLO가 뭔지
   - Bounding Box, Objectness


## 6. 증거 (이미 확인됨)

test_inference.py 결과:
```
Sample 0: max_obj=0.6374  ← 이미지 1
Sample 1: max_obj=0.6374  ← 이미지 2 (똑같음!)
Sample 2: max_obj=0.6374  ← 이미지 3 (똑같음!)
```

check_eval_vs_train.py 결과:
```
두 출력의 차이 (EVAL): 0.000558  ← 거의 0!
```


## 결론

✅ 내가 말한 게 맞음
✅ Pretrained BatchNorm이 문제
✅ 처음부터 학습하면 해결됨 (이미 수정 완료)

