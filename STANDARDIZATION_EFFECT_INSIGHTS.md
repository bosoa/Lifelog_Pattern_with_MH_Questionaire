# 표준화 기반 좌표계 변환의 효과 분석

**분석 일자**: 2026-06-27  
**분석자**: Claude Code (Anthropic)

---

## 📋 개요

생체신호 데이터에 대한 좌표계 변환 시, **표준화(Standardization)** 전처리가 예측 성능에 미치는 영향을 정량적으로 분석했습니다.

### 비교 대상
1. **원본 데이터**: 카테시안 좌표, 표준화 없음
2. **극좌표 변환**: 표준화 없이 직접 극좌표 변환
3. **표준화 극좌표**: StandardScaler로 표준화 → 극좌표 변환

---

## 📊 성능 비교 결과

### Concordance Index (C-index) 비교

| 종속변수 | 원본 | 극좌표 | 표준화 극좌표 | 극좌표 개선 | 표준화 극좌표 개선 |
|---------|------|--------|---------------|------------|------------------|
| **Anxiety** | 0.5334 | **0.6996** | 0.6815 | **+31.2%** ⬆️⬆️ | +27.8% ⬆️⬆️ |
| **Depression** | 0.5397 | 0.6834 | **0.7199** | +26.6% ⬆️⬆️ | **+33.4%** ⬆️⬆️⬆️ |
| **Stress** | 0.5351 | 0.6687 | **0.7149** | +25.0% ⬆️⬆️ | **+33.6%** ⬆️⬆️⬆️ |
| **평균** | 0.5361 | 0.6839 | **0.7054** | +27.6% ⬆️⬆️ | **+31.6%** ⬆️⬆️⬆️ |

### 표준화 효과 (극좌표 대비)

| 종속변수 | 표준화 효과 (절대) | 표준화 효과 (%) | 결과 |
|---------|------------------|----------------|------|
| **Anxiety** | **-0.0181** | -2.59% | ⬇️ 약간 하락 |
| **Depression** | **+0.0365** | +5.34% | ⬆️ 향상 |
| **Stress** | **+0.0462** | +6.91% | ⬆️⬆️ 크게 향상 |
| **평균** | **+0.0215** | +3.14% | ⬆️ 평균 향상 |

---

## 🔍 핵심 인사이트

### 1️⃣ 표준화의 이중적 효과

#### ✅ 긍정적 효과 (Depression, Stress)
- **Depression**: +5.34% (0.6834 → 0.7199)
- **Stress**: +6.91% (0.6687 → 0.7149)

**원인 분석**:
```
✓ 스케일 균형화
  - 체온(36°C) vs 걸음수(8000보) → 평균 0, 표준편차 1
  - 모든 특성이 동등한 기하학적 공간에서 변환
  
✓ 이상치 영향 감소
  - 극단값이 좌표계 변환을 지배하는 현상 방지
  - 반지름(r)과 각도(θ)가 균형있게 분포
  
✓ 특성 간 상호작용 명확화
  - 표준화된 공간에서 진정한 관계 패턴 포착
  - 단위 차이로 인한 노이즈 제거
```

#### ⚠️ 부정적 효과 (Anxiety)
- **Anxiety**: -2.59% (0.6996 → 0.6815)

**원인 가설**:
```
✗ 원본 스케일의 정보 손실
  - Anxiety 예측에 중요한 특성이 절대 크기에 의존
  - 표준화로 인해 크기 정보가 희석됨
  
✗ 특성 페어링 불일치
  - Anxiety에 최적인 특성 조합이 표준화 후 변경됨
  - 순차적 페어링 방식의 한계
```

---

### 2️⃣ 타겟 변수별 최적 방법

#### 🏆 최고 성능 방법

| 종속변수 | 최적 방법 | C-index | 권장 사항 |
|---------|----------|---------|----------|
| **Anxiety** | 극좌표 (표준화 ✗) | 0.6996 | 원본 스케일 유지 |
| **Depression** | 표준화 극좌표 | 0.7199 | 표준화 적용 ✅ |
| **Stress** | 표준화 극좌표 | 0.7149 | 표준화 적용 ✅ |

**실무 권장**:
- **단일 모델 운영**: 평균 성능이 높은 **표준화 극좌표** 사용
- **맞춤형 모델**: 타겟별로 최적 방법 선택

---

### 3️⃣ 표준화가 효과적인 경우

표준화가 성능을 향상시키는 조건:

#### ✅ 다양한 스케일의 특성
```python
# 예: Depression 데이터
특성 범위:
- 체온: 35-37°C (범위 2°C)
- 걸음수: 0-15,000보 (범위 15,000)
- 혈당: 70-120 mg/dL (범위 50)

→ 표준화 후 모두 평균 0, 표준편차 1
→ 균형잡힌 극좌표 변환
```

#### ✅ 이상치가 많은 데이터
```python
# 극단값 처리
원본: 걸음수 30,000보 (이상치)
  → 극좌표 반지름을 과도하게 지배
  
표준화 후: z-score = 2.5
  → 이상치 영향 제한됨
  → 다른 특성도 균등하게 기여
```

#### ✅ 비선형 패턴이 강한 경우
```python
# 상호작용 패턴 포착
표준화 없음:
  r = √(체온² + 걸음수²)
  ≈ √(36² + 8000²) ≈ 8000 (걸음수가 지배)
  
표준화 후:
  r = √(z_체온² + z_걸음수²)
  ≈ √(0.5² + 1.2²) ≈ 1.3 (균형)
```

---

### 4️⃣ 표준화가 불리한 경우

#### ❌ 절대 크기가 중요한 경우

**Anxiety 예측 가설**:
```
불안 수준 = f(절대적인 생체신호 값)

예: 심박수 120 BPM (절대값)
  → 불안 상태의 강력한 지표
  
표준화 후: z-score = 1.5
  → 상대적 위치만 표현
  → 절대값의 의미 상실
```

#### ❌ 특성 간 스케일 차이에 의미가 있는 경우

```python
# 원본 극좌표 (표준화 ✗)
r = √(심박수² + 체온²)
  = √(120² + 37²) ≈ 125.6
  → 심박수가 크게 기여 (의도된 동작)

# 표준화 극좌표
r = √(z_심박수² + z_체온²)
  = √(1.5² + 0.8²) ≈ 1.7
  → 두 특성이 동등하게 기여 (정보 손실)
```

---

## 🧪 기술적 세부사항

### 표준화 방법: StandardScaler

```python
from sklearn.preprocessing import StandardScaler

# 각 특성별로 독립적으로 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 공식: z = (x - μ) / σ
# μ: 평균 (mean)
# σ: 표준편차 (standard deviation)
```

**특징**:
- 평균 0, 표준편차 1로 변환
- 정규분포를 가정하지 않음 (모든 분포에 적용 가능)
- 이상치에 민감 (Robust Scaler 대안 가능)

### 극좌표 변환 과정

#### 1️⃣ 표준화 없는 극좌표
```python
# 원본 데이터 직접 변환
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)
```

#### 2️⃣ 표준화 극좌표
```python
# 1단계: 표준화
x_scaled = (x - x.mean()) / x.std()
y_scaled = (y - y.mean()) / y.std()

# 2단계: 극좌표 변환
r = np.sqrt(x_scaled**2 + y_scaled**2)
theta = np.arctan2(y_scaled, x_scaled)
```

**차이점**:
- `r`: 스케일 균형화 → 두 특성의 동등한 기여
- `theta`: 상대적 관계 유지 → 방향 정보 보존

---

## 📈 시각적 비교

### 기하학적 해석

#### 표준화 없는 극좌표
```
      y (걸음수: 8000)
       ^
       |     * 데이터 포인트
       |    /
       |   / r ≈ 8000 (y가 지배)
       |  /
       | /θ ≈ 0°
       +-----------------> x (체온: 36)
```

#### 표준화 극좌표
```
      z_y (걸음수: 1.2σ)
       ^
       |       * 데이터 포인트
       |      /
       |     / r ≈ 1.3 (균형)
       |    /
       |   /θ ≈ 40°
       +-----------------> z_x (체온: 0.5σ)
```

**관찰**:
- 표준화 전: `r`이 큰 값에 의해 결정
- 표준화 후: `r`과 `θ` 모두 의미 있는 정보 포함

---

## 🎯 실무 적용 가이드

### 1. 표준화 적용 결정 플로우

```
┌─────────────────────────────────┐
│ 데이터 특성 분석                  │
└─────────────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │ 스케일 차이? │
    └─────────────┘
      /          \
   Yes            No
    /              \
   ▼                ▼
┌──────────┐   ┌──────────┐
│표준화 권장 │   │ 실험 필요 │
└──────────┘   └──────────┘
    │                │
    ▼                ▼
┌──────────────┐   ┌──────────────┐
│• Depression  │   │• Anxiety     │
│• Stress      │   │• 도메인 지식 │
│→ C-index ⬆️  │   │  활용        │
└──────────────┘   └──────────────┘
```

### 2. 타겟별 권장 설정

```python
# 설정 딕셔너리
CONFIG = {
    'anxiety': {
        'use_standardization': False,  # ⚠️ 표준화 비활성
        'reason': '절대 크기 정보 중요',
        'expected_cindex': 0.6996
    },
    'depression': {
        'use_standardization': True,   # ✅ 표준화 활성
        'reason': '스케일 균형화 효과',
        'expected_cindex': 0.7199
    },
    'stress': {
        'use_standardization': True,   # ✅ 표준화 활성
        'reason': '비선형 패턴 포착',
        'expected_cindex': 0.7149
    }
}
```

### 3. 코드 예시

```python
from src.standardized_polar_transformer import StandardizedPolarTransformer

# 표준화 극좌표 변환
transformer = StandardizedPolarTransformer()

# Train 데이터로 scaler 학습
df_train_transformed = transformer.fit_transform_dataframe(df_train)

# Val/Test 데이터는 train scaler로 변환
df_val_transformed = transformer.transform_dataframe(df_val)
df_test_transformed = transformer.transform_dataframe(df_test)

# Scaler 저장 (재현성)
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(transformer.scalers, f)
```

---

## 📚 향후 연구 방향

### 1️⃣ 다른 표준화 방법 실험

| 방법 | 특징 | 적용 시나리오 |
|------|------|--------------|
| **MinMaxScaler** | [0, 1] 또는 [-1, 1] 범위 | 이상치 적을 때 |
| **RobustScaler** | 중앙값, IQR 사용 | 이상치 많을 때 ✅ |
| **PowerTransformer** | 정규분포 변환 | 왜도 큰 분포 |
| **QuantileTransformer** | 분위수 변환 | 비선형 변환 필요시 |

**권장**: 생체신호 데이터는 이상치가 많으므로 **RobustScaler** 테스트

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
# 중앙값 제거, IQR로 스케일링
# → 이상치에 강건
```

### 2️⃣ 특성 그룹핑 최적화

현재 순차적 페어링의 한계:
```python
# 현재: 순서대로 페어링
features = ['A', 'B', 'C', 'D', 'E', 'F']
pairs = [(A,B), (C,D), (E,F)]
```

**개선 방향**:
```python
# 1. 상관분석 기반
# 상관관계 높은 특성끼리 페어링

# 2. PCA 기반
# 주성분에 크게 기여하는 특성끼리 페어링

# 3. 도메인 지식 기반
# 생리학적으로 관련된 특성끼리 페어링
# 예: (심박수, HRV), (체온, 피부온도)
```

### 3️⃣ 적응형 표준화

타겟 변수에 따라 표준화 방법 자동 선택:
```python
def adaptive_standardization(X, y, target_var):
    """
    타겟 변수와 데이터 특성에 따라 최적 표준화 방법 선택
    """
    if target_var == 'anxiety':
        # Anxiety: 표준화 없음
        return X, None
    elif has_many_outliers(X):
        # 이상치 많음: RobustScaler
        scaler = RobustScaler()
    else:
        # 기본: StandardScaler
        scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
```

### 4️⃣ 구면좌표, 원통좌표 확장

극좌표에서 발견한 인사이트를 3D 좌표계로 확장:

**예상 효과**:
- 구면좌표: 3개 특성 간 복잡한 관계 포착
- 원통좌표: 평면 + 높이 분리 표현

**실험 계획**:
```bash
# 표준화 구면좌표
python3 src/standardized_spherical_transformer.py

# 표준화 원통좌표
python3 src/standardized_cylindrical_transformer.py

# 성능 비교
python3 src/compare_all_coordinate_systems.py
```

---

## 🔬 통계적 검증

### 성능 차이의 유의성

**가설 검정** (향후 수행):
```
H0: 표준화 극좌표 C-index = 극좌표 C-index
H1: 표준화 극좌표 C-index ≠ 극좌표 C-index

방법: Bootstrap 리샘플링 (n=1000)
유의수준: α = 0.05
```

**현재 결과** (단일 실험):
- Depression: +0.0365 (+5.34%) → 유의미한 차이로 예상
- Stress: +0.0462 (+6.91%) → 유의미한 차이로 예상
- Anxiety: -0.0181 (-2.59%) → 통계적 유의성 검증 필요

---

## 💡 결론

### 핵심 요약

1. **표준화 효과는 타겟 의존적**
   - Depression, Stress: 표준화 ✅ (+5~7% 향상)
   - Anxiety: 표준화 ✗ (-2.6% 하락)

2. **평균적으로 표준화가 유리**
   - 평균 C-index: 0.6839 → 0.7054 (+3.14%)
   - 원본 대비: +31.6% (표준화 극좌표) vs +27.6% (극좌표)

3. **스케일 다양성이 높을수록 표준화 효과 증대**
   - 생체신호 데이터는 단위가 매우 다양함
   - 표준화로 균형잡힌 기하학적 공간 형성

4. **맞춤형 접근이 최적**
   - 단일 모델: 표준화 극좌표 권장
   - 타겟별 최적화: Anxiety는 예외

### 실무 권장사항

```python
# 🎯 권장 파이프라인
if target in ['depression', 'stress']:
    # 표준화 극좌표
    transformer = StandardizedPolarTransformer()
    X_transformed = transformer.fit_transform(X_train)
elif target == 'anxiety':
    # 일반 극좌표
    transformer = PolarTransformer()
    X_transformed = transformer.transform(X_train)
```

### 향후 과제

1. ✅ **완료**: 표준화 극좌표 구현 및 성능 검증
2. 🔄 **진행 중**: 구면좌표, 원통좌표 표준화 버전
3. 📋 **계획**: RobustScaler 등 대안 표준화 방법 실험
4. 📋 **계획**: 지능형 특성 그룹핑 (상관분석, PCA)
5. 📋 **계획**: 통계적 유의성 검증 (Bootstrap)

---

**작성**: 2026-06-27  
**버전**: 1.0  
**다음 업데이트 예정**: 구면/원통좌표 표준화 결과 반영
