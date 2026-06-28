# Lifelog Pattern Data Generation

라이프로그 패턴 데이터 생성 프로젝트

## 개요

KLOSDOM 프로젝트의 생체신호 센서 및 EMA(Ecological Momentary Assessment) 데이터를 기반으로 라이프로그 패턴 데이터를 생성하는 시스템입니다.

본 프로젝트는 2,682명의 사용자로부터 93일간(2025-11-01 ~ 2026-02-01) 수집된 18종의 센서 데이터와 4종의 설문 데이터를 활용하여 건강 및 생활 패턴을 분석합니다.

## 🎯 프로젝트 목표

**생체신호 및 활동 패턴을 통한 정신건강 예측**

본 프로젝트의 핵심 목표는 웨어러블 센서로 수집된 생리적·행동적 데이터를 기반으로 사용자의 정신건강 상태(불안, 우울, 스트레스)를 예측하는 것입니다.

### 예측 모델 구조

- **독립변수 (Features)**: 18개 센서 데이터 (whole_a 시리즈)
  - 심혈관계 지표: HRV, 심박수, 산소포화도, 혈당, 혈압
  - 수면 패턴: 깊은/REM/얕은 수면, 총 수면 시간, 기상/취침 시간
  - 활동 지표: 걸음수, 이동 거리, 지팡이 센서
  - 환경 및 기타: 체온, 피부 온도, 조도, 스크린 타임

- **종속변수 (Target)**: 3개 정신건강 지표 (whole_e 시리즈)
  - `whole_e01_anxiety`: 불안 척도
  - `whole_e02_depression`: 우울 척도
  - `whole_e04_stress`: 스트레스 척도

- **참고**: `whole_e03_sleep` (수면 품질)은 독립변수 또는 중간 매개변수로 활용 가능

## 주요 기능

### 1️⃣ PCA 분석 웹 UI
- 대화형 주성분 분석 인터페이스
- 5가지 시각화 (Scree Plot, 2D/3D Biplot, 로딩 히트맵, 상세 결과)
- 실시간 파라미터 조정 및 결과 다운로드

### 2️⃣ 패턴 기반 계층화 데이터 생성
- PCA 결과 기반 주요 변수 자동 선택 (18개 → 10개)
- K-means 클러스터링으로 3단계 활동 수준 분류
- 계층별 통계 분석 및 패턴 해석

### 3️⃣ 데이터 전처리 및 품질 관리
- Wide → Long format 자동 변환
- 4단계 결측치 처리 (평균 → 0 → inf 제거 → 행 삭제)
- 데이터 정규화 및 이상치 처리

### 4️⃣ 변수 중요도 분석
- PCA 로딩 기반 변수 기여도 계산
- 설명 분산 가중 중요도 산출
- 정신건강 예측 핵심 변수 식별

---

## 📊 데이터셋 구조

### 기본 정보

- **총 파일 수**: 22개
  - 독립변수 (센서 데이터): 18개 파일
  - 종속변수 (정신건강 지표): 3개 파일 (불안, 우울, 스트레스)
  - 기타: 1개 파일 (수면 품질)
- **총 사용자 수**: 2,682명
- **측정 기간**: 2025-11-01 ~ 2026-02-01 (93일)
- **데이터 형식**: CSV (Wide format)
- **컬럼 구조**: `ID` + 93개 날짜 컬럼 = 총 94개 컬럼

### 데이터 형식

모든 데이터는 **Wide format** 시계열 구조입니다:
- 각 행: 개별 사용자
- 첫 번째 컬럼: 사용자 ID (`as[8자리숫자]` 형식)
- 나머지 컬럼: 날짜별 측정값 (2025-11-01부터 2026-02-01까지)

**예시:**
```csv
ID,2025-11-01,2025-11-02,2025-11-03,...
as20931165,0,0,0,...
as21882927,0,23.5,42.6,...
```

---

## 📁 데이터 파일 목록

### 1. 센서 데이터 (whole_a 시리즈) - 18개 파일 ✅ **독립변수**

> 정신건강 예측 모델의 **입력 특징(Features)**으로 사용됩니다.

#### 심혈관계
| 파일명 | 측정 지표 | 단위 | 설명 |
|--------|----------|------|------|
| `whole_a01_hrv_*.csv` | 심박변이도 (HRV) | ms | 심장 박동 간격의 변동성 |
| `whole_a08_heart_beat_*.csv` | 심박수 | BPM | 분당 심박수 |
| `whole_a06_oxygen_saturation_*.csv` | 산소포화도 | % | 혈중 산소 포화도 |
| `whole_a17_blood_sugar_*.csv` | 혈당 | mg/dL | 혈당 수치 |
| `whole_a18_blood_pressure_*.csv` | 혈압 | mmHg | 혈압 수치 |

#### 수면 관련
| 파일명 | 측정 지표 | 단위 | 설명 |
|--------|----------|------|------|
| `whole_a04_deep_sleep_*.csv` | 깊은 수면 시간 | 분 | 깊은 수면 단계 지속 시간 |
| `whole_a05_rem_sleep_*.csv` | REM 수면 시간 | 분 | REM 수면 단계 지속 시간 |
| `whole_a10_light_sleep_time_*.csv` | 얕은 수면 시간 | 분 | 얕은 수면 단계 지속 시간 |
| `whole_a15_total_sleep_time_*.csv` | 총 수면 시간 | 분 | 전체 수면 시간 |
| `whole_a12_wakeup_time_*.csv` | 기상 시간 | 시각 | 아침 기상 시각 |
| `whole_a13_bed_time_*.csv` | 취침 시간 | 시각 | 밤 취침 시각 |

#### 활동 관련
| 파일명 | 측정 지표 | 단위 | 설명 |
|--------|----------|------|------|
| `whole_a02_walk_*.csv` | 걸음수 | 보 | 일일 걸음 수 |
| `whole_a11_moving_distance_*.csv` | 이동 거리 | m | 일일 이동 거리 |
| `whole_a03_stick_sensor_activity_*.csv` | 지팡이 센서 활동 | - | 지팡이 사용 활동 |

#### 환경 및 기타
| 파일명 | 측정 지표 | 단위 | 설명 |
|--------|----------|------|------|
| `whole_a09_body_temperature_*.csv` | 체온 | ℃ | 체온 |
| `whole_a16_skin_temperature_*.csv` | 피부 온도 | ℃ | 피부 표면 온도 |
| `whole_a14_lux_sensor_*.csv` | 조도 | lux | 주변 밝기 |
| `whole_a07_screen_time_*.csv` | 스크린 타임 | ms | 디지털 기기 사용 시간 |

### 2. 설문 데이터 (whole_e 시리즈) - 4개 파일

#### 🎯 종속변수 (예측 대상)

| 파일명 | 측정 지표 | 척도 범위 | 역할 | 설명 |
|--------|----------|-----------|------|------|
| `whole_e01_anxiety_*.csv` | 불안 척도 | 0~6 | **종속변수** | 불안 수준 자가 평가 (예측 대상) |
| `whole_e02_depression_*.csv` | 우울 척도 | 0~6 | **종속변수** | 우울 수준 자가 평가 (예측 대상) |
| `whole_e04_stress_*.csv` | 스트레스 척도 | 0~6 | **종속변수** | 스트레스 수준 자가 평가 (예측 대상) |

#### 🔄 기타 설문 지표

| 파일명 | 측정 지표 | 척도 범위 | 역할 | 설명 |
|--------|----------|-----------|------|------|
| `whole_e03_sleep_*.csv` | 수면 품질 척도 | 0~6 | 독립/매개변수 | 수면 품질 자가 평가 (보조 특징) |

**특징:**
- 리커트 척도 형식 (0~6점)
- 주기적 측정 (약 7일 간격)
- 0은 결측치(미응답)를 의미
- **e01, e02, e04**는 모델의 예측 대상(Target)
- **e03**은 독립변수로 활용하거나 매개효과 분석에 사용 가능

---

## ⚠️ 데이터 품질 및 주의사항

### 결측치 처리

**중요:** 모든 결측치는 `0`으로 표시되어 있습니다.

- **센서 데이터**: 0은 "측정되지 않음"을 의미 (실제 0 값과 구분 필요)
- **설문 데이터**: 0은 "응답하지 않음"을 의미

### 데이터 밀도

- **센서 데이터**: 매우 높은 결측률 (사용자별 편차 큼)
  - 웨어러블 기기 착용/미착용에 따라 데이터 수집 여부 결정
  - 일부 사용자는 특정 기간에만 데이터 존재
  - 예: `whole_a13_bed_time` 파일은 현재 모든 값이 0 (미수집 상태)

- **설문 데이터**: 주기적 측정 패턴
  - 평균 7일 간격으로 응답
  - 일부 사용자는 불규칙한 응답 패턴

### 데이터 예시

```
사용자: as21882927
- HRV: 0, 0, 0, 23.5, 0, 0, 0, 0, 0, 42.6, 11.7, 13.6...
  → 간헐적 측정, 0은 결측치
  
- 걸음수: 0, 3298, 1961, 4925, 0, 2549, 4726...
  → 일부 날짜만 측정됨
  
- 불안 척도: 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0...
  → 주기적(약 7일 간격) 측정
```

---

## 🚀 빠른 시작

### 설치 및 실행

```bash
# 저장소 clone
git clone https://github.com/bosoa/Lifelog_Pattern_Data_Generation.git
cd Lifelog_Pattern_Data_Generation

# 필요한 패키지 설치
pip install -r requirements.txt

# macOS 사용자는 XGBoost를 위한 추가 설치
brew install libomp

# 통합 대시보드 열기 (모든 분석 결과 확인)
open model_results/index.html
```

**📚 상세 가이드**:
- [QUICKSTART.md](QUICKSTART.md) - 5분 안에 시작하기
- [SETUP.md](SETUP.md) - 완전한 설치 및 실행 가이드
- [PCA_ANALYSIS_GUIDE.md](PCA_ANALYSIS_GUIDE.md) - PCA 분석 상세 가이드

## 🎯 빠른 시작: PCA 분석 웹 UI

### PCA 분석 웹 앱 실행

```bash
# 방법 1: 쉘 스크립트 사용 (권장)
./run_pca_app.sh

# 방법 2: 직접 실행
streamlit run src/app.py
```

웹 앱이 시작되면 브라우저에서 `http://localhost:8501`로 자동 접속됩니다.

### 주요 기능

- **대화형 분석**: 종속변수(불안/우울/스트레스) 선택 및 PCA 파라미터 조정
- **5가지 시각화**:
  1. 📉 **Scree Plot**: 주성분별 설명 분산
  2. 🎯 **2D Biplot**: 주성분 공간에서 데이터 분포 및 변수 기여도
  3. 🌐 **3D Plot**: 입체적 데이터 분포 시각화
  4. 🔥 **로딩 히트맵**: 센서별 주성분 기여도
  5. 📋 **상세 결과**: 주성분별 상위 기여 변수
- **결과 다운로드**: 로딩 행렬 및 변환된 주성분 데이터 CSV 파일

자세한 사용법은 [PCA_ANALYSIS_GUIDE.md](PCA_ANALYSIS_GUIDE.md)를 참조하세요.

---

## 📊 계층화 데이터 생성

PCA 분석 결과를 바탕으로 주요 변수를 선택하고, 패턴별로 계층화된 데이터를 생성합니다.

### 계층화 데이터 생성 실행

```python
python3 src/hierarchical_data_generator.py
```

또는 Python 스크립트로:

```python
from src.hierarchical_data_generator import HierarchicalDataGenerator

generator = HierarchicalDataGenerator(output_dir='hierarchical_data')

# 특정 종속변수에 대해 생성
result = generator.generate_hierarchical_data(
    target_variable='anxiety',  # 'anxiety', 'depression', 'stress'
    n_pca_components=10,        # PCA 주성분 개수
    n_top_features=10,          # 선택할 주요 변수 개수
    n_levels=3,                 # 계층 개수
    clustering_method='kmeans', # 'kmeans' 또는 'quantile'
    min_data_points=10          # 최소 데이터 포인트
)
```

### 생성되는 파일

각 종속변수당 8개 파일이 생성됩니다:

1. `*_hierarchical_data.csv` - 전체 계층화 데이터
2. `*_level_statistics.csv` - 계층별 통계 요약
3. `*_feature_importance.csv` - 변수 중요도
4. `*_selected_features.csv` - 선택된 주요 변수
5. `*_level_0_낮은_활동.csv` - 레벨 0 데이터
6. `*_level_1_중간_활동.csv` - 레벨 1 데이터
7. `*_level_2_높은_활동.csv` - 레벨 2 데이터
8. `*_summary_report.txt` - 텍스트 리포트

### 계층화 프로세스

계층화 데이터 생성은 다음 7단계로 진행됩니다:

```
┌─────────────────────────────────────────────────────────────┐
│                    계층화 데이터 생성 파이프라인                │
└─────────────────────────────────────────────────────────────┘

1️⃣ 데이터 로드
   ├─ 18개 센서 데이터 (whole_a 시리즈)
   ├─ 종속변수 (anxiety/depression/stress)
   └─ Wide → Long format 변환
            ↓
2️⃣ PCA 분석
   ├─ 데이터 정규화 (StandardScaler)
   ├─ 주성분 분석 (n_components=10)
   └─ 설명 분산 및 로딩 계산
            ↓
3️⃣ 주요 변수 선택
   ├─ 방법 1: PCA 로딩 상위 변수 (PC1~3 기준)
   ├─ 방법 2: 가중 중요도 (설명 분산으로 가중)
   └─ 두 방법 결합 → 상위 10개 변수 선택
            ↓
4️⃣ 데이터 필터링
   └─ 선택된 10개 변수만 추출
            ↓
5️⃣ 패턴 계층화
   ├─ K-means 클러스터링 (n_clusters=3)
   ├─ 데이터 정규화 후 클러스터링
   └─ 평균값 기준으로 레이블 재정렬
            ↓
6️⃣ 계층별 분석
   ├─ 각 계층의 샘플 수 및 비율
   ├─ 계층별 종속변수 통계
   └─ 계층별 센서 변수 평균값
            ↓
7️⃣ 결과 저장
   ├─ 전체 계층화 데이터 CSV
   ├─ 계층별 분리 데이터 CSV (3개)
   ├─ 통계 및 중요도 CSV (3개)
   └─ 요약 리포트 TXT
```

### 기술적 세부사항

#### 1. PCA 기반 변수 선택

**목적**: 18개 센서 중 정신건강 예측에 가장 중요한 변수 선택

**알고리즘**:
```python
# 방법 1: PCA 로딩 기반
for PC in [PC1, PC2, PC3]:
    상위 N개 변수 선택 (|로딩 값| 기준)

# 방법 2: 가중 중요도
for 각 변수:
    중요도 = Σ(|PC_i 로딩| × PC_i 설명분산)

# 최종 선택
선택된_변수 = 방법1 ∪ 방법2의 상위 10개
```

**선택 기준**:
- PC1~3이 전체 분산의 50~60% 설명
- 로딩 절대값 > 0.3인 변수 우선
- 설명 분산으로 가중치 부여

#### 2. K-means 계층화

**목적**: 유사한 센서 패턴을 가진 샘플들을 3개 계층으로 분류

**알고리즘**:
```python
# 1단계: 정규화
X_scaled = StandardScaler().fit_transform(X_selected)

# 2단계: 클러스터링
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 3단계: 레이블 재정렬 (평균값 기준)
for cluster in [0, 1, 2]:
    cluster_mean[cluster] = X[labels == cluster].mean()

sorted_labels = argsort(cluster_mean)
# 0 = 낮은 활동, 1 = 중간 활동, 2 = 높은 활동
```

**파라미터 설명**:
- `n_clusters=3`: 낮음/중간/높음 3단계
- `n_init=10`: 초기값 10번 시도 (안정성 향상)
- `random_state=42`: 재현 가능성

#### 3. 계층 해석

각 계층의 의미는 센서 값의 평균으로 결정됩니다:

| 계층 | 정의 | 센서 특징 | 비율 |
|------|------|----------|------|
| **레벨 0** | 낮은 활동 | 낮은 수면, 낮은 활동 | ~70% |
| **레벨 1** | 중간 활동 | 매우 낮은 수면, 중간 활동 | ~7% |
| **레벨 2** | 높은 활동 | 높은 수면, 높은 활동 | ~21% |

### 변수 중요도 계산 방식

```python
def calculate_importance(loadings, explained_variance):
    """
    각 변수의 중요도를 설명 분산으로 가중 평균
    
    importance[i] = Σ(|loading[i,j]| × explained_variance[j])
                    for j in top_5_components
    """
    importance = zeros(n_features)
    
    for pc_idx in range(5):  # 상위 5개 주성분
        pc_loadings = abs(loadings[:, pc_idx])
        weight = explained_variance[pc_idx]
        importance += pc_loadings * weight
    
    return importance
```

**해석**:
- 중요도 > 0.15: 매우 중요 (skin_temperature, body_temperature)
- 중요도 > 0.10: 중요 (total_sleep, blood_sugar, light_sleep)
- 중요도 > 0.08: 보통 (walk, deep_sleep, rem_sleep)

### 주요 발견사항

**공통 주요 변수 (상위 10개)**:
- 수면 관련 (4개): total_sleep, light_sleep, deep_sleep, rem_sleep
- 체온 관련 (2개): skin_temperature, body_temperature
- 활동 관련 (2개): walk, stick_sensor
- 생리 지표 (2개): blood_sugar, hrv

**계층 분포 및 특성**:

| 레벨 | 이름 | 비율 | 평균 수면 | 평균 체온 | 정신건강 점수 |
|------|------|------|-----------|-----------|--------------|
| 0 | 낮은 활동 | ~70% | 6.3시간 | 34.9°C | 2.46 |
| 1 | 중간 활동 | ~7% | 1.9시간 ⚠️ | 35.4°C | 2.43 |
| 2 | 높은 활동 | ~21% | 8.0시간 | 36.2°C | 2.49 |

**패턴 인사이트**:
1. **수면이 핵심**: 레벨 1의 극단적으로 낮은 수면 시간 (1.9시간)
2. **체온과 활동의 상관**: 높은 활동 계층일수록 체온 높음
3. **정신건강 점수**: 계층 간 큰 차이 없음 (2.43~2.49)
   - 개별 변수보다는 **패턴 조합**이 중요함을 시사

### 실제 적용 예시

#### 예시 1: 새로운 사용자 분류

```python
# 새로운 사용자 데이터
new_user = {
    'total_sleep': 420,        # 7시간
    'rem_sleep': 120,
    'deep_sleep': 90,
    'light_sleep': 210,
    'skin_temperature': 30.5,
    'body_temperature': 36.0,
    'walk': 8000,
    'stick_sensor': 50,
    'blood_sugar': 95,
    'hrv': 25
}

# 계층 예측
from src.pattern_analyzer import PatternAnalyzer
analyzer = PatternAnalyzer()
level = analyzer.predict_level(new_user)  # → 레벨 2 (높은 활동)
```

#### 예시 2: 계층 간 전이 분석

```python
# 시간에 따른 계층 변화 추적
user_history = data[data['ID'] == 'as12345'][['date', 'level']]

# 계층 전이 확률
transition_prob = pd.crosstab(
    user_history['level'].shift(),
    user_history['level'],
    normalize='index'
)

# 결과:
#        0      1      2
# 0   0.85   0.10   0.05  ← 레벨 0에서 계속 0일 확률 85%
# 1   0.40   0.20   0.40
# 2   0.10   0.05   0.85
```

#### 예시 3: 맞춤형 건강 관리

```python
def recommend_action(level, sensor_values):
    if level == 0:
        return {
            'priority': 'HIGH',
            'action': '수면 시간 증가',
            'target': '7-8시간',
            'current': sensor_values['total_sleep'] / 60
        }
    elif level == 1:
        return {
            'priority': 'URGENT',
            'action': '즉시 수면 관리 필요',
            'target': '최소 6시간',
            'current': sensor_values['total_sleep'] / 60
        }
    else:
        return {
            'priority': 'LOW',
            'action': '현재 패턴 유지',
            'current': sensor_values['total_sleep'] / 60
        }
```

---

## 사용법

### 1. 데이터 로드 예시

```python
import pandas as pd
import numpy as np

# ============================================
# 독립변수 (Features) - 센서 데이터 로드
# ============================================
# 심혈관계
hrv_data = pd.read_csv('KLOSDOM_Preprocessed_Dataset/whole_a01_hrv_20260621.csv')
heart_beat = pd.read_csv('KLOSDOM_Preprocessed_Dataset/whole_a08_heart_beat_20260621.csv')
oxygen = pd.read_csv('KLOSDOM_Preprocessed_Dataset/whole_a06_oxygen_saturation_20260621.csv')

# 수면 관련
deep_sleep = pd.read_csv('KLOSDOM_Preprocessed_Dataset/whole_a04_deep_sleep_20260621.csv')
rem_sleep = pd.read_csv('KLOSDOM_Preprocessed_Dataset/whole_a05_rem_sleep_20260621.csv')
total_sleep = pd.read_csv('KLOSDOM_Preprocessed_Dataset/whole_a15_total_sleep_time_20260621.csv')

# 활동 관련
walk = pd.read_csv('KLOSDOM_Preprocessed_Dataset/whole_a02_walk_20260621.csv')
distance = pd.read_csv('KLOSDOM_Preprocessed_Dataset/whole_a11_moving_distance_20260621.csv')

# ============================================
# 종속변수 (Target) - 정신건강 지표 로드
# ============================================
anxiety = pd.read_csv('KLOSDOM_Preprocessed_Dataset/whole_e01_anxiety_20260621.csv')
depression = pd.read_csv('KLOSDOM_Preprocessed_Dataset/whole_e02_depression_20260621.csv')
stress = pd.read_csv('KLOSDOM_Preprocessed_Dataset/whole_e04_stress_20260621.csv')

# ============================================
# 결측치 처리 (0을 NaN으로 변환)
# ============================================
# 센서 데이터
hrv_data = hrv_data.replace(0, np.nan)
walk = walk.replace(0, np.nan)

# 설문 데이터
anxiety = anxiety.replace(0, np.nan)
depression = depression.replace(0, np.nan)
stress = stress.replace(0, np.nan)
```

### 2. 데이터 전처리 예시

```python
# Wide format을 Long format으로 변환
def wide_to_long(df, value_name):
    """
    Wide format (사용자×날짜) -> Long format (사용자-날짜 행)
    """
    df_long = df.melt(
        id_vars=['ID'], 
        var_name='date', 
        value_name=value_name
    )
    df_long['date'] = pd.to_datetime(df_long['date'])
    return df_long

# 변환 예시
hrv_long = wide_to_long(hrv_data, 'hrv')
anxiety_long = wide_to_long(anxiety, 'anxiety_score')

# 데이터 병합 (사용자 ID + 날짜 기준)
merged_data = hrv_long.merge(
    anxiety_long, 
    on=['ID', 'date'], 
    how='inner'
)
```

### 3. 모델 학습 준비 예시

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 결측치 제거 (또는 보간)
merged_data = merged_data.dropna()

# 특징(X)과 타겟(y) 분리
X = merged_data[['hrv', 'walk_count', 'deep_sleep', ...]]  # 독립변수
y = merged_data['anxiety_score']  # 종속변수

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

### 4. 데이터 분석 시 고려사항

1. **결측치 처리**: 
   - 0 값을 NaN으로 변환 후 분석
   - 센서 데이터의 높은 결측률 고려
   - 보간(interpolation) 또는 제거(dropna) 전략 선택

2. **시계열 변환**: 
   - Wide format을 Long format으로 변환 필요
   - 날짜를 datetime 타입으로 변환

3. **데이터 정규화**: 
   - 센서별 측정 단위와 범위가 다름
   - StandardScaler 또는 MinMaxScaler 사용 권장

4. **사용자 필터링**: 
   - 충분한 데이터를 가진 사용자만 선별
   - 예: 최소 30일 이상 측정된 사용자

5. **시간 지연(Lag) 고려**:
   - 센서 데이터(독립변수)와 설문 응답(종속변수) 간 시간차 존재
   - 설문은 약 7일 간격으로 측정됨
   - 시계열 특징 추출 시 rolling window 활용

---

## 🤖 머신러닝 모델 비교

선택된 10개 주요 변수로 정신건강 점수를 예측하는 모델을 비교합니다.

### 실행 방법

```bash
python3 run_model_comparison.py
```

### 모델

1. **RandomForest** - 앙상블 결정 트리
2. **XGBoost** - 그래디언트 부스팅

### 평가 지표

- **R² Score**: 모델 설명력 (0~1, 높을수록 좋음)
- **MAE**: 평균 절대 오차 (낮을수록 좋음)
- **RMSE**: 평균 제곱근 오차 (낮을수록 좋음)
- **Feature Importance**: 변수 중요도

### 성능 결과

| 모델 | Anxiety R² | Depression R² | Stress R² |
|------|-----------|--------------|-----------|
| RandomForest | 0.345 | 0.366 | 0.342 |
| **XGBoost** | **0.359** | **0.380** | **0.357** |

XGBoost가 모든 종속변수에서 우수한 성능을 보입니다.

### 생성 결과

- `model_results/anxiety_model_comparison_report.html`
- `model_results/depression_model_comparison_report.html`
- `model_results/stress_model_comparison_report.html`

---

## 📊 데이터 분할 (Train/Val/Test)

재현 가능한 모델 학습을 위해 데이터를 7:2:1 비율로 분할합니다.

### 실행 방법

```bash
python3 src/data_splitter.py
```

### 분할 전략

- **Train**: 70% - 모델 학습용
- **Validation**: 20% - 하이퍼파라미터 튜닝용
- **Test**: 10% - 최종 성능 평가용

### 계층화 분할 (Stratified Split)

이진 분류 레이블의 클래스 비율을 유지하면서 분할:
- 발생(≥4점) vs 미발생(<4점) 비율 동일 유지
- Random seed 42로 재현성 보장

### 생성 결과

`data_splits/` 디렉토리:
- `anxiety_train.csv`, `anxiety_val.csv`, `anxiety_test.csv`
- `depression_train.csv`, `depression_val.csv`, `depression_test.csv`
- `stress_train.csv`, `stress_val.csv`, `stress_test.csv`
- `split_summary.csv` - 분할 요약

---

## 🏥 생존 분석 (Cox Proportional Hazards)

정신건강 사건(≥4점) 발생을 예측하는 생존 분석을 수행합니다.

### 실행 방법

```bash
# 전체 변수 생존 분석
python3 src/survival_analysis.py

# p-value < 0.05 변수만 사용 (개선된 모델)
python3 src/survival_analysis_filtered.py
```

### Cox PH 모델

**수식**:
```
h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ... + βₙXₙ)
```

- `h(t|X)`: 시점 t에서 개인 X의 위험 함수
- `h₀(t)`: 기준 위험 함수
- `βᵢ`: 회귀 계수
- `Xᵢ`: 예측 변수

### 주요 분석 결과

#### 1. C-index (Concordance Index)

모델의 예측 정확도 (0.5=무작위, 1.0=완벽):

| 종속변수 | 전체 변수 | 필터링 변수 (p<0.05) |
|---------|----------|---------------------|
| Anxiety | 0.6807 | **0.6838** ↑ |
| Depression | 0.6971 | **0.6998** ↑ |
| Stress | 0.6751 | **0.6795** ↑ |

#### 2. Hazard Ratio (HR)

변수 1단위 증가 시 사건 발생 위험 배수:

**주요 위험 요인 (HR > 1)**:
- `body_temperature`: HR 1.15 (p<0.001)
- `blood_sugar`: HR 1.08 (p<0.01)

**보호 요인 (HR < 1)**:
- `total_sleep`: HR 0.92 (p<0.01)
- `hrv`: HR 0.88 (p<0.05)

#### 3. Likelihood Ratio Test

모델의 전반적 유의성:
- Anxiety: p < 10⁻¹⁷⁷ (극도로 유의)
- Depression: p < 10⁻¹⁸⁴ (극도로 유의)
- Stress: p < 10⁻¹⁸⁰ (극도로 유의)

### 시각화

각 리포트는 다음을 포함:

1. **Kaplan-Meier 생존 곡선** - 시간에 따른 생존 확률
2. **Hazard Ratio 플롯** - 변수별 위험도 (95% CI)
3. **Nomogram** - 개인별 위험 점수 계산 도구
4. **Calibration Plot** - 예측 정확도 검증
5. **변수 선택 비교** (필터링 모델) - 전후 성능 비교

### 생성 결과

**전체 변수 모델**:
- `model_results/anxiety_survival_analysis_report.html`
- `model_results/depression_survival_analysis_report.html`
- `model_results/stress_survival_analysis_report.html`

**필터링 모델 (p<0.05)**:
- `model_results/anxiety_survival_analysis_filtered_report.html`
- `model_results/depression_survival_analysis_filtered_report.html`
- `model_results/stress_survival_analysis_filtered_report.html`

---

## 📊 통합 대시보드

모든 분석 결과를 하나의 웹 페이지에서 확인할 수 있습니다.

### 열기

```bash
open model_results/index.html
```

### 구조

```
┌──────────────────────────────────────────┐
│  통합 분석 대시보드                        │
├──────────────────────────────────────────┤
│ 🏠 홈                                     │
│   ├─ 프로젝트 개요                        │
│   ├─ 주요 발견                            │
│   └─ 계층 분포                            │
├──────────────────────────────────────────┤
│ 📊 데이터 분포                            │
│   ├─ 히스토그램                           │
│   ├─ 박스플롯                             │
│   └─ 바이올린 플롯                        │
├──────────────────────────────────────────┤
│ 🤖 모델 비교 ▼                            │
│   ├─ Anxiety                             │
│   ├─ Depression                          │
│   └─ Stress                              │
├──────────────────────────────────────────┤
│ 🏥 생존 분석 ▼                            │
│   ├─ Anxiety                             │
│   ├─ Depression                          │
│   └─ Stress                              │
├──────────────────────────────────────────┤
│ 🎯 생존(필터) ▼                           │
│   ├─ Anxiety (p<0.05)                    │
│   ├─ Depression (p<0.05)                 │
│   └─ Stress (p<0.05)                     │
└──────────────────────────────────────────┘
```

---

## 📂 프로젝트 구조

```
Lifelog_Pattern_Data_Generation/
├── KLOSDOM_Preprocessed_Dataset/       # 원본 데이터 (22개 CSV)
│   ├── whole_a*.csv                    # 센서 데이터 (18개)
│   └── whole_e*.csv                    # 설문 데이터 (4개)
├── src/                                 # 소스 코드
│   ├── data_loader.py                  # 데이터 로드 및 전처리
│   ├── pca_analyzer.py                 # PCA 분석 및 시각화
│   ├── pattern_analyzer.py             # 패턴 분석 및 계층화
│   ├── hierarchical_data_generator.py  # 계층화 데이터 생성
│   ├── binary_classification_converter.py  # 이진 분류 변환
│   ├── data_splitter.py                # 데이터 분할
│   ├── model_comparison.py             # 모델 비교
│   ├── survival_analysis.py            # 생존 분석
│   ├── survival_analysis_filtered.py   # 필터링 생존 분석
│   ├── data_distribution_visualizer.py # 데이터 분포 시각화
│   └── app.py                          # Streamlit 웹 앱
├── hierarchical_data/                   # 생성된 계층화 데이터
│   ├── *_hierarchical_data.csv         # 전체 계층화 데이터
│   ├── *_binary_classification.csv     # 이진 분류 데이터
│   ├── *_level_statistics.csv          # 계층별 통계
│   ├── *_feature_importance.csv        # 변수 중요도
│   └── *_level_*_*.csv                 # 계층별 데이터
├── data_splits/                         # 분할된 데이터 (Train/Val/Test)
│   ├── *_train.csv                     # 학습 데이터 (70%)
│   ├── *_val.csv                       # 검증 데이터 (20%)
│   ├── *_test.csv                      # 테스트 데이터 (10%)
│   └── split_summary.csv               # 분할 요약
├── model_results/                       # 분석 결과 HTML
│   ├── index.html                      # 통합 대시보드 ⭐
│   ├── data_distribution_report.html   # 데이터 분포
│   ├── *_model_comparison_report.html  # 모델 비교 (3개)
│   ├── *_survival_analysis_report.html # 생존 분석 (3개)
│   └── *_survival_analysis_filtered_report.html  # 필터링 (3개)
├── requirements.txt                     # Python 패키지 목록
├── run_pca_app.sh                      # PCA 웹 앱 실행
├── run_model_comparison.py             # 모델 비교 실행
├── README.md                           # 프로젝트 개요 (이 파일)
├── QUICKSTART.md                       # 빠른 시작 가이드
├── SETUP.md                            # 상세 설정 가이드
├── PCA_ANALYSIS_GUIDE.md               # PCA 분석 가이드
└── .gitignore
```

---

## 📝 데이터 명명 규칙

- **파일명 형식**: `whole_{타입}{번호}_{지표명}_{생성일자}.csv`
  - `타입`: `a`(센서) 또는 `e`(설문)
  - `번호`: 01~18 (센서), 01~04 (설문)
  - `지표명`: 측정 항목명 (영문)
  - `생성일자`: YYYYMMDD

- **사용자 ID 형식**: `as[8자리숫자]` (예: as20931165)

- **날짜 형식**: `YYYY-MM-DD` (예: 2025-11-01)

---

## 📊 통계 요약

### 데이터 규모
- **전체 데이터 포인트**: 2,682명 × 93일 × 22개 지표 = 약 550만 데이터 포인트
- **측정 기간**: 3개월 (93일)
- **총 사용자 수**: 2,682명

### 변수 구성
- **독립변수 (Features)**: 18개 센서 데이터
  - 심혈관계: 5개 (HRV, 심박수, 산소포화도, 혈당, 혈압)
  - 수면 관련: 6개 (깊은/REM/얕은 수면, 총 수면, 기상/취침 시간)
  - 활동 관련: 3개 (걸음수, 이동 거리, 지팡이 센서)
  - 환경 및 기타: 4개 (체온, 피부 온도, 조도, 스크린 타임)

- **종속변수 (Target)**: 3개 정신건강 지표
  - 불안 척도 (whole_e01)
  - 우울 척도 (whole_e02)
  - 스트레스 척도 (whole_e04)

- **기타**: 1개 (수면 품질 척도 - 독립/매개변수)

### 측정 빈도
- **센서 데이터**: 일일 측정 (간헐적)
- **설문 데이터**: 주간 측정 (약 7일 간격)

### 계층화 데이터 통계

**생성된 계층화 데이터**:
- **총 샘플 수**: 44,165개
  - 불안(anxiety): 14,685개
  - 우울(depression): 14,798개
  - 스트레스(stress): 14,682개

**선택된 주요 변수**: 10개 (18개에서 차원 축소)
1. skin_temperature (피부 온도) - 중요도: 0.152
2. body_temperature (체온) - 중요도: 0.146
3. total_sleep (총 수면 시간) - 중요도: 0.145
4. blood_sugar (혈당) - 중요도: 0.145
5. light_sleep (얕은 수면) - 중요도: 0.140
6. walk (걸음수) - 중요도: 0.136
7. deep_sleep (깊은 수면) - 중요도: 0.116
8. rem_sleep (REM 수면) - 중요도: 0.115
9. stick_sensor (지팡이 센서) - 중요도: 0.101
10. hrv (심박변이도) - 중요도: 0.089

**계층 분포** (불안 기준):

| 계층 | 레벨명 | 샘플 수 | 비율 | 특징 |
|------|--------|---------|------|------|
| 0 | 낮은 활동 | 10,481 | 71.4% | 수면 부족 (6.3시간), 낮은 활동 |
| 1 | 중간 활동 | 1,073 | 7.3% | 극심한 수면 부족 (1.9시간) |
| 2 | 높은 활동 | 3,131 | 21.3% | 충분한 수면 (8.0시간), 높은 활동 |

**계층별 정신건강 점수**:

| 종속변수 | 레벨 0 | 레벨 1 | 레벨 2 | 전체 평균 |
|---------|--------|--------|--------|----------|
| 불안 | 2.46 ± 1.10 | 2.43 ± 1.14 | 2.49 ± 1.07 | 2.46 |
| 우울 | 2.47 ± 1.15 | 2.49 ± 1.21 | 2.50 ± 1.12 | 2.48 |
| 스트레스 | 2.95 ± 1.14 | 2.98 ± 1.17 | 2.96 ± 1.12 | 2.96 |

---

## 🔗 관련 프로젝트

- KLOSDOM (Korea Lifelog Observatory for Sustainable Dementia Outcome Management)

---

## 라이선스

(작성 예정)

---

## 기여

(작성 예정)

---

## 📚 전체 실행 파이프라인

```bash
# 1️⃣ PCA 분석 웹 UI (대화형)
./run_pca_app.sh

# 2️⃣ 계층화 데이터 생성
python3 src/hierarchical_data_generator.py

# 3️⃣ 이진 분류 변환
python3 src/binary_classification_converter.py

# 4️⃣ 데이터 분할 (Train/Val/Test)
python3 src/data_splitter.py

# 5️⃣ 데이터 분포 시각화
python3 src/data_distribution_visualizer.py

# 6️⃣ 모델 비교
python3 run_model_comparison.py

# 7️⃣ 생존 분석
python3 src/survival_analysis.py
python3 src/survival_analysis_filtered.py

# 8️⃣ 통합 대시보드 확인
open model_results/index.html
```

---

## 🎓 참고 문헌

### 분석 방법론

- **PCA (Principal Component Analysis)**: Jolliffe, I. T. (2002). Principal Component Analysis.
- **K-means Clustering**: MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
- **Cox Proportional Hazards**: Cox, D. R. (1972). Regression models and life-tables.
- **RandomForest**: Breiman, L. (2001). Random forests.
- **XGBoost**: Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.

### 응용 분야

- **라이프로그 분석**: 웨어러블 센서 기반 건강 모니터링
- **정신건강 예측**: 생체신호와 정신건강의 상관관계
- **생존 분석**: 사건 발생 예측 및 위험 요인 식별

---

**Created**: 2026-06-21  
**Last Updated**: 2026-06-21

**Contact**: bosoagalaxy@gmail.com  
**GitHub**: https://github.com/bosoa/Lifelog_Pattern_Data_Generation
