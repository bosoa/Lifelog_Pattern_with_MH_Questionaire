# 🧠 Lifelog Pattern with Mental Health Questionnaire

정신건강 설문과 라이프로그 생체신호 데이터를 통합 분석하는 프로젝트

## 📊 프로젝트 개요

본 프로젝트는 다음 3가지 핵심 인사이트를 기반으로 정신건강 데이터를 분석합니다:

1. **종단 연구 설계**: 1회차-2회차 추적을 통한 정신건강 변화 패턴 분석
2. **다차원 정신건강 평가**: 여러 척도를 통합한 정신건강 유형 분류
3. **라이프로그 통합**: 생체신호와 정신건강 지표 간 관계 탐색

## 📁 데이터셋

### 정신건강 설문 데이터
- **위치**: `MentalHealth_Questionaire/`
- **1회차**: 2,859명 참여자
- **2회차**: 1,463명 (51.2% 추적률)
- **주요 척도**:
  - PHQ-9 (우울)
  - GAD-7 (불안)
  - ISI (불면증)
  - PSS-10 (스트레스)
  - 외로움 척도
  - WHOQOL-BREF (삶의 질)

### 라이프로그 데이터
- **위치**: `KLOSDOM_Preprocessed_Dataset/`
- **생체신호**: 18개 유형 (HRV, 수면, 활동량, 심박수, 체온 등)
- **매칭률**: 90.1% (2,576명)

## 🔬 분석 결과

### 주요 발견사항

#### 1. 종단 분석 - 시간 경과 변화
- ✅ **모든 정신건강 지표 유의한 개선** (p < 0.05)
  - PHQ-9 우울: -0.82점 (p < 0.001)
  - GAD-7 불안: -0.80점 (p < 0.001)
  - ISI 불면증: -0.90점 (p < 0.001)

#### 2. 다차원 프로파일링
- **2개 클러스터 발견**:
  - 양호군: 56.1% (1,604명)
  - 취약군: 43.9% (1,255명)
- **우울-불안 공병성**: r=0.838 (매우 높음)

#### 3. 라이프로그 통합
- **46개 유의한 상관관계** 발견
- **예측 모델 한계**: 생체신호만으로는 정신건강 예측 어려움 (R² ≈ 0)

## 📂 프로젝트 구조

```
Lifelog_Pattern_with_MH_Questionaire/
├── MentalHealth_Questionaire/          # 정신건강 설문 원본 데이터
│   ├── (PSS추가)전체대상자_천안설문_1회차_250905.xlsx
│   ├── (PSS추가)전체대상자_천안설문_2회차_250905.xlsx
│   ├── 250822_코딩북.xlsx
│   └── 대상자 리스트(가입일)_1022.csv
│
├── KLOSDOM_Preprocessed_Dataset/       # 라이프로그 생체신호 데이터
│   ├── whole_a01_hrv_20260622.csv
│   ├── whole_a02_walk_20260622.csv
│   └── ... (18개 생체신호 파일)
│
├── src/                                # 분석 스크립트
│   ├── mh_data_loader.py               # 데이터 로더
│   ├── mh_data_exploration.py          # 데이터 탐색
│   ├── mh_longitudinal_analysis.py     # 종단 분석
│   ├── mh_multidimensional_profiling.py # 다차원 프로파일링
│   └── mh_lifelog_integration.py       # 라이프로그 통합
│
└── analysis_results/                   # 분석 결과
    ├── 01_data_exploration/            # 전처리 데이터 (11 CSV)
    ├── 02_longitudinal_analysis/       # 종단 분석 (10 PNG + 3 CSV)
    ├── 03_multidimensional_profiling/  # 프로파일링 (6 PNG + 2 CSV)
    ├── 04_lifelog_integration/         # 통합 분석 (2 PNG + 2 CSV)
    ├── 05_prediction_models/           # 예측 모델 (1 CSV)
    └── FINAL_REPORT.html               # 최종 통합 리포트
```

## 🚀 사용 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 데이터 탐색
```bash
python src/mh_data_exploration.py
```

### 3. 종단 분석
```bash
python src/mh_longitudinal_analysis.py
```

### 4. 다차원 프로파일링
```bash
python src/mh_multidimensional_profiling.py
```

### 5. 라이프로그 통합 분석
```bash
python src/mh_lifelog_integration.py
```

## 📊 결과 확인

최종 통합 리포트를 브라우저에서 확인:
```
analysis_results/FINAL_REPORT.html
```

## 📈 주요 지표

| 지표 | 값 |
|------|-----|
| 1회차 참여자 | 2,859명 |
| 2회차 참여자 | 1,463명 (51.2%) |
| 라이프로그 매칭 | 2,576명 (90.1%) |
| 정신건강 클러스터 | 2개 |
| 유의한 상관관계 | 46개 |
| 생성 결과 파일 | 43개 |

## 🔬 분석 방법론

- **통계 분석**: Paired t-test, Pearson correlation
- **클러스터링**: K-means, PCA, 계층적 클러스터링
- **머신러닝**: Random Forest (예측 모델)
- **시각화**: Matplotlib, Seaborn

## 📝 주요 발견

### ✅ 강점
1. 높은 추적률과 매칭률로 신뢰성 있는 종단 분석 가능
2. 다차원 척도 통합으로 정신건강 유형 분류 성공
3. 생체신호와 정신건강 지표 간 연관성 확인

### ⚠️ 한계
1. 생체신호만으로는 정신건강 상태 예측 어려움
2. 추가 특징 필요 (심리사회적 요인, 환경 요인 등)
3. 시계열 패턴 분석 필요

## 🎯 향후 연구 방향

1. **특징 공학 강화**: 생체신호의 시계열 패턴, 변화율 등 파생 특징 생성
2. **다중 모달리티**: 생체신호 + 설문 + 행동 데이터 통합
3. **딥러닝 적용**: LSTM, Transformer 등 시계열 모델
4. **개인화 모델**: 개인별 베이스라인 대비 변화 탐지
5. **개입 효과 분석**: 정신건강 개선 요인 탐색

## 📄 라이선스

본 프로젝트는 연구 목적으로 사용됩니다.

## 👥 기여자

- 박지환 (bosoagalaxy@gmail.com)

## 📅 업데이트

- 2026-06-28: 프로젝트 완료 및 최종 리포트 생성
