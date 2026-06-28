# 프로젝트 설정 가이드

GitHub에서 프로젝트를 클론한 후 바로 실행할 수 있도록 하는 완전한 설정 가이드입니다.

## 📋 목차

1. [환경 요구사항](#환경-요구사항)
2. [프로젝트 클론 및 설치](#프로젝트-클론-및-설치)
3. [데이터 구조 확인](#데이터-구조-확인)
4. [전체 파이프라인 실행](#전체-파이프라인-실행)
5. [생성된 결과물](#생성된-결과물)
6. [트러블슈팅](#트러블슈팅)

---

## 🖥️ 환경 요구사항

### 필수 소프트웨어

- **Python**: 3.8 이상
- **pip**: 최신 버전
- **Git**: 프로젝트 클론용
- **브라우저**: Chrome, Firefox, Safari 등 (웹 UI 및 결과 확인용)

### macOS 추가 요구사항

XGBoost 실행을 위해 OpenMP 라이브러리 필요:

```bash
brew install libomp
```

### 권장 사양

- **RAM**: 최소 8GB (16GB 권장)
- **디스크**: 최소 1GB 여유 공간
- **CPU**: 멀티코어 프로세서 권장 (병렬 처리)

---

## 📥 프로젝트 클론 및 설치

### 1. 프로젝트 클론

```bash
# GitHub에서 저장소 클론
git clone https://github.com/bosoa/Lifelog_Pattern_Data_Generation.git

# 프로젝트 디렉토리로 이동
cd Lifelog_Pattern_Data_Generation
```

### 2. Python 패키지 설치

```bash
# requirements.txt의 모든 패키지 설치
pip install -r requirements.txt
```

**설치되는 주요 패키지**:
- `pandas`: 데이터 처리
- `numpy`: 수치 연산
- `scikit-learn`: 머신러닝 (PCA, 클러스터링, 모델)
- `matplotlib`, `seaborn`, `plotly`: 시각화
- `streamlit`: 웹 UI
- `xgboost`: XGBoost 모델
- `torch`: Neural Network
- `lifelines`: Cox Proportional Hazards 생존 분석

### 3. 설치 확인

```bash
# Python 버전 확인
python3 --version  # Python 3.8 이상이어야 함

# 패키지 설치 확인
python3 -c "import pandas, sklearn, streamlit, xgboost, lifelines; print('✅ 모든 패키지 설치 완료!')"
```

---

## 📂 데이터 구조 확인

### 클론 후 데이터 확인

프로젝트를 클론하면 다음 데이터가 포함되어 있습니다:

```
Lifelog_Pattern_Data_Generation/
├── KLOSDOM_Preprocessed_Dataset/    # 원본 데이터 (22개 CSV)
│   ├── whole_a*.csv                 # 센서 데이터 (18개)
│   └── whole_e*.csv                 # 설문 데이터 (4개)
├── hierarchical_data/                # 계층화 데이터 (생성됨)
│   ├── *_binary_classification.csv
│   ├── *_hierarchical_data.csv
│   └── ... (각 종속변수당 8개 파일)
├── data_splits/                      # 분할된 데이터 (생성됨)
│   ├── anxiety_train.csv
│   ├── anxiety_val.csv
│   ├── anxiety_test.csv
│   └── ... (각 종속변수당 3개 파일)
└── model_results/                    # 분석 결과 HTML (생성됨)
    ├── index.html                   # 통합 대시보드
    ├── *_model_comparison_report.html
    ├── *_survival_analysis_report.html
    └── data_distribution_report.html
```

### 데이터 확인 명령어

```bash
# 원본 데이터 파일 개수 확인
ls KLOSDOM_Preprocessed_Dataset/*.csv | wc -l
# 결과: 22 (18개 센서 + 4개 설문)

# 계층화 데이터 확인
ls hierarchical_data/*.csv | wc -l
# 결과: 27 (각 종속변수당 8~9개 파일)

# 데이터 분할 확인
ls data_splits/*.csv | wc -l
# 결과: 9 (각 종속변수당 3개: train, val, test)

# 분석 결과 HTML 확인
ls model_results/*.html | wc -l
# 결과: 13 (통합 대시보드 + 개별 리포트)
```

---

## 🚀 전체 파이프라인 실행

### 옵션 1: 통합 대시보드만 확인 (가장 빠름)

이미 생성된 결과물이 있으므로 바로 확인 가능:

```bash
# 통합 대시보드 열기
open model_results/index.html

# 또는 브라우저에서 직접 파일 열기
# Chrome: Cmd+O → index.html 선택
# Firefox: Ctrl+O → index.html 선택
```

### 옵션 2: PCA 분석 웹 UI 실행

대화형 주성분 분석 인터페이스:

```bash
# 방법 1: 쉘 스크립트 사용
./run_pca_app.sh

# 방법 2: Streamlit 직접 실행
streamlit run src/app.py
```

브라우저가 자동으로 `http://localhost:8501`을 엽니다.

**기능**:
- 종속변수 선택 (불안/우울/스트레스)
- PCA 파라미터 조정 (주성분 개수, 필터 기준)
- 5가지 시각화 (Scree Plot, 2D/3D Biplot, 히트맵)
- 결과 CSV 다운로드

### 옵션 3: 전체 파이프라인 재실행

데이터를 처음부터 다시 생성하려면:

#### 1단계: 계층화 데이터 생성

```bash
python3 src/hierarchical_data_generator.py
```

**소요 시간**: 약 2~3분  
**출력**: `hierarchical_data/` 디렉토리에 24개 파일 생성

#### 2단계: 이진 분류 변환

```bash
python3 src/binary_classification_converter.py
```

**소요 시간**: 약 10초  
**출력**: `*_binary_classification.csv` 파일 (3개)

#### 3단계: 데이터 분할 (Train/Val/Test)

```bash
python3 src/data_splitter.py
```

**소요 시간**: 약 5초  
**출력**: `data_splits/` 디렉토리에 9개 파일 생성

#### 4단계: 모델 비교 (RandomForest, XGBoost)

```bash
python3 run_model_comparison.py
```

**소요 시간**: 약 3~5분  
**출력**: `model_results/*_model_comparison_report.html` (3개)

#### 5단계: 데이터 분포 시각화

```bash
python3 src/data_distribution_visualizer.py
```

**소요 시간**: 약 30초  
**출력**: `model_results/data_distribution_report.html`

#### 6단계: 생존 분석 (Cox PH)

```bash
# 전체 변수 생존 분석
python3 src/survival_analysis.py

# p-value 필터링 생존 분석
python3 src/survival_analysis_filtered.py
```

**소요 시간**: 각 약 1~2분  
**출력**: `model_results/*_survival_analysis*.html` (6개)

#### 7단계: 통합 대시보드 확인

```bash
open model_results/index.html
```

---

## 📊 생성된 결과물

### 1. 계층화 데이터 (`hierarchical_data/`)

각 종속변수(anxiety, depression, stress)당 8개 파일:

| 파일명 | 설명 | 크기 |
|--------|------|------|
| `*_hierarchical_data.csv` | 전체 계층화 데이터 (10개 변수 + 레벨 + 타겟) | ~2MB |
| `*_level_statistics.csv` | 계층별 통계 (평균, 표준편차, 샘플 수) | ~1KB |
| `*_feature_importance.csv` | 변수 중요도 순위 | ~1KB |
| `*_selected_features.csv` | 선택된 10개 주요 변수 목록 | <1KB |
| `*_level_0_낮은_활동.csv` | 레벨 0 데이터 (~70% 샘플) | ~1.5MB |
| `*_level_1_중간_활동.csv` | 레벨 1 데이터 (~7% 샘플) | ~150KB |
| `*_level_2_높은_활동.csv` | 레벨 2 데이터 (~21% 샘플) | ~500KB |
| `*_summary_report.txt` | 텍스트 요약 리포트 | ~5KB |

**추가 파일**:
- `*_binary_classification.csv`: 이진 분류 데이터 (3개)
- `binary_classification_summary.csv`: 이진 분류 요약

**총 크기**: ~20MB

### 2. 분할 데이터 (`data_splits/`)

각 종속변수당 3개 파일 (7:2:1 비율):

| 파일명 | 샘플 수 (예: anxiety) | 비율 |
|--------|----------------------|------|
| `*_train.csv` | ~10,280개 | 70% |
| `*_val.csv` | ~2,937개 | 20% |
| `*_test.csv` | ~1,468개 | 10% |

**추가 파일**:
- `split_summary.csv`: 분할 요약

**총 크기**: ~15MB

### 3. 분석 결과 HTML (`model_results/`)

| 파일명 | 설명 | 주요 내용 |
|--------|------|----------|
| **`index.html`** ⭐ | **통합 대시보드** | 모든 분석 결과 통합 |
| `data_distribution_report.html` | 데이터 분포 분석 | 히스토그램, 박스플롯, 바이올린 플롯 |
| `anxiety_model_comparison_report.html` | 불안 모델 비교 | RandomForest vs XGBoost |
| `depression_model_comparison_report.html` | 우울 모델 비교 | 성능 지표, 특징 중요도 |
| `stress_model_comparison_report.html` | 스트레스 모델 비교 | 예측 vs 실제 플롯 |
| `anxiety_survival_analysis_report.html` | 불안 생존 분석 | Cox PH, Nomogram, Calibration |
| `depression_survival_analysis_report.html` | 우울 생존 분석 | Kaplan-Meier, Hazard Ratios |
| `stress_survival_analysis_report.html` | 스트레스 생존 분석 | C-index, LR Test |
| `anxiety_survival_analysis_filtered_report.html` | 불안 필터링 생존 분석 | p<0.05 변수만 사용 |
| `depression_survival_analysis_filtered_report.html` | 우울 필터링 생존 분석 | 개선된 모델 성능 |
| `stress_survival_analysis_filtered_report.html` | 스트레스 필터링 생존 분석 | 변수 선택 전후 비교 |

**총 크기**: ~18MB (내장된 base64 이미지 포함)

### 4. 주요 결과 요약

#### 변수 중요도 (상위 10개)

1. **skin_temperature** (0.152) - 피부 온도
2. **body_temperature** (0.146) - 체온
3. **total_sleep** (0.145) - 총 수면 시간
4. **blood_sugar** (0.145) - 혈당
5. **light_sleep** (0.140) - 얕은 수면
6. **walk** (0.136) - 걸음수
7. **deep_sleep** (0.116) - 깊은 수면
8. **rem_sleep** (0.115) - REM 수면
9. **stick_sensor** (0.101) - 지팡이 센서
10. **hrv** (0.089) - 심박변이도

#### 모델 성능 (R² Score)

**RandomForest**:
- Anxiety: 0.3451
- Depression: 0.3658
- Stress: 0.3423

**XGBoost**:
- Anxiety: 0.3592
- Depression: 0.3801
- Stress: 0.3565

#### 생존 분석 (C-index)

**전체 변수**:
- Anxiety: 0.6807
- Depression: 0.6971
- Stress: 0.6751

**필터링 변수 (p<0.05)**:
- Anxiety: 0.6838 ↑
- Depression: 0.6998 ↑
- Stress: 0.6795 ↑

#### 이진 분류 발생률 (≥4점)

- Depression: 16.3%
- Anxiety: 15.0%
- Stress: 24.6%

---

## 🔧 트러블슈팅

### 문제 1: libomp 오류 (macOS XGBoost)

**증상**:
```
Library not loaded: @rpath/libomp.dylib
```

**해결**:
```bash
brew install libomp
```

### 문제 2: PCA NaN 오류

**증상**:
```
Input X contains NaN. PCA does not accept missing values
```

**원인**: 결측치 처리 실패

**해결**: 이미 4단계 NaN 처리가 구현되어 있음 (`data_loader.py`)
```python
# 1단계: 평균값 채우기
# 2단계: 0으로 채우기
# 3단계: inf 제거
# 4단계: NaN 행 삭제
```

### 문제 3: Streamlit 포트 충돌

**증상**:
```
Port 8501 is already in use
```

**해결**:
```bash
# 다른 포트 사용
streamlit run src/app.py --server.port 8502

# 또는 기존 프로세스 종료
lsof -ti:8501 | xargs kill -9
```

### 문제 4: 메모리 부족

**증상**:
프로그램이 중간에 멈추거나 느려짐

**해결**:
```python
# hierarchical_data_generator.py 수정
# min_data_points를 증가시켜 샘플 수 감소
generator = HierarchicalDataGenerator(min_data_points=50)  # 기본값: 10
```

### 문제 5: 패키지 버전 충돌

**증상**:
```
ImportError: cannot import name 'xxx'
```

**해결**:
```bash
# 가상환경 생성 (권장)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# requirements.txt 재설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 문제 6: Git LFS 대용량 파일

**증상**:
CSV 파일이 제대로 다운로드되지 않음

**해결**:
```bash
# Git LFS 설치
brew install git-lfs  # macOS
# apt install git-lfs  # Ubuntu

# LFS 초기화 및 파일 다운로드
git lfs install
git lfs pull
```

### 문제 7: JavaScript 버튼 미작동

**증상**:
대시보드에서 탭 버튼 클릭이 작동하지 않음

**해결**:
브라우저 캐시 삭제 후 새로고침 (Cmd+Shift+R / Ctrl+Shift+F5)

---

## 🎓 학습 자료

### 문서

- **README.md**: 프로젝트 개요 및 데이터 구조
- **PCA_ANALYSIS_GUIDE.md**: PCA 분석 상세 가이드
- **SETUP.md** (이 문서): 설치 및 실행 가이드

### 코드 구조

```
src/
├── data_loader.py              # 데이터 로드 및 전처리
├── pca_analyzer.py             # PCA 분석 및 시각화
├── pattern_analyzer.py         # 패턴 분석 및 클러스터링
├── hierarchical_data_generator.py  # 계층화 데이터 생성 (메인)
├── binary_classification_converter.py  # 이진 분류 변환
├── data_splitter.py            # Train/Val/Test 분할
├── model_comparison.py         # 모델 비교 (RF, XGBoost)
├── survival_analysis.py        # Cox PH 생존 분석
├── survival_analysis_filtered.py  # p-value 필터링 생존 분석
├── data_distribution_visualizer.py  # 데이터 분포 시각화
└── app.py                      # Streamlit 웹 UI
```

### 실행 스크립트

```
run_pca_app.sh              # PCA 웹 UI 실행
run_model_comparison.py     # 모델 비교 실행
```

---

## 📞 지원

문제가 발생하면:

1. **GitHub Issues**: https://github.com/bosoa/Lifelog_Pattern_Data_Generation/issues
2. **이메일**: bosoagalaxy@gmail.com

---

**Last Updated**: 2026-06-21
