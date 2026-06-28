# 빠른 시작 가이드

**5분 안에 프로젝트 실행하기**

## 🚀 최소 단계로 시작하기

### 1단계: 프로젝트 준비 (1분)

```bash
# 저장소 클론
git clone https://github.com/bosoa/Lifelog_Pattern_Data_Generation.git
cd Lifelog_Pattern_Data_Generation

# 패키지 설치
pip install -r requirements.txt

# macOS 사용자는 추가 설치
brew install libomp
```

### 2단계: 결과 확인 (즉시)

이미 생성된 분석 결과를 바로 확인:

```bash
# 통합 대시보드 열기
open model_results/index.html
```

**대시보드에서 확인 가능한 내용**:
- 🏠 프로젝트 개요
- 📊 데이터 분포 분석
- 🤖 모델 비교 (RandomForest vs XGBoost)
- 🏥 생존 분석 (Cox PH, Nomogram)
- 🎯 필터링 생존 분석 (p<0.05)

---

## 🎯 주요 작업별 실행 명령어

### PCA 분석 웹 UI

대화형 주성분 분석:

```bash
./run_pca_app.sh
```

브라우저에서 `http://localhost:8501` 자동 열림

**기능**:
- 종속변수 선택
- 파라미터 조정
- 5가지 시각화
- CSV 다운로드

### 전체 파이프라인 재실행

모든 데이터를 처음부터 생성:

```bash
# 1. 계층화 데이터 생성 (2분)
python3 src/hierarchical_data_generator.py

# 2. 이진 분류 변환 (10초)
python3 src/binary_classification_converter.py

# 3. 데이터 분할 (5초)
python3 src/data_splitter.py

# 4. 모델 비교 (3분)
python3 run_model_comparison.py

# 5. 데이터 분포 (30초)
python3 src/data_distribution_visualizer.py

# 6. 생존 분석 (2분)
python3 src/survival_analysis.py
python3 src/survival_analysis_filtered.py

# 7. 결과 확인
open model_results/index.html
```

---

## 📊 주요 결과 위치

| 결과 | 파일 경로 | 설명 |
|------|----------|------|
| **통합 대시보드** | `model_results/index.html` | 모든 분석 결과 한눈에 보기 |
| **계층화 데이터** | `hierarchical_data/*.csv` | PCA 기반 10개 변수 선택 데이터 |
| **분할 데이터** | `data_splits/*.csv` | Train/Val/Test (7:2:1) |
| **모델 비교** | `model_results/*_model_comparison_report.html` | RF vs XGBoost |
| **생존 분석** | `model_results/*_survival_analysis_report.html` | Cox PH, Nomogram |

---

## 🔑 핵심 발견

### 주요 변수 (상위 5개)

1. **skin_temperature** (0.152) - 피부 온도
2. **body_temperature** (0.146) - 체온
3. **total_sleep** (0.145) - 총 수면 시간
4. **blood_sugar** (0.145) - 혈당
5. **light_sleep** (0.140) - 얕은 수면

### 모델 성능

**XGBoost** (최고 성능):
- Anxiety R²: 0.359
- Depression R²: 0.380
- Stress R²: 0.357

**Cox PH C-index** (필터링):
- Anxiety: 0.684
- Depression: 0.700
- Stress: 0.680

### 계층 분포

- **레벨 0** (70%): 낮은 활동, 수면 6.3시간
- **레벨 1** (7%): 극심한 수면 부족 1.9시간 ⚠️
- **레벨 2** (21%): 높은 활동, 수면 8.0시간

---

## 🆘 문제 해결

### libomp 오류 (macOS)

```bash
brew install libomp
```

### 포트 충돌 (Streamlit)

```bash
streamlit run src/app.py --server.port 8502
```

### 패키지 오류

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📚 더 알아보기

- **상세 설정**: [SETUP.md](SETUP.md)
- **프로젝트 개요**: [README.md](README.md)
- **PCA 가이드**: [PCA_ANALYSIS_GUIDE.md](PCA_ANALYSIS_GUIDE.md)

---

**최초 작성**: 2026-06-21
