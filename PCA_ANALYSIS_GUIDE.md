# PCA 분석 가이드

## 개요

본 프로젝트는 KLOSDOM 데이터셋의 18개 센서 데이터(독립변수)를 주성분 분석(PCA)하여 정신건강 지표(불안, 우울, 스트레스)와의 관계를 탐색합니다.

## 시스템 구성

### 📁 파일 구조

```
Lifelog_Pattern_Data_Generation/
├── src/
│   ├── data_loader.py      # 데이터 로드 및 전처리 모듈
│   ├── pca_analyzer.py     # PCA 분석 및 시각화 모듈
│   └── app.py              # Streamlit 웹 애플리케이션
├── KLOSDOM_Preprocessed_Dataset/  # 원본 데이터
├── requirements.txt         # 필요한 Python 패키지
├── run_pca_app.sh          # 웹 앱 실행 스크립트
└── PCA_ANALYSIS_GUIDE.md   # 이 파일
```

## 설치 및 실행

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 웹 앱 실행

#### 방법 1: 쉘 스크립트 사용 (권장)
```bash
./run_pca_app.sh
```

#### 방법 2: 직접 실행
```bash
streamlit run src/app.py
```

### 3. 브라우저 접속

웹 앱이 시작되면 자동으로 브라우저가 열리거나, 다음 주소로 접속합니다:
```
http://localhost:8501
```

## 사용 방법

### 📊 분석 설정

1. **종속변수 선택**
   - 불안 (Anxiety)
   - 우울 (Depression)
   - 스트레스 (Stress)

2. **주성분 개수 설정**
   - 최소: 2개
   - 최대: 18개 (전체 센서 수)
   - 권장: 10개 (총 분산의 대부분을 설명)

3. **최소 데이터 포인트 수**
   - 사용자별 최소 측정 횟수
   - 권장: 10회 이상

4. **분석 시작** 버튼 클릭

### 📈 시각화 탭

#### 1️⃣ Scree Plot
- **목적**: 각 주성분의 설명력 확인
- **해석**:
  - 왼쪽 그래프: 개별 주성분의 설명 분산
  - 오른쪽 그래프: 누적 설명 분산
  - "Elbow" 지점을 찾아 필요한 주성분 수 결정

#### 2️⃣ 2D Biplot
- **목적**: PC1과 PC2로 데이터 분포 및 변수 기여도 시각화
- **구성 요소**:
  - **점**: 개별 샘플 (색상 = 타겟 점수)
  - **빨간 화살표**: 각 센서 변수의 방향과 크기
- **해석**:
  - 화살표가 길수록 해당 변수의 영향력이 큼
  - 같은 방향을 가리키는 화살표는 상관관계가 높음
  - 색상이 진한 영역은 타겟 점수가 높음

#### 3️⃣ 3D Plot
- **목적**: 상위 3개 주성분으로 데이터 분포 입체 시각화
- **조작**: 마우스 드래그로 회전, 스크롤로 줌

#### 4️⃣ 로딩 히트맵
- **목적**: 각 센서가 주성분에 미치는 영향력을 색상으로 표시
- **색상 의미**:
  - 빨간색: 양의 기여도 (값이 클수록 PC 값 증가)
  - 파란색: 음의 기여도 (값이 클수록 PC 값 감소)
- **활용**: 각 주성분이 어떤 센서들의 조합인지 파악

#### 5️⃣ 상세 결과
- **주성분별 상위 기여 변수**: 특정 PC에 가장 영향을 많이 미치는 센서들
- **전체 로딩 행렬**: 모든 센서 × 모든 주성분의 기여도

### 💾 결과 다운로드

1. **로딩 행렬 (CSV)**
   - 각 센서가 각 주성분에 미치는 영향력
   - 추가 분석에 활용 가능

2. **주성분 데이터 (CSV)**
   - 변환된 주성분 값 + 타겟 점수
   - 머신러닝 모델의 입력 데이터로 사용 가능

## PCA 결과 해석 가이드

### 1. 설명 분산 분석

- **90% 이상 설명**: 충분한 정보 보존
- **70-90% 설명**: 일부 정보 손실, 차원 축소 효과
- **70% 미만**: 정보 손실이 큼, 주성분 개수 증가 고려

### 2. 주성분 해석

#### PC1 (첫 번째 주성분)
- 가장 큰 분산을 설명
- 일반적으로 "전반적인 건강 상태" 또는 "활동 수준"을 나타냄
- 로딩 값이 높은 변수들을 확인하여 의미 파악

#### PC2 (두 번째 주성분)
- PC1과 독립적인 두 번째로 큰 분산
- 종종 "수면 패턴" 또는 "스트레스 지표"를 나타냄

#### PC3 이상
- 점진적으로 설명력이 감소
- 특정 생리 패턴이나 노이즈를 포함할 수 있음

### 3. 변수 기여도 분석

#### 높은 로딩 (|값| > 0.5)
- 해당 주성분을 강하게 대표하는 변수
- 주성분의 의미를 해석하는 핵심 센서

#### 낮은 로딩 (|값| < 0.2)
- 해당 주성분과 관련성이 낮음
- 다른 주성분에서 중요할 수 있음

### 4. 종속변수별 패턴 비교

#### 불안 (Anxiety)
- 심박변이도(HRV), 심박수 관련 PC가 중요할 가능성
- 스크린 타임, 활동량과의 관계 확인

#### 우울 (Depression)
- 수면 패턴 관련 PC가 중요할 가능성
- 활동량, 이동 거리와의 관계 확인

#### 스트레스 (Stress)
- 생리적 지표(체온, 피부 온도) 관련 PC가 중요할 가능성
- 수면의 질, 활동 패턴과의 관계 확인

## 코드 구조 설명

### data_loader.py

**주요 클래스**: `KLOSDOMDataLoader`

**주요 메서드**:
- `load_sensor_data()`: 18개 센서 데이터 로드
- `load_survey_data()`: 4개 설문 데이터 로드
- `prepare_pca_data()`: PCA를 위한 데이터 전처리
  - Wide → Long format 변환
  - 결측치 처리
  - 독립변수/종속변수 분리

### pca_analyzer.py

**주요 클래스**: `PCAAnalyzer`

**주요 메서드**:
- `fit_transform()`: PCA 수행
- `get_explained_variance()`: 설명 분산 계산
- `get_loadings()`: 주성분 로딩 행렬
- `plot_scree()`: Scree plot 생성
- `plot_biplot_2d()`: 2D Biplot 생성
- `plot_3d_scatter()`: 3D 산점도 생성
- `plot_loadings_heatmap()`: 로딩 히트맵 생성

### app.py

**Streamlit 웹 애플리케이션**

**주요 기능**:
- 사이드바: 분석 설정
- 메인 화면: 5개 탭으로 구성된 시각화
- 캐싱: 데이터 로드 및 PCA 결과 캐싱으로 성능 최적화

## 고급 사용법

### Python 스크립트로 직접 사용

```python
from src.data_loader import KLOSDOMDataLoader
from src.pca_analyzer import PCAAnalyzer

# 데이터 로드
loader = KLOSDOMDataLoader()
X, y, feature_names = loader.prepare_pca_data(
    target_variable='anxiety',
    min_data_points=10
)

# PCA 수행
analyzer = PCAAnalyzer(n_components=10)
analyzer.fit_transform(X, feature_names)

# 결과 확인
stats = analyzer.get_summary_stats()
print(stats)

# 시각화
scree_fig = analyzer.plot_scree()
scree_fig.show()

# 로딩 행렬
loadings = analyzer.get_loadings()
print(loadings)
```

### 배치 분석 (모든 종속변수)

```python
targets = ['anxiety', 'depression', 'stress']

for target in targets:
    print(f"\n분석 중: {target}")
    X, y, feature_names = loader.prepare_pca_data(target)
    analyzer = PCAAnalyzer(n_components=10)
    analyzer.fit_transform(X, feature_names)
    
    # 결과 저장
    loadings = analyzer.get_loadings()
    loadings.to_csv(f'pca_loadings_{target}.csv')
```

## 문제 해결

### 오류: ModuleNotFoundError

```bash
# 필요한 패키지가 설치되지 않음
pip install -r requirements.txt
```

### 오류: FileNotFoundError

```bash
# 데이터 디렉토리 경로 확인
# 스크립트는 프로젝트 루트에서 실행되어야 함
cd /path/to/Lifelog_Pattern_Data_Generation
./run_pca_app.sh
```

### 성능 이슈 (느린 로딩)

- `min_data_points`를 높여서 샘플 수 감소
- 브라우저 캐시 및 Streamlit 캐시 활용
- 주성분 개수 감소

### 메모리 부족

- 데이터 크기가 큰 경우, 일부 센서만 선택하여 분석
- `data_loader.py`에서 필요한 센서만 로드하도록 수정

## 참고 자료

### PCA 이론
- [Scikit-learn PCA 문서](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [PCA 설명 (StatQuest)](https://www.youtube.com/watch?v=FgakZw6K1QQ)

### Streamlit
- [Streamlit 공식 문서](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)

### Plotly
- [Plotly Python 문서](https://plotly.com/python/)
- [Plotly 예제](https://plotly.com/python/plotly-express/)

## 라이선스

(프로젝트 라이선스에 따름)

---

**Last Updated**: 2026-06-21
