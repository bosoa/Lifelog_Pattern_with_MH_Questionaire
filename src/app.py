"""
PCA 분석 Web UI - Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from data_loader import KLOSDOMDataLoader
from pca_analyzer import PCAAnalyzer


# 페이지 설정
st.set_page_config(
    page_title="KLOSDOM PCA 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data(target_variable: str, min_data_points: int):
    """데이터 로드 (캐싱)"""
    loader = KLOSDOMDataLoader()
    X, y, feature_names = loader.prepare_pca_data(
        target_variable=target_variable,
        min_data_points=min_data_points
    )
    return X, y, feature_names


@st.cache_data
def perform_pca(X: pd.DataFrame, feature_names: list, n_components: int):
    """PCA 수행 (캐싱)"""
    analyzer = PCAAnalyzer(n_components=n_components)
    analyzer.fit_transform(X, feature_names)
    return analyzer


def main():
    st.title("🧠 KLOSDOM 정신건강 예측 - PCA 분석")
    st.markdown("""
    생체신호 센서 데이터(18개 독립변수)를 주성분 분석(PCA)하여
    정신건강 지표(불안, 우울, 스트레스)와의 관계를 시각화합니다.
    """)

    # 사이드바 설정
    st.sidebar.header("⚙️ 분석 설정")

    # 종속변수 선택
    target_variable = st.sidebar.selectbox(
        "분석할 종속변수 선택",
        options=['anxiety', 'depression', 'stress'],
        format_func=lambda x: {
            'anxiety': '불안 (Anxiety)',
            'depression': '우울 (Depression)',
            'stress': '스트레스 (Stress)'
        }[x],
        index=0
    )

    # PCA 설정
    n_components = st.sidebar.slider(
        "주성분 개수",
        min_value=2,
        max_value=18,
        value=10,
        help="분석에 사용할 주성분의 개수"
    )

    min_data_points = st.sidebar.slider(
        "최소 데이터 포인트 수",
        min_value=5,
        max_value=30,
        value=10,
        help="사용자별 최소 데이터 포인트 수"
    )

    # 분석 시작 버튼
    if st.sidebar.button("🚀 분석 시작", type="primary"):
        st.session_state.run_analysis = True

    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False

    # 초기 화면
    if not st.session_state.run_analysis:
        st.info("👈 왼쪽 사이드바에서 설정을 조정하고 '분석 시작' 버튼을 클릭하세요.")

        # 데이터셋 정보 표시
        st.subheader("📁 데이터셋 정보")
        try:
            loader = KLOSDOMDataLoader()
            info = loader.get_dataset_info()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 사용자 수", f"{info['n_users']:,}명")
            with col2:
                st.metric("측정 기간", f"{info['n_dates']}일")
            with col3:
                st.metric("센서 종류", f"{info['n_sensors']}개")
            with col4:
                st.metric("설문 항목", f"{info['n_surveys']}개")

            # 센서 목록
            with st.expander("📊 독립변수 (센서 데이터) 목록"):
                sensor_df = pd.DataFrame({
                    '번호': range(1, len(info['sensor_names']) + 1),
                    '센서명': info['sensor_names']
                })
                st.dataframe(sensor_df, use_container_width=True)

        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {str(e)}")

        return

    # 분석 실행
    st.markdown("---")

    # 데이터 로드
    with st.spinner(f"데이터 로드 중... (종속변수: {target_variable})"):
        try:
            X, y, feature_names = load_data(target_variable, min_data_points)

            # 데이터 통계
            st.subheader(f"📈 데이터 통계 - {target_variable.upper()}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("샘플 수", f"{len(X):,}")
            with col2:
                st.metric("특징 수", len(feature_names))
            with col3:
                st.metric("타겟 평균", f"{y.mean():.2f}")
            with col4:
                st.metric("타겟 표준편차", f"{y.std():.2f}")

        except Exception as e:
            st.error(f"데이터 로드 실패: {str(e)}")
            return

    # PCA 수행
    with st.spinner("PCA 분석 중..."):
        try:
            analyzer = perform_pca(X, feature_names, n_components)
            stats = analyzer.get_summary_stats()

            st.success("✅ PCA 분석 완료!")

            # PCA 결과 요약
            st.subheader("📊 PCA 결과 요약")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("주성분 개수", stats['n_components'])
            with col2:
                st.metric("총 설명 분산", f"{stats['total_variance_explained']*100:.1f}%")
            with col3:
                st.metric("90% 설명에 필요한 PC 수", stats['n_components_90'])

        except Exception as e:
            st.error(f"PCA 분석 실패: {str(e)}")
            return

    # 탭으로 시각화 구분
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Scree Plot",
        "🎯 2D Biplot",
        "🌐 3D Plot",
        "🔥 로딩 히트맵",
        "📋 상세 결과"
    ])

    with tab1:
        st.subheader("Scree Plot - 주성분별 설명 분산")
        st.markdown("""
        각 주성분이 설명하는 분산의 비율을 보여줍니다.
        왼쪽: 개별 주성분의 설명력 | 오른쪽: 누적 설명력
        """)
        scree_fig = analyzer.plot_scree()
        st.plotly_chart(scree_fig, use_container_width=True)

        # 설명 분산 테이블
        with st.expander("📊 설명 분산 상세 데이터"):
            explained_var, cumulative_var = analyzer.get_explained_variance()
            var_df = pd.DataFrame({
                '주성분': [f'PC{i+1}' for i in range(len(explained_var))],
                '설명 분산 (%)': explained_var * 100,
                '누적 설명 분산 (%)': cumulative_var * 100
            })
            st.dataframe(var_df, use_container_width=True)

    with tab2:
        st.subheader("2D Biplot - PC1 vs PC2")
        st.markdown("""
        - **점**: 개별 샘플 (색상 = 타겟 점수)
        - **빨간 화살표**: 각 센서 변수의 기여도 방향
        - 화살표가 길수록 해당 변수의 영향력이 큼
        """)
        biplot_fig = analyzer.plot_biplot_2d(
            y=y,
            title=f"{target_variable.upper()} - PCA Biplot (PC1 vs PC2)"
        )
        st.plotly_chart(biplot_fig, use_container_width=True)

    with tab3:
        st.subheader("3D 산점도 - PC1 vs PC2 vs PC3")
        st.markdown("상위 3개 주성분으로 구성된 3차원 공간에서의 데이터 분포")
        scatter_3d_fig = analyzer.plot_3d_scatter(
            y=y,
            title=f"{target_variable.upper()} - 3D PCA Plot"
        )
        st.plotly_chart(scatter_3d_fig, use_container_width=True)

    with tab4:
        st.subheader("주성분 로딩 히트맵")
        st.markdown("""
        각 센서 변수가 주성분에 미치는 영향력을 색상으로 표시합니다.
        - **빨간색**: 양의 기여도
        - **파란색**: 음의 기여도
        """)

        n_pc_heatmap = st.slider(
            "표시할 주성분 개수",
            min_value=3,
            max_value=min(10, n_components),
            value=5,
            key="heatmap_slider"
        )

        heatmap_fig = analyzer.plot_loadings_heatmap(n_components=n_pc_heatmap)
        st.plotly_chart(heatmap_fig, use_container_width=True)

    with tab5:
        st.subheader("📋 주성분별 상위 기여 변수")

        # 주성분 선택
        pc_options = [f'PC{i+1}' for i in range(min(5, n_components))]
        selected_pc = st.selectbox(
            "주성분 선택",
            options=pc_options,
            index=0
        )

        pc_index = int(selected_pc.replace('PC', '')) - 1

        # 상위 변수 표시
        n_top = st.slider("표시할 변수 개수", 5, 18, 10, key="top_features_slider")
        top_features = analyzer.get_top_features(pc_index=pc_index, n_top=n_top)

        st.dataframe(top_features, use_container_width=True)

        # 전체 로딩 행렬
        with st.expander("🔍 전체 로딩 행렬 보기"):
            loadings_df = analyzer.get_loadings()
            st.dataframe(loadings_df, use_container_width=True)

    # 다운로드 섹션
    st.markdown("---")
    st.subheader("💾 결과 다운로드")

    col1, col2 = st.columns(2)

    with col1:
        # 로딩 행렬 다운로드
        loadings_csv = analyzer.get_loadings().to_csv()
        st.download_button(
            label="📥 로딩 행렬 다운로드 (CSV)",
            data=loadings_csv,
            file_name=f"pca_loadings_{target_variable}.csv",
            mime="text/csv"
        )

    with col2:
        # 변환된 주성분 데이터 다운로드
        pca_data = pd.DataFrame(
            analyzer.X_pca,
            columns=[f'PC{i+1}' for i in range(analyzer.X_pca.shape[1])]
        )
        pca_data[f'{target_variable}_score'] = y.values
        pca_csv = pca_data.to_csv(index=False)
        st.download_button(
            label="📥 주성분 데이터 다운로드 (CSV)",
            data=pca_csv,
            file_name=f"pca_transformed_{target_variable}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
