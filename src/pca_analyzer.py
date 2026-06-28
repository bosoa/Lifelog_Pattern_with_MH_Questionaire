"""
PCA 분석 모듈
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class PCAAnalyzer:
    """PCA 분석 및 시각화 클래스"""

    def __init__(self, n_components: int = None):
        """
        Args:
            n_components: 주성분 개수 (None이면 전체 특징 수)
        """
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = None
        self.X_scaled = None
        self.X_pca = None
        self.feature_names = None

    def fit_transform(
        self,
        X: pd.DataFrame,
        feature_names: List[str]
    ) -> np.ndarray:
        """
        PCA 수행

        Args:
            X: 입력 데이터
            feature_names: 특징 이름 리스트

        Returns:
            변환된 주성분 데이터
        """
        self.feature_names = feature_names

        # 데이터 검증
        if X.isna().any().any():
            raise ValueError(
                f"입력 데이터에 NaN 값이 포함되어 있습니다. "
                f"NaN 개수: {X.isna().sum().sum()}"
            )

        if np.isinf(X.values).any():
            raise ValueError("입력 데이터에 무한대(inf) 값이 포함되어 있습니다.")

        # 데이터 정규화
        self.X_scaled = self.scaler.fit_transform(X)

        # PCA 수행
        n_comp = self.n_components or len(feature_names)
        self.pca = PCA(n_components=n_comp)
        self.X_pca = self.pca.fit_transform(self.X_scaled)

        return self.X_pca

    def get_explained_variance(self) -> Tuple[np.ndarray, np.ndarray]:
        """설명 분산 반환"""
        if self.pca is None:
            raise ValueError("PCA를 먼저 수행해야 합니다.")

        return (
            self.pca.explained_variance_ratio_,
            np.cumsum(self.pca.explained_variance_ratio_)
        )

    def get_loadings(self) -> pd.DataFrame:
        """주성분 로딩 반환"""
        if self.pca is None:
            raise ValueError("PCA를 먼저 수행해야 합니다.")

        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_names
        )
        return loadings

    def plot_scree(self) -> go.Figure:
        """Scree plot 생성"""
        explained_var, cumulative_var = self.get_explained_variance()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('개별 설명 분산', '누적 설명 분산'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 개별 설명 분산
        fig.add_trace(
            go.Bar(
                x=[f'PC{i+1}' for i in range(len(explained_var))],
                y=explained_var * 100,
                name='설명 분산',
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        # 누적 설명 분산
        fig.add_trace(
            go.Scatter(
                x=[f'PC{i+1}' for i in range(len(cumulative_var))],
                y=cumulative_var * 100,
                mode='lines+markers',
                name='누적 설명 분산',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ),
            row=1, col=2
        )

        fig.update_xaxes(title_text="주성분", row=1, col=1)
        fig.update_xaxes(title_text="주성분", row=1, col=2)
        fig.update_yaxes(title_text="설명 분산 (%)", row=1, col=1)
        fig.update_yaxes(title_text="누적 설명 분산 (%)", row=1, col=2)

        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Scree Plot - 주성분 설명력"
        )

        return fig

    def plot_biplot_2d(self, y: pd.Series = None, title: str = "PCA Biplot") -> go.Figure:
        """2D Biplot 생성"""
        if self.X_pca is None:
            raise ValueError("PCA를 먼저 수행해야 합니다.")

        # 주성분 데이터
        pc1 = self.X_pca[:, 0]
        pc2 = self.X_pca[:, 1]

        # 타겟 변수가 있으면 색상으로 표시
        if y is not None:
            fig = go.Figure(data=go.Scatter(
                x=pc1,
                y=pc2,
                mode='markers',
                marker=dict(
                    size=5,
                    color=y,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="타겟 점수")
                ),
                text=[f'Score: {val:.1f}' for val in y],
                hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>%{text}<extra></extra>'
            ))
        else:
            fig = go.Figure(data=go.Scatter(
                x=pc1,
                y=pc2,
                mode='markers',
                marker=dict(size=5, color='blue', opacity=0.5)
            ))

        # 로딩 벡터 추가
        loadings = self.get_loadings()
        scale = 3.0  # 화살표 스케일

        for i, feature in enumerate(self.feature_names):
            fig.add_annotation(
                ax=0, ay=0,
                axref='x', ayref='y',
                x=loadings.iloc[i, 0] * scale,
                y=loadings.iloc[i, 1] * scale,
                xref='x', yref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red',
                opacity=0.7
            )
            fig.add_annotation(
                x=loadings.iloc[i, 0] * scale,
                y=loadings.iloc[i, 1] * scale,
                text=feature,
                showarrow=False,
                font=dict(size=10, color='red'),
                xshift=10,
                yshift=10
            )

        explained_var, _ = self.get_explained_variance()
        fig.update_layout(
            title=title,
            xaxis_title=f'PC1 ({explained_var[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({explained_var[1]*100:.1f}%)',
            height=700,
            width=900,
            hovermode='closest'
        )

        return fig

    def plot_loadings_heatmap(self, n_components: int = 5) -> go.Figure:
        """주성분 로딩 히트맵"""
        loadings = self.get_loadings()
        loadings_subset = loadings.iloc[:, :n_components]

        fig = go.Figure(data=go.Heatmap(
            z=loadings_subset.values,
            x=loadings_subset.columns,
            y=loadings_subset.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(loadings_subset.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="로딩 값")
        ))

        fig.update_layout(
            title=f'상위 {n_components}개 주성분의 변수 기여도',
            xaxis_title='주성분',
            yaxis_title='센서 변수',
            height=600,
            width=800
        )

        return fig

    def plot_3d_scatter(self, y: pd.Series = None, title: str = "3D PCA Plot") -> go.Figure:
        """3D 산점도"""
        if self.X_pca is None:
            raise ValueError("PCA를 먼저 수행해야 합니다.")

        pc1 = self.X_pca[:, 0]
        pc2 = self.X_pca[:, 1]
        pc3 = self.X_pca[:, 2]

        if y is not None:
            fig = go.Figure(data=[go.Scatter3d(
                x=pc1,
                y=pc2,
                z=pc3,
                mode='markers',
                marker=dict(
                    size=4,
                    color=y,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="타겟 점수")
                ),
                text=[f'Score: {val:.1f}' for val in y],
                hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<br>%{text}<extra></extra>'
            )])
        else:
            fig = go.Figure(data=[go.Scatter3d(
                x=pc1,
                y=pc2,
                z=pc3,
                mode='markers',
                marker=dict(size=4, color='blue', opacity=0.5)
            )])

        explained_var, _ = self.get_explained_variance()
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f'PC1 ({explained_var[0]*100:.1f}%)',
                yaxis_title=f'PC2 ({explained_var[1]*100:.1f}%)',
                zaxis_title=f'PC3 ({explained_var[2]*100:.1f}%)',
            ),
            height=700,
            width=900
        )

        return fig

    def get_top_features(self, pc_index: int = 0, n_top: int = 5) -> pd.DataFrame:
        """특정 주성분에 가장 기여하는 변수들"""
        loadings = self.get_loadings()
        pc_name = f'PC{pc_index + 1}'

        top_features = loadings[pc_name].abs().nlargest(n_top)
        result = pd.DataFrame({
            '변수': top_features.index,
            '로딩 값': loadings.loc[top_features.index, pc_name].values,
            '절대값': top_features.values
        })

        return result

    def get_summary_stats(self) -> Dict:
        """PCA 결과 요약 통계"""
        explained_var, cumulative_var = self.get_explained_variance()

        # 90% 이상 설명하는데 필요한 주성분 수
        n_components_90 = np.argmax(cumulative_var >= 0.9) + 1

        return {
            'n_components': self.pca.n_components_,
            'n_features': len(self.feature_names),
            'total_variance_explained': cumulative_var[-1],
            'n_components_90': n_components_90,
            'variance_90': cumulative_var[n_components_90 - 1] if n_components_90 > 0 else 0
        }
