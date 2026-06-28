"""
패턴 분석 및 계층화 모듈
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PatternAnalyzer:
    """센서 데이터 패턴 분석 및 계층화"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.feature_importance = None
        self.selected_features = None

    def select_top_features_from_pca(
        self,
        loadings_df: pd.DataFrame,
        n_components: int = 3,
        n_features_per_pc: int = 5
    ) -> List[str]:
        """
        PCA 로딩에서 상위 기여 변수 선택

        Args:
            loadings_df: PCA 로딩 DataFrame (변수 × 주성분)
            n_components: 분석할 주성분 개수
            n_features_per_pc: 각 주성분당 선택할 변수 개수

        Returns:
            선택된 변수 이름 리스트
        """
        selected_features = set()

        for i in range(min(n_components, len(loadings_df.columns))):
            pc_name = f'PC{i+1}'
            if pc_name in loadings_df.columns:
                # 절대값 기준 상위 변수 선택
                top_vars = loadings_df[pc_name].abs().nlargest(n_features_per_pc).index
                selected_features.update(top_vars)

        self.selected_features = list(selected_features)
        return self.selected_features

    def calculate_feature_importance(
        self,
        loadings_df: pd.DataFrame,
        explained_variance: np.ndarray
    ) -> pd.DataFrame:
        """
        변수별 중요도 계산 (설명 분산으로 가중 평균)

        Args:
            loadings_df: PCA 로딩 DataFrame
            explained_variance: 각 주성분의 설명 분산

        Returns:
            변수별 중요도 DataFrame
        """
        # 각 변수의 가중 중요도 계산
        weighted_importance = np.zeros(len(loadings_df))

        for i, var_ratio in enumerate(explained_variance):
            if i < len(loadings_df.columns):
                pc_loadings = loadings_df.iloc[:, i].abs().values
                weighted_importance += pc_loadings * var_ratio

        importance_df = pd.DataFrame({
            'feature': loadings_df.index,
            'importance': weighted_importance
        }).sort_values('importance', ascending=False)

        self.feature_importance = importance_df
        return importance_df

    def hierarchical_clustering(
        self,
        X: pd.DataFrame,
        n_clusters: int = 3,
        random_state: int = 42
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        K-means 클러스터링으로 계층 분류

        Args:
            X: 입력 데이터
            n_clusters: 클러스터(계층) 개수
            random_state: 랜덤 시드

        Returns:
            labels: 클러스터 레이블
            cluster_stats: 클러스터별 통계
        """
        # 데이터 정규화
        X_scaled = self.scaler.fit_transform(X)

        # K-means 클러스터링
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        labels = self.kmeans.fit_predict(X_scaled)

        # 클러스터별 통계 계산
        cluster_stats = self._calculate_cluster_stats(X, labels)

        return labels, cluster_stats

    def _calculate_cluster_stats(
        self,
        X: pd.DataFrame,
        labels: np.ndarray
    ) -> pd.DataFrame:
        """클러스터별 통계 계산"""
        stats_list = []

        for cluster_id in range(labels.max() + 1):
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]

            stats = {
                'cluster': cluster_id,
                'n_samples': cluster_mask.sum(),
                'percentage': (cluster_mask.sum() / len(X)) * 100
            }

            # 각 변수의 평균과 표준편차
            for col in X.columns:
                stats[f'{col}_mean'] = cluster_data[col].mean()
                stats[f'{col}_std'] = cluster_data[col].std()

            stats_list.append(stats)

        return pd.DataFrame(stats_list)

    def create_hierarchical_labels(
        self,
        X: pd.DataFrame,
        method: str = 'kmeans',
        n_levels: int = 3
    ) -> pd.Series:
        """
        계층 레이블 생성

        Args:
            X: 입력 데이터
            method: 계층화 방법 ('kmeans', 'quantile')
            n_levels: 계층 개수

        Returns:
            계층 레이블
        """
        if method == 'kmeans':
            labels, _ = self.hierarchical_clustering(X, n_clusters=n_levels)
            # 평균값으로 레이블 정렬 (0=낮음, 1=중간, 2=높음)
            cluster_means = pd.Series(labels).groupby(labels).apply(
                lambda x: X.iloc[x.index].mean().mean()
            )
            label_mapping = {old: new for new, old in enumerate(cluster_means.argsort())}
            labels = pd.Series([label_mapping[l] for l in labels])

        elif method == 'quantile':
            # 변수 평균의 quantile 기반 분류
            X_mean = X.mean(axis=1)
            labels = pd.qcut(
                X_mean,
                q=n_levels,
                labels=range(n_levels),
                duplicates='drop'
            )

        else:
            raise ValueError(f"지원하지 않는 방법: {method}")

        return labels

    def analyze_pattern_by_target(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        labels: pd.Series
    ) -> pd.DataFrame:
        """
        계층별 타겟 변수 분석

        Args:
            X: 센서 데이터
            y: 타겟 변수 (불안, 우울, 스트레스)
            labels: 계층 레이블

        Returns:
            계층별 타겟 통계
        """
        # 인덱스 리셋하여 동일하게 맞춤
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        if isinstance(labels, pd.Series):
            labels = labels.reset_index(drop=True)

        analysis = []

        for level in sorted(labels.unique()):
            level_mask = labels == level
            level_y = y[level_mask]
            level_X = X[level_mask]

            analysis.append({
                'level': level,
                'level_name': self._get_level_name(level, labels.nunique()),
                'n_samples': level_mask.sum(),
                'percentage': (level_mask.sum() / len(X)) * 100,
                'target_mean': level_y.mean(),
                'target_std': level_y.std(),
                'target_min': level_y.min(),
                'target_max': level_y.max(),
                # 각 변수의 평균
                **{f'{col}_mean': level_X[col].mean() for col in X.columns}
            })

        return pd.DataFrame(analysis)

    def _get_level_name(self, level: int, n_levels: int) -> str:
        """계층 이름 생성"""
        if n_levels == 3:
            names = ['낮은 활동', '중간 활동', '높은 활동']
        elif n_levels == 4:
            names = ['매우 낮음', '낮음', '높음', '매우 높음']
        elif n_levels == 5:
            names = ['매우 낮음', '낮음', '보통', '높음', '매우 높음']
        else:
            names = [f'레벨 {i}' for i in range(n_levels)]

        return names[level] if level < len(names) else f'레벨 {level}'

    def generate_pattern_summary(
        self,
        pattern_analysis: pd.DataFrame,
        feature_importance: pd.DataFrame
    ) -> Dict:
        """패턴 분석 요약 생성"""
        summary = {
            'n_levels': len(pattern_analysis),
            'n_features': len(feature_importance),
            'top_features': feature_importance.head(10)['feature'].tolist(),
            'level_distribution': pattern_analysis[['level_name', 'n_samples', 'percentage']].to_dict('records'),
            'target_by_level': pattern_analysis[['level_name', 'target_mean', 'target_std']].to_dict('records')
        }

        return summary
