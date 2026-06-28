"""
다차원 정신건강 프로파일링
- 척도 간 상관관계 분석
- 클러스터링을 통한 정신건강 유형 분류
- 유형별 특성 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = Path("analysis_results/03_multidimensional_profiling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_correlation_heatmap(df, scale_cols, output_path):
    """척도 간 상관관계 히트맵"""
    corr_matrix = df[scale_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    # 히트맵 그리기
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title('정신건강 척도 간 상관관계', fontsize=14, fontweight='bold', pad=20)

    # 레이블 설정
    labels = [col.replace('_Score', '').replace('_', ' ') for col in scale_cols]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"상관관계 히트맵 저장: {output_path}")
    plt.close()


def plot_pairwise_scatter(df, scale_cols, output_path):
    """척도 간 산점도 행렬"""
    # 샘플링 (너무 많으면 시각화가 느림)
    if len(df) > 1000:
        df_sample = df[scale_cols].sample(1000, random_state=42)
    else:
        df_sample = df[scale_cols]

    # 산점도 행렬
    g = sns.pairplot(df_sample, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20})
    g.fig.suptitle('척도 간 산점도 행렬', y=1.01, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"산점도 행렬 저장: {output_path}")
    plt.close()


def perform_clustering(df, scale_cols, n_clusters=4):
    """
    K-means 클러스터링 수행

    Args:
        df: 데이터프레임
        scale_cols: 척도 컬럼 리스트
        n_clusters: 클러스터 개수

    Returns:
        클러스터 레이블이 추가된 데이터프레임
    """
    # 결측치 제거
    df_clean = df[scale_cols].dropna()

    # 표준화
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # 결과 저장
    df_result = df_clean.copy()
    df_result['Cluster'] = cluster_labels

    print(f"\n클러스터링 완료: {n_clusters}개 클러스터")
    print(f"유효 데이터: {len(df_result)}명")
    print(f"\n클러스터별 인원:")
    for i in range(n_clusters):
        count = (cluster_labels == i).sum()
        pct = count / len(cluster_labels) * 100
        print(f"  클러스터 {i}: {count}명 ({pct:.1f}%)")

    return df_result, kmeans, scaler


def analyze_cluster_profiles(df_clustered, scale_cols):
    """클러스터별 프로필 분석"""
    print(f"\n{'='*80}")
    print("클러스터별 정신건강 프로필")
    print(f"{'='*80}")

    n_clusters = df_clustered['Cluster'].nunique()

    profiles = []
    for cluster_id in range(n_clusters):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]

        print(f"\n클러스터 {cluster_id} (n={len(cluster_data)}):")

        profile = {'Cluster': cluster_id, 'N': len(cluster_data)}

        for col in scale_cols:
            mean = cluster_data[col].mean()
            std = cluster_data[col].std()
            print(f"  {col}: {mean:.2f} (±{std:.2f})")
            profile[f'{col}_Mean'] = mean
            profile[f'{col}_SD'] = std

        profiles.append(profile)

    return pd.DataFrame(profiles)


def plot_cluster_profiles(profile_df, scale_cols, output_path):
    """클러스터 프로필 시각화"""
    n_clusters = len(profile_df)
    n_scales = len(scale_cols)

    fig, axes = plt.subplots(1, n_scales, figsize=(18, 5))
    fig.suptitle('클러스터별 정신건강 프로필', fontsize=16, fontweight='bold')

    for idx, col in enumerate(scale_cols):
        ax = axes[idx]

        means = [profile_df.loc[i, f'{col}_Mean'] for i in range(n_clusters)]
        stds = [profile_df.loc[i, f'{col}_SD'] for i in range(n_clusters)]

        x = range(n_clusters)
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
               color=sns.color_palette("husl", n_clusters))

        ax.set_xlabel('클러스터')
        ax.set_ylabel('점수')
        ax.set_title(col.replace('_Score', ''), fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"클러스터 프로필 저장: {output_path}")
    plt.close()


def plot_pca_clusters(df_clustered, scale_cols, output_path):
    """PCA를 사용한 클러스터 시각화"""
    # 표준화
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clustered[scale_cols])

    # PCA (2차원)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 8))

    n_clusters = df_clustered['Cluster'].nunique()
    colors = sns.color_palette("husl", n_clusters)

    for cluster_id in range(n_clusters):
        mask = df_clustered['Cluster'] == cluster_id
        ax.scatter(pca_data[mask, 0], pca_data[mask, 1],
                  c=[colors[cluster_id]], label=f'클러스터 {cluster_id}',
                  alpha=0.6, s=50)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('PCA를 통한 클러스터 시각화', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"PCA 클러스터 시각화 저장: {output_path}")
    plt.close()

    return pca


def plot_dendrogram_clustering(df, scale_cols, output_path):
    """계층적 클러스터링 덴드로그램"""
    # 샘플링 (너무 많으면 느림)
    if len(df) > 500:
        df_sample = df[scale_cols].dropna().sample(500, random_state=42)
    else:
        df_sample = df[scale_cols].dropna()

    # 표준화
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_sample)

    # 계층적 클러스터링
    linkage_matrix = linkage(scaled_data, method='ward')

    # 덴드로그램
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, ax=ax, no_labels=True)
    ax.set_title('계층적 클러스터링 덴드로그램', fontsize=14, fontweight='bold')
    ax.set_xlabel('샘플')
    ax.set_ylabel('거리')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"덴드로그램 저장: {output_path}")
    plt.close()


def find_optimal_clusters(df, scale_cols, max_clusters=10):
    """Elbow method로 최적 클러스터 수 찾기"""
    df_clean = df[scale_cols].dropna()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    inertias = []
    silhouette_scores = []

    from sklearn.metrics import silhouette_score

    K = range(2, max_clusters + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(scaled_data, kmeans.labels_)
        silhouette_scores.append(score)

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot
    ax1.plot(K, inertias, 'bo-')
    ax1.set_xlabel('클러스터 수 (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Silhouette plot
    ax2.plot(K, silhouette_scores, 'ro-')
    ax2.set_xlabel('클러스터 수 (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "optimal_clusters.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"최적 클러스터 분석 저장: {output_path}")
    plt.close()

    # 최적값 출력
    optimal_k = K[np.argmax(silhouette_scores)]
    print(f"\n권장 클러스터 수: {optimal_k} (Silhouette score = {max(silhouette_scores):.3f})")

    return optimal_k


def main():
    """메인 실행 함수"""
    print("="*80)
    print("다차원 정신건강 프로파일링")
    print("="*80)

    # 데이터 로드 (1회차)
    df_wave1 = pd.read_csv("analysis_results/01_data_exploration/wave1_full.csv")

    # 척도 컬럼
    scale_cols = ['PHQ9_Score', 'GAD7_Score', 'ISI_Score', 'Loneliness_Score', 'WHOQOL_Score']

    print(f"\n분석 대상: {len(df_wave1)}명")
    print(f"척도: {len(scale_cols)}개")

    # 1. 상관관계 분석
    print(f"\n{'='*80}")
    print("척도 간 상관관계 분석")
    print(f"{'='*80}")

    corr_matrix = df_wave1[scale_cols].corr()
    print("\n상관계수 매트릭스:")
    print(corr_matrix.round(3).to_string())

    # 상관관계 히트맵
    plot_correlation_heatmap(df_wave1, scale_cols, OUTPUT_DIR / "correlation_heatmap.png")

    # 산점도 행렬
    print("\n산점도 행렬 생성 중...")
    plot_pairwise_scatter(df_wave1, scale_cols, OUTPUT_DIR / "pairwise_scatter.png")

    # 2. 최적 클러스터 수 찾기
    print(f"\n{'='*80}")
    print("최적 클러스터 수 탐색")
    print(f"{'='*80}")

    optimal_k = find_optimal_clusters(df_wave1, scale_cols, max_clusters=8)

    # 3. 클러스터링 수행
    print(f"\n{'='*80}")
    print(f"K-means 클러스터링 (k={optimal_k})")
    print(f"{'='*80}")

    df_clustered, kmeans, scaler = perform_clustering(df_wave1, scale_cols, n_clusters=optimal_k)

    # 4. 클러스터 프로필 분석
    profile_df = analyze_cluster_profiles(df_clustered, scale_cols)

    # 프로필 저장
    profile_df.to_csv(OUTPUT_DIR / "cluster_profiles.csv", index=False)
    print(f"\n클러스터 프로필 저장: {OUTPUT_DIR / 'cluster_profiles.csv'}")

    # 5. 시각화
    print(f"\n{'='*80}")
    print("클러스터 시각화")
    print(f"{'='*80}\n")

    plot_cluster_profiles(profile_df, scale_cols, OUTPUT_DIR / "cluster_profiles.png")
    plot_pca_clusters(df_clustered, scale_cols, OUTPUT_DIR / "pca_clusters.png")
    plot_dendrogram_clustering(df_wave1, scale_cols, OUTPUT_DIR / "dendrogram.png")

    # 6. 클러스터링 결과 저장
    df_clustered.to_csv(OUTPUT_DIR / "wave1_clustered.csv", index=False)
    print(f"\n클러스터링 결과 저장: {OUTPUT_DIR / 'wave1_clustered.csv'}")

    print(f"\n{'='*80}")
    print("다차원 프로파일링 완료!")
    print(f"{'='*80}")

    return df_clustered, profile_df, optimal_k


if __name__ == "__main__":
    df_clustered, profile_df, optimal_k = main()
