"""
라이프로그-정신건강 통합 분석
생체신호 데이터와 정신건강 설문 데이터를 통합하여 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = Path("analysis_results/04_lifelog_integration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_lifelog_data(file_pattern):
    """
    라이프로그 데이터 로드 및 집계

    Args:
        file_pattern: 파일 패턴 (예: "KLOSDOM_Preprocessed_Dataset/whole_a*.csv")

    Returns:
        집계된 라이프로그 데이터프레임
    """
    files = sorted(glob.glob(file_pattern))

    print(f"라이프로그 파일 로드: {len(files)}개")

    all_features = {}

    for file in files:
        feature_name = file.split('/')[-1].split('_')[1]  # a01, a02, etc.
        df = pd.read_csv(file)

        # 날짜 컬럼 (ID 제외)
        date_cols = [col for col in df.columns if col != 'ID']

        # 각 사용자별 통계 계산
        df['mean'] = df[date_cols].mean(axis=1, skipna=True)
        df['std'] = df[date_cols].std(axis=1, skipna=True)
        df['median'] = df[date_cols].median(axis=1, skipna=True)
        df['min'] = df[date_cols].min(axis=1, skipna=True)
        df['max'] = df[date_cols].max(axis=1, skipna=True)

        # ID와 통계만 저장
        all_features[feature_name] = df[['ID', 'mean', 'std', 'median', 'min', 'max']].rename(
            columns={
                'mean': f'{feature_name}_mean',
                'std': f'{feature_name}_std',
                'median': f'{feature_name}_median',
                'min': f'{feature_name}_min',
                'max': f'{feature_name}_max'
            }
        )

        print(f"  {feature_name}: {len(df)}명")

    # 모든 특징 통합
    df_lifelog = all_features[list(all_features.keys())[0]]
    for feature_name in list(all_features.keys())[1:]:
        df_lifelog = df_lifelog.merge(all_features[feature_name], on='ID', how='outer')

    print(f"\n통합 라이프로그 데이터: {len(df_lifelog)}명, {len(df_lifelog.columns)}개 특징")

    return df_lifelog


def integrate_with_questionnaire(df_lifelog, df_quest):
    """
    라이프로그와 설문 데이터 통합

    Args:
        df_lifelog: 라이프로그 데이터
        df_quest: 설문 데이터

    Returns:
        통합 데이터프레임
    """
    # ID 기준 병합
    df_integrated = df_lifelog.merge(df_quest, on='ID', how='inner')

    print(f"통합 데이터: {len(df_integrated)}명")
    print(f"  라이프로그: {len(df_lifelog)}명")
    print(f"  설문: {len(df_quest)}명")
    print(f"  매칭률: {len(df_integrated)/len(df_quest)*100:.1f}%")

    return df_integrated


def analyze_correlations(df, lifelog_features, mh_scores):
    """
    라이프로그 특징과 정신건강 지표 간 상관관계 분석

    Args:
        df: 통합 데이터프레임
        lifelog_features: 라이프로그 특징 리스트
        mh_scores: 정신건강 점수 리스트

    Returns:
        상관관계 매트릭스
    """
    print(f"\n{'='*80}")
    print("라이프로그-정신건강 상관관계 분석")
    print(f"{'='*80}")

    # 유효한 컬럼만 선택
    available_lifelog = [col for col in lifelog_features if col in df.columns]
    available_mh = [col for col in mh_scores if col in df.columns]

    if not available_lifelog or not available_mh:
        print("분석 가능한 데이터가 없습니다.")
        return None

    # 상관계수 계산
    corr_results = []

    for lifelog_col in available_lifelog:
        for mh_col in available_mh:
            # 유효한 데이터만 선택
            valid_data = df[[lifelog_col, mh_col]].dropna()

            if len(valid_data) > 30:  # 최소 샘플 크기
                r, p = stats.pearsonr(valid_data[lifelog_col], valid_data[mh_col])

                corr_results.append({
                    'Lifelog': lifelog_col,
                    'MH_Score': mh_col,
                    'r': r,
                    'p': p,
                    'n': len(valid_data)
                })

    corr_df = pd.DataFrame(corr_results)

    # 유의한 상관관계만 출력
    sig_corr = corr_df[corr_df['p'] < 0.05].sort_values('r', key=abs, ascending=False)

    print(f"\n유의한 상관관계 (p < 0.05): {len(sig_corr)}개")
    print("\n상위 20개:")
    print(sig_corr.head(20)[['Lifelog', 'MH_Score', 'r', 'p', 'n']].to_string(index=False))

    return corr_df


def plot_correlation_heatmap(corr_df, output_path, top_n=30):
    """상관관계 히트맵"""
    if corr_df is None or len(corr_df) == 0:
        print("상관관계 데이터가 없습니다.")
        return

    # 상위 N개만 선택 (절대값 기준)
    top_corr = corr_df.nlargest(top_n, 'r', keep='all')

    # 피봇 테이블 생성
    pivot = top_corr.pivot_table(values='r', index='Lifelog', columns='MH_Score')

    if pivot.empty:
        print("시각화할 데이터가 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(10, 12))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, vmin=-0.5, vmax=0.5,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title(f'라이프로그-정신건강 상관관계 (상위 {top_n}개)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('정신건강 지표')
    ax.set_ylabel('라이프로그 특징')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n상관관계 히트맵 저장: {output_path}")
    plt.close()


def plot_scatter_examples(df, corr_df, output_path, n_examples=6):
    """대표적인 상관관계 산점도"""
    if corr_df is None or len(corr_df) == 0:
        return

    # 강한 상관관계 선택
    sig_corr = corr_df[corr_df['p'] < 0.01].sort_values('r', key=abs, ascending=False)

    if len(sig_corr) < n_examples:
        n_examples = len(sig_corr)

    if n_examples == 0:
        print("산점도를 그릴 유의한 상관관계가 없습니다.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('라이프로그-정신건강 대표 상관관계', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for idx in range(n_examples):
        row = sig_corr.iloc[idx]
        lifelog_col = row['Lifelog']
        mh_col = row['MH_Score']
        r = row['r']
        p = row['p']

        ax = axes[idx]

        # 유효 데이터
        valid_data = df[[lifelog_col, mh_col]].dropna()

        ax.scatter(valid_data[lifelog_col], valid_data[mh_col], alpha=0.5, s=20)

        # 회귀선
        z = np.polyfit(valid_data[lifelog_col], valid_data[mh_col], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(valid_data[lifelog_col].min(), valid_data[lifelog_col].max(), 100)
        ax.plot(x_line, p_line(x_line), "r-", alpha=0.8, linewidth=2)

        ax.set_xlabel(lifelog_col, fontsize=9)
        ax.set_ylabel(mh_col, fontsize=9)
        ax.set_title(f'r={r:.3f}, p={p:.4f}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 빈 subplot 제거
    for idx in range(n_examples, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"산점도 예시 저장: {output_path}")
    plt.close()


def main():
    """메인 실행 함수"""
    print("="*80)
    print("라이프로그-정신건강 통합 분석")
    print("="*80)

    # 1. 라이프로그 데이터 로드 (생체신호만)
    print(f"\n{'='*80}")
    print("라이프로그 데이터 로드 및 집계")
    print(f"{'='*80}\n")

    df_lifelog = load_lifelog_data("KLOSDOM_Preprocessed_Dataset/whole_a*.csv")

    # 2. 설문 데이터 로드
    print(f"\n{'='*80}")
    print("정신건강 설문 데이터 로드")
    print(f"{'='*80}\n")

    df_quest = pd.read_csv("analysis_results/01_data_exploration/wave1_full.csv")
    print(f"설문 데이터: {len(df_quest)}명")

    # 3. 데이터 통합
    print(f"\n{'='*80}")
    print("데이터 통합")
    print(f"{'='*80}\n")

    df_integrated = integrate_with_questionnaire(df_lifelog, df_quest)

    # 통합 데이터 저장
    df_integrated.to_csv(OUTPUT_DIR / "integrated_data.csv", index=False)
    print(f"\n통합 데이터 저장: {OUTPUT_DIR / 'integrated_data.csv'}")

    # 4. 상관관계 분석
    # 주요 라이프로그 특징 (평균값)
    lifelog_features = [col for col in df_integrated.columns if '_mean' in col]

    # 정신건강 척도
    mh_scores = ['PHQ9_Score', 'GAD7_Score', 'ISI_Score', 'Loneliness_Score', 'WHOQOL_Score']

    corr_df = analyze_correlations(df_integrated, lifelog_features, mh_scores)

    if corr_df is not None:
        # 상관관계 저장
        corr_df.to_csv(OUTPUT_DIR / "correlations.csv", index=False)
        print(f"\n상관관계 저장: {OUTPUT_DIR / 'correlations.csv'}")

        # 5. 시각화
        print(f"\n{'='*80}")
        print("시각화")
        print(f"{'='*80}")

        plot_correlation_heatmap(corr_df, OUTPUT_DIR / "correlation_heatmap.png", top_n=40)
        plot_scatter_examples(df_integrated, corr_df, OUTPUT_DIR / "scatter_examples.png", n_examples=6)

    print(f"\n{'='*80}")
    print("라이프로그-정신건강 통합 분석 완료!")
    print(f"{'='*80}")

    return df_integrated, corr_df


if __name__ == "__main__":
    df_integrated, corr_df = main()
