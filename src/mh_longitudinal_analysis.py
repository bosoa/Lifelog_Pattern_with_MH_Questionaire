"""
종단 분석: 1회차-2회차 정신건강 변화 추적
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = Path("analysis_results/02_longitudinal_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_change_scores(df1, df2, scale_cols):
    """
    변화 점수 계산

    Args:
        df1: 1회차 데이터
        df2: 2회차 데이터
        scale_cols: 척도 컬럼 리스트

    Returns:
        변화 점수 데이터프레임
    """
    # ID로 정렬 확인
    df1_sorted = df1.sort_values('ID').reset_index(drop=True)
    df2_sorted = df2.sort_values('ID').reset_index(drop=True)

    # ID 일치 확인
    if not (df1_sorted['ID'].equals(df2_sorted['ID'])):
        print("경고: ID가 일치하지 않습니다. 매칭을 다시 수행합니다.")
        # ID 기준으로 병합
        merged = df1_sorted[['ID'] + scale_cols].merge(
            df2_sorted[['ID'] + scale_cols],
            on='ID',
            suffixes=('_W1', '_W2')
        )
        return merged

    # 변화 점수 계산
    change_data = {'ID': df1_sorted['ID']}

    for col in scale_cols:
        if col in df1_sorted.columns and col in df2_sorted.columns:
            change_data[f'{col}_W1'] = df1_sorted[col]
            change_data[f'{col}_W2'] = df2_sorted[col]
            change_data[f'{col}_Change'] = df2_sorted[col] - df1_sorted[col]
            change_data[f'{col}_PctChange'] = (
                (df2_sorted[col] - df1_sorted[col]) / (df1_sorted[col] + 0.01) * 100
            )

    return pd.DataFrame(change_data)


def categorize_change(change_score, threshold=2):
    """
    변화를 범주화

    Args:
        change_score: 변화 점수
        threshold: 변화 판정 임계값

    Returns:
        '개선', '유지', '악화' 중 하나
    """
    if pd.isna(change_score):
        return np.nan
    elif change_score < -threshold:
        return '개선'
    elif change_score > threshold:
        return '악화'
    else:
        return '유지'


def analyze_change_patterns(change_df, scale_name, score_col, threshold=2):
    """
    변화 패턴 분석

    Args:
        change_df: 변화 점수 데이터프레임
        scale_name: 척도 이름
        score_col: 점수 컬럼명 (예: 'PHQ9_Score')
        threshold: 변화 판정 임계값
    """
    print(f"\n{'='*80}")
    print(f"{scale_name} 변화 분석")
    print(f"{'='*80}")

    w1_col = f'{score_col}_W1'
    w2_col = f'{score_col}_W2'
    change_col = f'{score_col}_Change'

    if change_col not in change_df.columns:
        print(f"{change_col} 컬럼이 없습니다.")
        return None

    # 유효한 데이터만 선택
    valid_data = change_df[[w1_col, w2_col, change_col]].dropna()
    n = len(valid_data)

    print(f"\n분석 대상: {n}명")
    print(f"\n기본 통계:")
    print(f"  1회차 평균: {valid_data[w1_col].mean():.2f} (SD={valid_data[w1_col].std():.2f})")
    print(f"  2회차 평균: {valid_data[w2_col].mean():.2f} (SD={valid_data[w2_col].std():.2f})")
    print(f"  평균 변화: {valid_data[change_col].mean():.2f} (SD={valid_data[change_col].std():.2f})")

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(valid_data[w1_col], valid_data[w2_col])
    print(f"\nPaired t-test:")
    print(f"  t = {t_stat:.3f}, p = {p_value:.4f}")
    if p_value < 0.001:
        print(f"  → 매우 유의한 변화 (p < 0.001)")
    elif p_value < 0.01:
        print(f"  → 유의한 변화 (p < 0.01)")
    elif p_value < 0.05:
        print(f"  → 유의한 변화 (p < 0.05)")
    else:
        print(f"  → 유의하지 않음 (p >= 0.05)")

    # Effect size (Cohen's d)
    mean_diff = valid_data[w2_col].mean() - valid_data[w1_col].mean()
    pooled_std = np.sqrt((valid_data[w1_col].std()**2 + valid_data[w2_col].std()**2) / 2)
    cohens_d = mean_diff / pooled_std
    print(f"  Cohen's d = {cohens_d:.3f}", end="")
    if abs(cohens_d) < 0.2:
        print(" (작은 효과)")
    elif abs(cohens_d) < 0.5:
        print(" (중간 효과)")
    else:
        print(" (큰 효과)")

    # 변화 방향 분류
    change_category = valid_data[change_col].apply(lambda x: categorize_change(x, threshold))

    print(f"\n변화 패턴 (임계값 = ±{threshold}):")
    for cat in ['개선', '유지', '악화']:
        count = (change_category == cat).sum()
        pct = count / n * 100
        print(f"  {cat}: {count}명 ({pct:.1f}%)")

    return {
        'n': n,
        'mean_w1': valid_data[w1_col].mean(),
        'mean_w2': valid_data[w2_col].mean(),
        'mean_change': valid_data[change_col].mean(),
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'change_category': change_category
    }


def plot_change_distribution(change_df, scale_name, score_col, output_path):
    """변화 분포 시각화"""
    w1_col = f'{score_col}_W1'
    w2_col = f'{score_col}_W2'
    change_col = f'{score_col}_Change'

    valid_data = change_df[[w1_col, w2_col, change_col]].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{scale_name} 종단 변화 분석', fontsize=16, fontweight='bold')

    # 1. 1회차 vs 2회차 산점도
    ax1 = axes[0, 0]
    ax1.scatter(valid_data[w1_col], valid_data[w2_col], alpha=0.5)
    ax1.plot([valid_data[w1_col].min(), valid_data[w1_col].max()],
             [valid_data[w1_col].min(), valid_data[w1_col].max()],
             'r--', label='변화 없음')
    ax1.set_xlabel('1회차 점수')
    ax1.set_ylabel('2회차 점수')
    ax1.set_title('1회차 vs 2회차 점수')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 변화 점수 히스토그램
    ax2 = axes[0, 1]
    ax2.hist(valid_data[change_col], bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='r', linestyle='--', label='변화 없음')
    ax2.axvline(valid_data[change_col].mean(), color='g', linestyle='-',
                linewidth=2, label=f'평균 변화 = {valid_data[change_col].mean():.2f}')
    ax2.set_xlabel('변화 점수 (2회차 - 1회차)')
    ax2.set_ylabel('빈도')
    ax2.set_title('변화 점수 분포')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Box plot (1회차 vs 2회차)
    ax3 = axes[1, 0]
    data_to_plot = [valid_data[w1_col], valid_data[w2_col]]
    bp = ax3.boxplot(data_to_plot, labels=['1회차', '2회차'],
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_ylabel('점수')
    ax3.set_title('1회차 vs 2회차 분포')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 변화 패턴 파이 차트
    ax4 = axes[1, 1]
    change_category = valid_data[change_col].apply(lambda x: categorize_change(x, 2))
    category_counts = change_category.value_counts()
    colors = {'개선': '#4CAF50', '유지': '#FFC107', '악화': '#F44336'}
    ax4.pie(category_counts.values, labels=category_counts.index,
            autopct='%1.1f%%', colors=[colors.get(cat, 'gray') for cat in category_counts.index],
            startangle=90)
    ax4.set_title('변화 패턴 분포')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n시각화 저장: {output_path}")
    plt.close()


def plot_individual_trajectories(change_df, scale_name, score_col, output_path, n_sample=100):
    """개별 변화 궤적 시각화 (샘플링)"""
    w1_col = f'{score_col}_W1'
    w2_col = f'{score_col}_W2'

    valid_data = change_df[[w1_col, w2_col]].dropna()

    # 샘플링
    if len(valid_data) > n_sample:
        sample_data = valid_data.sample(n_sample, random_state=42)
    else:
        sample_data = valid_data

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx in range(len(sample_data)):
        row = sample_data.iloc[idx]
        x = [1, 2]
        y = [row[w1_col], row[w2_col]]

        if row[w2_col] < row[w1_col]:  # 개선
            color = '#4CAF50'
            alpha = 0.3
        elif row[w2_col] > row[w1_col]:  # 악화
            color = '#F44336'
            alpha = 0.3
        else:  # 유지
            color = '#9E9E9E'
            alpha = 0.2

        ax.plot(x, y, color=color, alpha=alpha, linewidth=0.5)

    # 평균 궤적
    mean_w1 = valid_data[w1_col].mean()
    mean_w2 = valid_data[w2_col].mean()
    ax.plot([1, 2], [mean_w1, mean_w2], color='blue', linewidth=3,
            marker='o', markersize=10, label='평균', zorder=100)

    ax.set_xlim(0.8, 2.2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['1회차', '2회차'])
    ax.set_ylabel('점수')
    ax.set_title(f'{scale_name} 개별 변화 궤적 (n={len(sample_data)})')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"궤적 시각화 저장: {output_path}")
    plt.close()


def main():
    """메인 실행 함수"""
    print("="*80)
    print("종단 분석: 1회차-2회차 정신건강 변화")
    print("="*80)

    # 데이터 로드
    matched_df1 = pd.read_csv("analysis_results/01_data_exploration/wave1_matched.csv")
    matched_df2 = pd.read_csv("analysis_results/01_data_exploration/wave2_matched.csv")

    print(f"\n매칭된 데이터: {len(matched_df1)}명")

    # 척도 목록
    scales = {
        'PHQ9_Score': 'PHQ-9 (우울)',
        'GAD7_Score': 'GAD-7 (불안)',
        'ISI_Score': 'ISI (불면증)',
        'Loneliness_Score': '외로움',
        'WHOQOL_Score': 'WHOQOL (삶의 질)'
    }

    # PSS는 1회차에만 있으므로 제외하거나 별도 처리

    # 변화 점수 계산
    print(f"\n{'='*80}")
    print("변화 점수 계산")
    print(f"{'='*80}")

    change_df = calculate_change_scores(matched_df1, matched_df2, list(scales.keys()))

    # 변화 점수 저장
    change_df.to_csv(OUTPUT_DIR / "change_scores.csv", index=False)
    print(f"\n변화 점수 저장: {OUTPUT_DIR / 'change_scores.csv'}")

    # 각 척도별 분석
    results = {}
    for score_col, scale_name in scales.items():
        result = analyze_change_patterns(change_df, scale_name, score_col, threshold=2)
        if result:
            results[scale_name] = result

            # 시각화
            plot_path = OUTPUT_DIR / f"{score_col}_change_distribution.png"
            plot_change_distribution(change_df, scale_name, score_col, plot_path)

            # 개별 궤적
            traj_path = OUTPUT_DIR / f"{score_col}_trajectories.png"
            plot_individual_trajectories(change_df, scale_name, score_col, traj_path, n_sample=200)

    # 전체 요약 테이블
    print(f"\n{'='*80}")
    print("전체 척도 변화 요약")
    print(f"{'='*80}")

    summary_data = []
    for scale_name, result in results.items():
        summary_data.append({
            '척도': scale_name,
            '분석 대상(명)': result['n'],
            '1회차 평균': f"{result['mean_w1']:.2f}",
            '2회차 평균': f"{result['mean_w2']:.2f}",
            '평균 변화': f"{result['mean_change']:.2f}",
            'p-value': f"{result['p_value']:.4f}",
            "Cohen's d": f"{result['cohens_d']:.3f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # 요약 저장
    summary_df.to_csv(OUTPUT_DIR / "change_summary.csv", index=False)
    print(f"\n요약 저장: {OUTPUT_DIR / 'change_summary.csv'}")

    print(f"\n{'='*80}")
    print("종단 분석 완료!")
    print(f"{'='*80}")

    return change_df, results


if __name__ == "__main__":
    change_df, results = main()
