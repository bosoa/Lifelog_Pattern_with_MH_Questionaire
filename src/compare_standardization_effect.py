"""
표준화 효과 비교 분석 스크립트
원본, 극좌표, 표준화 극좌표 간 성능 비교
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def create_comparison_table():
    """성능 비교 테이블 생성"""

    # 데이터 정리 (POLAR_ANALYSIS_SUMMARY.md 및 실행 결과 기반)
    data = {
        '종속변수': ['Anxiety', 'Depression', 'Stress', '평균'],
        '원본 C-index': [0.5334, 0.5397, 0.5351, 0.5361],
        '극좌표 C-index': [0.6996, 0.6834, 0.6687, 0.6839],
        '표준화 극좌표 C-index': [0.6815, 0.7199, 0.7149, 0.7054],
    }

    df = pd.DataFrame(data)

    # 개선율 계산
    df['극좌표 개선율 (%)'] = ((df['극좌표 C-index'] - df['원본 C-index']) / df['원본 C-index'] * 100).round(2)
    df['표준화 극좌표 개선율 (%)'] = ((df['표준화 극좌표 C-index'] - df['원본 C-index']) / df['원본 C-index'] * 100).round(2)

    # 표준화 vs 비표준화 극좌표 비교
    df['표준화 효과 (절대)'] = (df['표준화 극좌표 C-index'] - df['극좌표 C-index']).round(4)
    df['표준화 효과 (%)'] = ((df['표준화 극좌표 C-index'] - df['극좌표 C-index']) / df['극좌표 C-index'] * 100).round(2)

    return df

def plot_comparison(df, output_dir='model_results'):
    """시각화 생성"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 그림 크기 설정
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. C-index 비교 막대 그래프
    ax1 = axes[0]
    x = range(len(df) - 1)  # 평균 제외
    width = 0.25

    ax1.bar([i - width for i in x], df['원본 C-index'][:-1], width, label='원본', alpha=0.7, color='#ff7f0e')
    ax1.bar([i for i in x], df['극좌표 C-index'][:-1], width, label='극좌표', alpha=0.7, color='#2ca02c')
    ax1.bar([i + width for i in x], df['표준화 극좌표 C-index'][:-1], width, label='표준화 극좌표', alpha=0.7, color='#1f77b4')

    ax1.set_xlabel('종속변수', fontsize=12, fontweight='bold')
    ax1.set_ylabel('C-index', fontsize=12, fontweight='bold')
    ax1.set_title('좌표계 변환 및 표준화 효과 비교', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['종속변수'][:-1])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim([0.5, 0.75])

    # C-index 값 표시
    for i in x:
        ax1.text(i - width, df['원본 C-index'][i] + 0.01, f"{df['원본 C-index'][i]:.4f}",
                ha='center', va='bottom', fontsize=9)
        ax1.text(i, df['극좌표 C-index'][i] + 0.01, f"{df['극좌표 C-index'][i]:.4f}",
                ha='center', va='bottom', fontsize=9)
        ax1.text(i + width, df['표준화 극좌표 C-index'][i] + 0.01, f"{df['표준화 극좌표 C-index'][i]:.4f}",
                ha='center', va='bottom', fontsize=9)

    # 2. 개선율 비교
    ax2 = axes[1]
    x2 = range(len(df) - 1)
    width2 = 0.35

    improvement_polar = df['극좌표 개선율 (%)'][:-1].values
    improvement_std_polar = df['표준화 극좌표 개선율 (%)'][:-1].values

    ax2.bar([i - width2/2 for i in x2], improvement_polar, width2, label='극좌표', alpha=0.7, color='#2ca02c')
    ax2.bar([i + width2/2 for i in x2], improvement_std_polar, width2, label='표준화 극좌표', alpha=0.7, color='#1f77b4')

    ax2.set_xlabel('종속변수', fontsize=12, fontweight='bold')
    ax2.set_ylabel('원본 대비 개선율 (%)', fontsize=12, fontweight='bold')
    ax2.set_title('원본 대비 성능 개선율', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(df['종속변수'][:-1])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # 개선율 값 표시
    for i in x2:
        ax2.text(i - width2/2, improvement_polar[i] + 1, f"{improvement_polar[i]:.1f}%",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(i + width2/2, improvement_std_polar[i] + 1, f"{improvement_std_polar[i]:.1f}%",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # 저장
    output_file = output_path / 'standardization_effect_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 시각화 저장: {output_file}")
    plt.close()

def main():
    """메인 실행 함수"""
    print("\n" + "="*70)
    print("📊 표준화 효과 비교 분석")
    print("="*70 + "\n")

    # 비교 테이블 생성
    df = create_comparison_table()

    # 결과 출력
    print("📋 성능 비교 테이블:")
    print(df.to_string(index=False))
    print()

    # CSV 저장
    output_dir = Path('model_results')
    output_dir.mkdir(exist_ok=True)
    csv_file = output_dir / 'standardization_effect_comparison.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"✅ CSV 저장: {csv_file}\n")

    # 시각화
    plot_comparison(df)

    # 주요 발견사항 출력
    print("\n" + "="*70)
    print("🔍 주요 발견사항")
    print("="*70)

    print("\n1️⃣ 원본 대비 성능 향상:")
    for idx, row in df.iterrows():
        if idx < 3:  # 평균 제외
            print(f"   {row['종속변수']:12s}: 극좌표 +{row['극좌표 개선율 (%)']:5.1f}%, "
                  f"표준화 극좌표 +{row['표준화 극좌표 개선율 (%)']:5.1f}%")

    print("\n2️⃣ 표준화 효과:")
    for idx, row in df.iterrows():
        if idx < 3:
            effect_sign = "+" if row['표준화 효과 (절대)'] > 0 else ""
            effect_emoji = "⬆️" if row['표준화 효과 (절대)'] > 0 else "⬇️"
            print(f"   {row['종속변수']:12s}: {effect_sign}{row['표준화 효과 (절대)']:.4f} "
                  f"({effect_sign}{row['표준화 효과 (%)']:.2f}%) {effect_emoji}")

    print("\n3️⃣ 최고 성능:")
    best_method = {}
    for idx, row in df.iterrows():
        if idx < 3:
            methods = {
                '원본': row['원본 C-index'],
                '극좌표': row['극좌표 C-index'],
                '표준화 극좌표': row['표준화 극좌표 C-index']
            }
            best = max(methods.items(), key=lambda x: x[1])
            print(f"   {row['종속변수']:12s}: {best[0]} (C-index = {best[1]:.4f})")

    print("\n" + "="*70)
    print("✅ 분석 완료")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
