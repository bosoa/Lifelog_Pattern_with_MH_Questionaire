"""
표준화 효과 논문용 그림 생성
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 영문 폰트 설정 (논문용)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

def create_figure1_performance_comparison():
    """Figure 1: C-index Performance Comparison Across Methods"""

    data = {
        'Method': ['Original', 'Polar', 'Std Polar', 'Spherical', 'Std Spherical', 'Cylindrical', 'Std Cylindrical'],
        'Anxiety': [0.5334, 0.6996, 0.6815, 0.6897, 0.6850, 0.6825, 0.6800],
        'Depression': [0.5397, 0.6834, 0.7199, 0.6891, 0.7150, 0.6863, 0.7100],
        'Stress': [0.5351, 0.6687, 0.7149, 0.6730, 0.7216, 0.6839, 0.7180],
    }

    df = pd.DataFrame(data)
    df['Average'] = df[['Anxiety', 'Depression', 'Stress']].mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Figure 1: C-index Performance Comparison Across Methods', fontsize=16, fontweight='bold', y=0.995)

    targets = ['Anxiety', 'Depression', 'Stress', 'Average']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for ax, target, color in zip(axes.flat, targets, colors):
        bars = ax.barh(df['Method'], df[target], color=color, alpha=0.7, edgecolor='black', linewidth=1.2)

        # 값 표시
        for bar, val in zip(bars, df[target]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('C-index', fontsize=11, fontweight='bold')
        ax.set_title(f'{target}', fontsize=13, fontweight='bold')
        ax.set_xlim([0.5, 0.75])
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # 기준선 표시
        baseline = df[df['Method'] == 'Original'][target].values[0]
        ax.axvline(baseline, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline:.4f}', alpha=0.7)
        ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    return fig

def create_figure2_standardization_effect():
    """Figure 2: Standardization Effect by Coordinate System"""

    # 표준화 효과 (델타 C-index)
    data = {
        'Coordinate': ['Polar', 'Spherical', 'Cylindrical'],
        'Anxiety': [-0.0181, -0.0047, -0.0025],
        'Depression': [+0.0365, +0.0259, +0.0237],
        'Stress': [+0.0462, +0.0486, +0.0341],
    }

    df = pd.DataFrame(data)
    df['Average'] = df[['Anxiety', 'Depression', 'Stress']].mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df['Coordinate']))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, df['Anxiety'], width, label='Anxiety', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, df['Depression'], width, label='Depression', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, df['Stress'], width, label='Stress', color='#2ecc71', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, df['Average'], width, label='Average', color='#9b59b6', alpha=0.8)

    # 값 표시
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:+.4f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8, fontweight='bold')

    ax.set_xlabel('Coordinate System', fontsize=12, fontweight='bold')
    ax.set_ylabel('Δ C-index (Standardized - Original)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: Standardization Effect by Coordinate System', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Coordinate'])
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    return fig

def create_figure3_improvement_heatmap():
    """Figure 3: Performance Improvement Heatmap"""

    # 원본 대비 개선율 (%)
    data = np.array([
        [31.16, 26.63, 24.97],  # Polar
        [27.77, 33.39, 33.60],  # Std Polar
        [29.30, 27.68, 25.77],  # Spherical
        [28.42, 32.48, 34.85],  # Std Spherical
        [27.95, 27.16, 27.81],  # Cylindrical
        [27.48, 31.55, 34.18],  # Std Cylindrical
    ])

    fig, ax = plt.subplots(figsize=(10, 7))

    methods = ['Polar', 'Std Polar', 'Spherical', 'Std Spherical', 'Cylindrical', 'Std Cylindrical']
    targets = ['Anxiety', 'Depression', 'Stress']

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=20, vmax=35)

    # 축 설정
    ax.set_xticks(np.arange(len(targets)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(targets, fontsize=11)
    ax.set_yticklabels(methods, fontsize=11)

    # 값 표시
    for i in range(len(methods)):
        for j in range(len(targets)):
            text = ax.text(j, i, f'{data[i, j]:.1f}%',
                          ha="center", va="center", color="black",
                          fontsize=10, fontweight='bold')

    ax.set_title('Figure 3: Performance Improvement vs. Baseline (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Variable', fontsize=12, fontweight='bold')
    ax.set_ylabel('Method', fontsize=12, fontweight='bold')

    # 컬러바
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (%)', fontsize=11, fontweight='bold')

    plt.tight_layout()

    return fig

def create_figure4_geometric_explanation():
    """Figure 4: Geometric Explanation of Standardization Effect"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Figure 4: Geometric Effect of Standardization on Polar Transformation',
                 fontsize=14, fontweight='bold')

    # 시뮬레이션 데이터 생성
    np.random.seed(42)
    n_samples = 100

    # 예시: 체온(35-37) vs 걸음수(0-15000)
    temperature = np.random.normal(36, 0.5, n_samples)
    steps = np.random.normal(8000, 2000, n_samples)

    # 1. 표준화 없이 극좌표 변환
    ax1 = axes[0]
    r_unstd = np.sqrt(temperature**2 + steps**2)
    theta_unstd = np.arctan2(steps, temperature)

    ax1.scatter(theta_unstd, r_unstd, alpha=0.6, s=50, c='#e74c3c', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('θ (angle, radians)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('r (magnitude)', fontsize=11, fontweight='bold')
    ax1.set_title('A) Without Standardization\n(Temperature: 36°C, Steps: 8000)', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)

    # 통계 표시
    ax1.text(0.05, 0.95, f'r mean: {r_unstd.mean():.1f}\nθ range: {theta_unstd.ptp():.3f}',
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 표준화 후 극좌표 변환
    ax2 = axes[1]

    # 표준화
    temp_std = (temperature - temperature.mean()) / temperature.std()
    steps_std = (steps - steps.mean()) / steps.std()

    r_std = np.sqrt(temp_std**2 + steps_std**2)
    theta_std = np.arctan2(steps_std, temp_std)

    ax2.scatter(theta_std, r_std, alpha=0.6, s=50, c='#3498db', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('θ (angle, radians)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('r (magnitude)', fontsize=11, fontweight='bold')
    ax2.set_title('B) With Standardization\n(Both features: μ=0, σ=1)', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)

    # 통계 표시
    ax2.text(0.05, 0.95, f'r mean: {r_std.mean():.1f}\nθ range: {theta_std.ptp():.3f}',
            transform=ax2.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    return fig

def main():
    """모든 그림 생성"""

    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("📊 논문용 그림 생성 중...")
    print("="*70 + "\n")

    # Figure 1
    print("생성 중: Figure 1 - Performance Comparison")
    fig1 = create_figure1_performance_comparison()
    fig1.savefig(output_dir / 'figure1_performance_comparison.png', dpi=300, bbox_inches='tight')
    fig1.savefig(output_dir / 'figure1_performance_comparison.pdf', bbox_inches='tight')
    plt.close(fig1)
    print("✅ Figure 1 저장 완료")

    # Figure 2
    print("생성 중: Figure 2 - Standardization Effect")
    fig2 = create_figure2_standardization_effect()
    fig2.savefig(output_dir / 'figure2_standardization_effect.png', dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / 'figure2_standardization_effect.pdf', bbox_inches='tight')
    plt.close(fig2)
    print("✅ Figure 2 저장 완료")

    # Figure 3
    print("생성 중: Figure 3 - Improvement Heatmap")
    fig3 = create_figure3_improvement_heatmap()
    fig3.savefig(output_dir / 'figure3_improvement_heatmap.png', dpi=300, bbox_inches='tight')
    fig3.savefig(output_dir / 'figure3_improvement_heatmap.pdf', bbox_inches='tight')
    plt.close(fig3)
    print("✅ Figure 3 저장 완료")

    # Figure 4
    print("생성 중: Figure 4 - Geometric Explanation")
    fig4 = create_figure4_geometric_explanation()
    fig4.savefig(output_dir / 'figure4_geometric_explanation.png', dpi=300, bbox_inches='tight')
    fig4.savefig(output_dir / 'figure4_geometric_explanation.pdf', bbox_inches='tight')
    plt.close(fig4)
    print("✅ Figure 4 저장 완료")

    print("\n" + "="*70)
    print("✅ 모든 그림 생성 완료")
    print("="*70)
    print(f"\n📁 저장 위치: {output_dir}/")
    print("   - figure1_performance_comparison.png/.pdf")
    print("   - figure2_standardization_effect.png/.pdf")
    print("   - figure3_improvement_heatmap.png/.pdf")
    print("   - figure4_geometric_explanation.png/.pdf")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
