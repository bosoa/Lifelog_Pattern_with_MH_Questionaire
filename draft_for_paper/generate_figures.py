#!/usr/bin/env python3
"""
Generate figures for coordinate transformation paper
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Load data
data_path = Path(__file__).parent.parent / "model_results_grouping_method_comparison" / "comparison_results.csv"
df = pd.read_csv(data_path)

output_dir = Path(__file__).parent / "figures"
output_dir.mkdir(exist_ok=True)

# Figure 1: Performance by Grouping Method
print("Generating Figure 1: Performance by Grouping Method...")
fig, ax = plt.subplots(figsize=(8, 5))

method_performance = df.groupby('method')['c_index'].agg(['mean', 'std']).reset_index()
method_performance = method_performance.sort_values('mean', ascending=False)
method_performance['method'] = method_performance['method'].str.upper()

colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
bars = ax.bar(method_performance['method'], method_performance['mean'],
              yerr=method_performance['std'], capsize=5, color=colors, alpha=0.8)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Feature Grouping Method', fontweight='bold')
ax.set_ylabel('Mean C-index', fontweight='bold')
ax.set_title('Performance Comparison by Feature Grouping Method', fontweight='bold', pad=20)
ax.set_ylim([0.62, 0.68])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.65, color='red', linestyle='--', alpha=0.5, label='Baseline (Sequential)')

plt.tight_layout()
plt.savefig(output_dir / 'figure1_grouping_method_performance.png', bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir / 'figure1_grouping_method_performance.png'}")

# Figure 2: Performance by Coordinate System
print("Generating Figure 2: Performance by Coordinate System...")
fig, ax = plt.subplots(figsize=(8, 5))

coord_performance = df.groupby('coord_type')['c_index'].agg(['mean', 'std']).reset_index()
coord_performance = coord_performance.sort_values('mean', ascending=False)
coord_performance['coord_type'] = coord_performance['coord_type'].str.capitalize()

colors = ['#e67e22', '#9b59b6', '#16a085']
bars = ax.bar(coord_performance['coord_type'], coord_performance['mean'],
              yerr=coord_performance['std'], capsize=5, color=colors, alpha=0.8)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Coordinate System', fontweight='bold')
ax.set_ylabel('Mean C-index', fontweight='bold')
ax.set_title('Performance Comparison by Coordinate System', fontweight='bold', pad=20)
ax.set_ylim([0.62, 0.68])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'figure2_coordinate_system_performance.png', bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir / 'figure2_coordinate_system_performance.png'}")

# Figure 3: Target-Specific Best Combinations
print("Generating Figure 3: Target-Specific Best Combinations...")
fig, ax = plt.subplots(figsize=(10, 6))

# Get best combination for each target
best_combinations = []
for target in ['anxiety', 'depression', 'stress']:
    target_data = df[df['target'] == target]
    best_idx = target_data['c_index'].idxmax()
    best_row = target_data.loc[best_idx]
    best_combinations.append({
        'target': target.capitalize(),
        'method': best_row['method'].upper(),
        'coord_type': best_row['coord_type'].capitalize(),
        'c_index': best_row['c_index']
    })

best_df = pd.DataFrame(best_combinations)

# Create grouped bar chart
x = np.arange(len(best_df))
width = 0.6

bars = ax.bar(x, best_df['c_index'], width, alpha=0.8,
              color=['#3498db', '#e74c3c', '#2ecc71'])

# Add value labels
for i, (bar, row) in enumerate(zip(bars, best_combinations)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2., height - 0.025,
            f"{row['method']}\n{row['coord_type']}",
            ha='center', va='top', fontsize=8, color='white', fontweight='bold')

ax.set_xlabel('Mental Health Target', fontweight='bold')
ax.set_ylabel('C-index', fontweight='bold')
ax.set_title('Best Performing Combinations by Target', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(best_df['target'])
ax.set_ylim([0.5, 0.75])
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Clinical Utility Threshold')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'figure3_target_best_combinations.png', bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir / 'figure3_target_best_combinations.png'}")

# Figure 4: Heatmap of all combinations
print("Generating Figure 4: Performance Heatmap...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, target in enumerate(['anxiety', 'depression', 'stress']):
    target_data = df[df['target'] == target]

    # Create pivot table
    pivot = target_data.pivot_table(values='c_index',
                                     index='method',
                                     columns='coord_type')
    pivot.index = [m.upper() for m in pivot.index]
    pivot.columns = [c.capitalize() for c in pivot.columns]

    # Reorder for consistency
    pivot = pivot.reindex(['SEQUENTIAL', 'CORRELATION', 'PCA', 'HYBRID'])
    pivot = pivot[['Polar', 'Spherical', 'Cylindrical']]

    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn',
                ax=axes[idx], vmin=0.60, vmax=0.70,
                cbar_kws={'label': 'C-index'},
                linewidths=0.5, linecolor='gray')

    axes[idx].set_title(f'{target.capitalize()}', fontweight='bold', fontsize=12)
    axes[idx].set_xlabel('Coordinate System', fontweight='bold')
    axes[idx].set_ylabel('Grouping Method', fontweight='bold')

plt.suptitle('C-index Performance Heatmap Across All Combinations',
             fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'figure4_performance_heatmap.png', bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir / 'figure4_performance_heatmap.png'}")

# Figure 5: Improvement over original baseline
print("Generating Figure 5: Improvement over Original Baseline...")
fig, ax = plt.subplots(figsize=(10, 6))

# Original baselines (from paper)
baselines = {
    'anxiety': 0.5334,
    'depression': 0.5397,
    'stress': 0.5351
}

# Calculate improvements for best combinations
improvements = []
for target in ['anxiety', 'depression', 'stress']:
    target_data = df[df['target'] == target]
    best_c_index = target_data['c_index'].max()
    baseline = baselines[target]
    improvement = ((best_c_index - baseline) / baseline) * 100
    improvements.append({
        'target': target.capitalize(),
        'improvement': improvement,
        'baseline': baseline,
        'best': best_c_index
    })

imp_df = pd.DataFrame(improvements)

x = np.arange(len(imp_df))
width = 0.35

bars1 = ax.bar(x - width/2, imp_df['baseline'], width, label='Original Baseline',
               alpha=0.8, color='#95a5a6')
bars2 = ax.bar(x + width/2, imp_df['best'], width, label='Best with Intelligent Grouping',
               alpha=0.8, color='#2ecc71')

# Add improvement percentage labels
for i, (bar, imp) in enumerate(zip(bars2, imp_df['improvement'])):
    height = bar.get_height()
    ax.annotate(f'+{imp:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='green')

ax.set_xlabel('Mental Health Target', fontweight='bold')
ax.set_ylabel('C-index', fontweight='bold')
ax.set_title('Improvement Over Original Baseline', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(imp_df['target'])
ax.set_ylim([0.5, 0.75])
ax.legend()
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Clinical Threshold')

plt.tight_layout()
plt.savefig(output_dir / 'figure5_improvement_baseline.png', bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir / 'figure5_improvement_baseline.png'}")

# Figure 6: Method comparison across targets
print("Generating Figure 6: Method Comparison Across Targets...")
fig, ax = plt.subplots(figsize=(12, 6))

methods = ['sequential', 'correlation', 'pca', 'hybrid']
targets = ['anxiety', 'depression', 'stress']
x = np.arange(len(methods))
width = 0.25

for i, target in enumerate(targets):
    target_data = df[df['target'] == target]
    method_means = target_data.groupby('method')['c_index'].mean()
    method_means = method_means.reindex(methods)

    offset = width * (i - 1)
    bars = ax.bar(x + offset, method_means, width, label=target.capitalize(), alpha=0.8)

ax.set_xlabel('Feature Grouping Method', fontweight='bold')
ax.set_ylabel('Mean C-index', fontweight='bold')
ax.set_title('Feature Grouping Method Performance Across Mental Health Targets',
             fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in methods])
ax.legend(title='Target', title_fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0.62, 0.70])

plt.tight_layout()
plt.savefig(output_dir / 'figure6_method_comparison_by_target.png', bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir / 'figure6_method_comparison_by_target.png'}")

print("\n✅ All figures generated successfully!")
print(f"📁 Figures saved to: {output_dir}")
