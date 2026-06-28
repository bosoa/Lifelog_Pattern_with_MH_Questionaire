"""
종속변수 데이터 분포 시각화 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


class DataDistributionVisualizer:
    """종속변수 데이터 분포 시각화 클래스"""

    def __init__(self):
        self.html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>정신건강 점수 데이터 분포 분석</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 50px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.8em;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        .header p {{
            opacity: 0.95;
            font-size: 1.2em;
        }}
        .content {{
            padding: 50px;
        }}
        .section {{
            margin-bottom: 50px;
            padding: 40px;
            background: #f8f9fa;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            color: #667eea;
            font-size: 2em;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 4px solid #667eea;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-card h4 {{
            color: #667eea;
            font-size: 0.95em;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}
        .stat-card p {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 1.3em;
            color: #555;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 30px;
            margin: 40px 0;
        }}
        .comparison-card {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .comparison-card h3 {{
            color: #667eea;
            margin-bottom: 20px;
            text-align: center;
            font-size: 1.5em;
        }}
        .footer {{
            background: #f8f9fa;
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.95em;
        }}
        .insight-box {{
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 5px solid #667eea;
            padding: 25px;
            margin: 30px 0;
            border-radius: 10px;
        }}
        .insight-box h4 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        .insight-box p {{
            color: #555;
            line-height: 1.8;
        }}
        @media (max-width: 1200px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 정신건강 점수 데이터 분포 분석</h1>
            <p>Anxiety, Depression, Stress 점수 분포 및 통계 분석</p>
            <p style="margin-top: 10px; opacity: 0.8;">생성 일시: {generation_time}</p>
        </div>

        <div class="content">
            {content}
        </div>

        <div class="footer">
            <p><strong>KLOSDOM Lifelog Pattern Data Generation System</strong></p>
            <p>Korea Lifelog Observatory for Sustainable Dementia Outcome Management</p>
            <p style="margin-top: 10px; opacity: 0.7;">Generated: {generation_time}</p>
        </div>
    </div>
</body>
</html>
"""

    def _fig_to_base64(self, fig) -> str:
        """matplotlib figure를 base64 인코딩"""
        buffer = BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return f'data:image/png;base64,{image_base64}'

    def _create_distribution_histogram(self, data: pd.DataFrame, target_var: str) -> str:
        """분포 히스토그램 생성"""
        # 무한대 값 제거
        data = data.copy()
        score_col = f'{target_var}_score'
        data = data[np.isfinite(data[score_col])]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 전체 분포
        axes[0].hist(data[score_col], bins=30, color='#667eea',
                     alpha=0.7, edgecolor='black')
        axes[0].set_xlabel(f'{target_var.upper()} 점수', fontsize=12)
        axes[0].set_ylabel('빈도', fontsize=12)
        axes[0].set_title(f'{target_var.upper()} 점수 분포',
                          fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # 통계 정보 추가
        mean_val = data[f'{target_var}_score'].mean()
        median_val = data[f'{target_var}_score'].median()
        axes[0].axvline(mean_val, color='red', linestyle='--',
                       linewidth=2, label=f'평균: {mean_val:.2f}')
        axes[0].axvline(median_val, color='green', linestyle='--',
                       linewidth=2, label=f'중앙값: {median_val:.2f}')
        axes[0].legend()

        # 계층별 분포
        colors = ['#ffc107', '#ff9800', '#4caf50']
        for level in sorted(data['level'].unique()):
            level_data = data[data['level'] == level][f'{target_var}_score']
            axes[1].hist(level_data, bins=20, alpha=0.5,
                        label=f'레벨 {level}', color=colors[level],
                        edgecolor='black')

        axes[1].set_xlabel(f'{target_var.upper()} 점수', fontsize=12)
        axes[1].set_ylabel('빈도', fontsize=12)
        axes[1].set_title(f'계층별 {target_var.upper()} 점수 분포',
                          fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        return self._fig_to_base64(fig)

    def _create_boxplot(self, data: pd.DataFrame, target_var: str) -> str:
        """박스플롯 생성"""
        # 무한대 값 제거
        data = data.copy()
        score_col = f'{target_var}_score'
        data = data[np.isfinite(data[score_col])]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 전체 박스플롯
        axes[0].boxplot([data[score_col]],
                       labels=[target_var.upper()],
                       patch_artist=True,
                       boxprops=dict(facecolor='#667eea', alpha=0.7))
        axes[0].set_ylabel('점수', fontsize=12)
        axes[0].set_title(f'{target_var.upper()} 점수 박스플롯',
                          fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # 계층별 박스플롯
        colors = ['#ffc107', '#ff9800', '#4caf50']
        box_data = [data[data['level'] == level][f'{target_var}_score'].values
                    for level in sorted(data['level'].unique())]
        bp = axes[1].boxplot(box_data,
                            labels=[f'레벨 {i}' for i in range(len(box_data))],
                            patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1].set_ylabel('점수', fontsize=12)
        axes[1].set_title(f'계층별 {target_var.upper()} 점수 박스플롯',
                          fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        return self._fig_to_base64(fig)

    def _create_violin_plot(self, data: pd.DataFrame, target_var: str) -> str:
        """바이올린 플롯 생성"""
        # 무한대 값 제거
        data = data.copy()
        score_col = f'{target_var}_score'
        data = data[np.isfinite(data[score_col])]

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#ffc107', '#ff9800', '#4caf50']
        positions = []
        for level in sorted(data['level'].unique()):
            level_data = data[data['level'] == level][score_col].values
            positions.append(level_data)

        parts = ax.violinplot(positions,
                             positions=range(len(positions)),
                             showmeans=True,
                             showmedians=True)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([f'레벨 {i}' for i in range(len(positions))])
        ax.set_ylabel('점수', fontsize=12)
        ax.set_title(f'계층별 {target_var.upper()} 점수 분포 (Violin Plot)',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        return self._fig_to_base64(fig)

    def _calculate_statistics(self, data: pd.DataFrame, target_var: str) -> dict:
        """통계량 계산"""
        # 무한대 값 제거
        score_col = f'{target_var}_score'
        score_data = data[score_col]
        score_data = score_data[np.isfinite(score_data)]

        stats = {
            '샘플 수': len(score_data),
            '평균': score_data.mean(),
            '표준편차': score_data.std(),
            '최소값': score_data.min(),
            '25% 백분위': score_data.quantile(0.25),
            '중앙값': score_data.median(),
            '75% 백분위': score_data.quantile(0.75),
            '최대값': score_data.max(),
            '왜도': score_data.skew(),
            '첨도': score_data.kurtosis()
        }

        return stats

    def _generate_stats_section(self, stats: dict, target_var: str) -> str:
        """통계 섹션 생성"""
        html = f'<div class="section"><h2>{target_var.upper()} 통계 요약</h2>'
        html += '<div class="stats-grid">'

        for key, value in stats.items():
            if isinstance(value, (int, np.integer)):
                formatted_value = f'{value:,}'
            else:
                formatted_value = f'{value:.4f}'

            html += f'''
            <div class="stat-card">
                <h4>{key}</h4>
                <p>{formatted_value}</p>
            </div>
            '''

        html += '</div></div>'
        return html

    def _generate_distribution_section(self, data: pd.DataFrame, target_var: str) -> str:
        """분포 시각화 섹션 생성"""
        html = f'<div class="section"><h2>{target_var.upper()} 점수 분포</h2>'

        # 히스토그램
        hist_img = self._create_distribution_histogram(data, target_var)
        html += f'''
        <div class="chart-container">
            <p class="chart-title">히스토그램 분석</p>
            <img src="{hist_img}" alt="Distribution Histogram" />
        </div>
        '''

        # 박스플롯
        box_img = self._create_boxplot(data, target_var)
        html += f'''
        <div class="chart-container">
            <p class="chart-title">박스플롯 분석</p>
            <img src="{box_img}" alt="Boxplot" />
        </div>
        '''

        # 바이올린 플롯
        violin_img = self._create_violin_plot(data, target_var)
        html += f'''
        <div class="chart-container">
            <p class="chart-title">바이올린 플롯 (밀도 분포)</p>
            <img src="{violin_img}" alt="Violin Plot" />
        </div>
        '''

        html += '</div>'
        return html

    def _generate_level_comparison_section(self, data: pd.DataFrame, target_var: str) -> str:
        """계층별 비교 섹션"""
        html = f'<div class="section"><h2>계층별 {target_var.upper()} 점수 비교</h2>'
        html += '<div class="stats-grid">'

        for level in sorted(data['level'].unique()):
            level_data = data[data['level'] == level][f'{target_var}_score']
            html += f'''
            <div class="stat-card">
                <h4>레벨 {level}</h4>
                <p>{level_data.mean():.2f}</p>
                <small style="color: #666;">평균 점수</small>
            </div>
            <div class="stat-card">
                <h4>레벨 {level} 표준편차</h4>
                <p>{level_data.std():.2f}</p>
                <small style="color: #666;">변동성</small>
            </div>
            '''

        html += '</div></div>'
        return html

    def generate_report(self, data_paths: list, output_path: str):
        """전체 리포트 생성"""
        print("\n📊 데이터 분포 시각화 리포트 생성 중...")

        content_sections = []

        # 개요 섹션
        content_sections.append('''
        <div class="section">
            <h2>📋 분석 개요</h2>
            <div class="insight-box">
                <h4>분석 목적</h4>
                <p>본 리포트는 KLOSDOM 데이터셋의 정신건강 점수(Anxiety, Depression, Stress)의
                분포 특성을 분석하고, 계층별 차이를 시각화하여 데이터의 특성을 이해하는 것을 목표로 합니다.</p>
            </div>
        </div>
        ''')

        # 각 종속변수별 분석
        for filepath in data_paths:
            filename = Path(filepath).stem
            target = filename.split('_')[0]

            print(f"   분석 중: {target.upper()}")

            # 데이터 로드
            data = pd.read_csv(filepath)

            # 통계 요약
            stats = self._calculate_statistics(data, target)
            content_sections.append(self._generate_stats_section(stats, target))

            # 분포 시각화
            content_sections.append(self._generate_distribution_section(data, target))

            # 계층별 비교
            content_sections.append(self._generate_level_comparison_section(data, target))

        # 3개 변수 비교 섹션 추가
        content_sections.append(self._create_comparison_section(data_paths))

        # HTML 생성
        html_content = self.html_template.format(
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content="\n".join(content_sections)
        )

        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"   ✓ 리포트 저장: {output_path}")
        return output_path

    def _create_comparison_section(self, data_paths: list) -> str:
        """3개 변수 비교 섹션"""
        html = '<div class="section"><h2>🔍 종속변수 간 비교</h2>'

        # 데이터 로드
        all_data = {}
        for filepath in data_paths:
            filename = Path(filepath).stem
            target = filename.split('_')[0]
            if target in ['anxiety', 'depression', 'stress']:
                all_data[target] = pd.read_csv(filepath)

        # 비교 차트 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 평균 점수 비교
        ax = axes[0, 0]
        targets = list(all_data.keys())
        means = [all_data[t][f'{t}_score'].mean() for t in targets]
        colors = ['#667eea', '#764ba2', '#ffc107']
        ax.bar(targets, means, color=colors[:len(targets)], alpha=0.7, edgecolor='black')
        ax.set_ylabel('평균 점수', fontsize=12)
        ax.set_title('종속변수별 평균 점수', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 2. 표준편차 비교
        ax = axes[0, 1]
        stds = [all_data[t][f'{t}_score'].std() for t in targets]
        ax.bar(targets, stds, color=colors[:len(targets)], alpha=0.7, edgecolor='black')
        ax.set_ylabel('표준편차', fontsize=12)
        ax.set_title('종속변수별 표준편차 (변동성)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 3. 분포 범위 비교 (박스플롯)
        ax = axes[1, 0]
        box_data = [all_data[t][f'{t}_score'].values for t in targets]
        bp = ax.boxplot(box_data, labels=targets, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(targets)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('점수', fontsize=12)
        ax.set_title('종속변수별 점수 분포 비교', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 4. 계층별 평균 비교
        ax = axes[1, 1]
        x = np.arange(3)  # 레벨 0, 1, 2
        width = 0.25
        for i, target in enumerate(targets):
            level_means = [all_data[target][all_data[target]['level'] == level][f'{target}_score'].mean()
                          for level in range(3)]
            ax.bar(x + i*width, level_means, width, label=target.upper(),
                   color=colors[i], alpha=0.7)

        ax.set_xlabel('계층 레벨', fontsize=12)
        ax.set_ylabel('평균 점수', fontsize=12)
        ax.set_title('계층별 종속변수 평균 점수', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['레벨 0', '레벨 1', '레벨 2'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        img_base64 = self._fig_to_base64(fig)
        html += f'''
        <div class="chart-container">
            <p class="chart-title">종속변수 간 종합 비교</p>
            <img src="{img_base64}" alt="Comparison Chart" />
        </div>
        '''

        html += '</div>'
        return html


def main():
    """실행 예시"""
    import sys
    sys.path.append('src')
    from data_loader import KLOSDOMDataLoader
    import tempfile
    import os

    loader = KLOSDOMDataLoader()

    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    data_paths = []

    print("\n📊 원본 데이터(1-10점) 로드 중...")

    # 각 타겟별로 원본 데이터 로드 및 임시 파일 생성
    for target in ['anxiety', 'depression', 'stress']:
        X, y, feature_names = loader.prepare_pca_data(
            target_variable=target,
            min_data_points=10
        )

        # 데이터 결합
        data = X.copy()
        data[f'{target}_score'] = y.values

        # 레벨 생성 (1-3점: 낮음, 4-6점: 중간, 7-10점: 높음)
        data['level'] = pd.cut(y, bins=[0, 3, 6, 10], labels=[0, 1, 2], include_lowest=True)
        data['level_name'] = data['level'].map({0: '낮음', 1: '중간', 2: '높음'})

        # 임시 파일 저장
        temp_file = os.path.join(temp_dir, f'{target}_hierarchical_data.csv')
        data.to_csv(temp_file, index=False)
        data_paths.append(temp_file)

        print(f"   ✓ {target.upper()}: {len(y):,}개 샘플 (범위: {y.min():.0f}-{y.max():.0f})")

    # 리포트 생성
    visualizer = DataDistributionVisualizer()
    output_path = "model_results/data_distribution_report.html"
    visualizer.generate_report(data_paths, output_path)

    # 임시 파일 삭제
    for filepath in data_paths:
        os.remove(filepath)
    os.rmdir(temp_dir)

    print(f"\n✅ 데이터 분포 리포트 생성 완료!")
    print(f"   📄 {output_path}")


if __name__ == "__main__":
    main()
