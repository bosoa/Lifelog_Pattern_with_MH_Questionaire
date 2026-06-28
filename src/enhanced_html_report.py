"""
개선된 HTML 리포트 생성 - 계층화 데이터 시각화 포함
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import base64
from io import BytesIO

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns

    # 한글 폰트 설정 (macOS)
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


class EnhancedHTMLReportGenerator:
    """계층화 데이터 시각화를 포함한 HTML 리포트 생성기"""

    def __init__(self):
        self.html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>정신건강 예측 모델 비교 리포트</title>
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
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        .header p {{
            opacity: 0.95;
            font-size: 1.1em;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 6px 12px rgba(102,126,234,0.3);
            transition: transform 0.3s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-card h3 {{
            font-size: 0.95em;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        .metric-card p {{
            font-size: 2.2em;
            font-weight: bold;
        }}
        .metric-card.best {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        tr:hover {{
            background-color: #f5f7ff;
        }}
        .best-model {{
            background-color: #d4edda !important;
            font-weight: bold;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }}
        .chart-container h3 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.3em;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }}
        .feature-importance {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .feature-bar {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .feature-bar .label {{
            font-size: 0.95em;
            margin-bottom: 8px;
            font-weight: 500;
            color: #333;
        }}
        .feature-bar .bar {{
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 25px;
            border-radius: 5px;
            transition: width 0.5s ease;
            position: relative;
        }}
        .feature-bar .bar::after {{
            content: attr(data-value);
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: white;
            font-weight: bold;
            font-size: 0.85em;
        }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }}
        .tab {{
            padding: 12px 24px;
            background: transparent;
            border: none;
            cursor: pointer;
            font-size: 1em;
            color: #666;
            transition: all 0.3s ease;
            border-bottom: 3px solid transparent;
        }}
        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
            font-weight: 600;
        }}
        .tab:hover {{
            color: #667eea;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .level-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            margin: 0 5px;
        }}
        .level-0 {{ background-color: #ffc107; color: #000; }}
        .level-1 {{ background-color: #ff9800; color: white; }}
        .level-2 {{ background-color: #4caf50; color: white; }}
        .footer {{
            background: #f8f9fa;
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.95em;
        }}
        .insight-box {{
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }}
        .insight-box h4 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 정신건강 예측 모델 분석 리포트</h1>
            <p>생성 일시: {generation_time}</p>
            <p>분석 대상: <strong>{target_variable}</strong></p>
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

    <script>
        // 탭 전환 기능
        function switchTab(tabName) {{
            // 모든 탭과 콘텐츠 비활성화
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            // 선택된 탭과 콘텐츠 활성화
            document.querySelector(`.tab[onclick="switchTab('${{tabName}}')"]`).classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }}
    </script>
</body>
</html>
"""

    def generate_full_report(
        self,
        results: Dict,
        summary_df: pd.DataFrame,
        hierarchical_data: pd.DataFrame,
        target_variable: str,
        output_path: str
    ):
        """계층화 데이터 시각화를 포함한 완전한 HTML 리포트 생성"""
        print(f"\n📄 개선된 HTML 리포트 생성 중...")

        content_sections = []

        # 1. 개요 및 주요 지표
        content_sections.append(self._generate_overview(summary_df, hierarchical_data))

        # 2. 계층화 데이터 시각화
        if PLOT_AVAILABLE:
            content_sections.append(self._generate_hierarchical_visualization(hierarchical_data, target_variable))

        # 3. 모델 성능 비교
        content_sections.append(self._generate_model_comparison(results, summary_df))

        # 4. 특징 중요도
        content_sections.append(self._generate_feature_importance(results))

        # 5. 상세 분석
        content_sections.append(self._generate_detailed_analysis(hierarchical_data, target_variable))

        # HTML 조합
        html_content = self.html_template.format(
            generation_time=datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S"),
            target_variable=target_variable.upper(),
            content="\n".join(content_sections)
        )

        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"   ✓ 리포트 저장: {output_path}")
        return output_path

    def _generate_overview(self, summary_df: pd.DataFrame, data: pd.DataFrame) -> str:
        """개요 섹션"""
        best_model = summary_df.iloc[0]

        level_counts = data['level'].value_counts().sort_index()
        level_names = data['level_name'].unique()

        html = '<div class="section">'
        html += '<h2>📊 분석 개요</h2>'
        html += '<div class="metrics-grid">'

        # 주요 메트릭
        html += f'''
        <div class="metric-card best">
            <h3>최고 성능 모델</h3>
            <p>{best_model["모델"]}</p>
        </div>
        <div class="metric-card">
            <h3>Test R² Score</h3>
            <p>{best_model["Test R²"]:.4f}</p>
        </div>
        <div class="metric-card">
            <h3>Test MAE</h3>
            <p>{best_model["Test MAE"]:.4f}</p>
        </div>
        <div class="metric-card">
            <h3>총 샘플 수</h3>
            <p>{len(data):,}</p>
        </div>
        '''

        # 계층별 분포
        for level in sorted(level_counts.index):
            count = level_counts[level]
            pct = (count / len(data)) * 100
            html += f'''
            <div class="metric-card">
                <h3>레벨 {level} 샘플</h3>
                <p>{count:,} <span style="font-size:0.5em;">({pct:.1f}%)</span></p>
            </div>
            '''

        html += '</div></div>'
        return html

    def _generate_hierarchical_visualization(self, data: pd.DataFrame, target_var: str) -> str:
        """계층화 데이터 시각화"""
        if not PLOT_AVAILABLE:
            return ""

        html = '<div class="section">'
        html += '<h2>📈 계층화 데이터 시각화</h2>'

        # 변수 컬럼 추출
        exclude_cols = ['level', 'level_name', f'{target_var}_score']
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        # 1. 계층별 분포 (파이 차트)
        html += self._create_level_distribution_chart(data)

        # 2. 계층별 주요 변수 비교 (레이더 차트)
        html += self._create_radar_chart(data, feature_cols[:8], target_var)

        # 3. 계층별 박스플롯
        html += self._create_boxplots(data, feature_cols[:6], target_var)

        # 4. 변수 간 상관관계 히트맵
        html += self._create_correlation_heatmap(data, feature_cols)

        html += '</div>'
        return html

    def _create_level_distribution_chart(self, data: pd.DataFrame) -> str:
        """계층별 분포 파이 차트"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 파이 차트
        level_counts = data['level'].value_counts().sort_index()
        colors = ['#ffc107', '#ff9800', '#4caf50']
        ax1.pie(level_counts, labels=[f'레벨 {i}' for i in level_counts.index],
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('계층별 샘플 분포', fontsize=14, fontweight='bold')

        # 막대 그래프
        ax2.bar([f'레벨 {i}' for i in level_counts.index], level_counts, color=colors)
        ax2.set_ylabel('샘플 수')
        ax2.set_title('계층별 샘플 수', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        return self._fig_to_html(fig, '계층별 데이터 분포')

    def _create_radar_chart(self, data: pd.DataFrame, features: List[str], target_var: str) -> str:
        """계층별 레이더 차트"""
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='polar')

        # 각 계층별 평균 계산
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]

        colors = ['#ffc107', '#ff9800', '#4caf50']

        for level in sorted(data['level'].unique()):
            level_data = data[data['level'] == level]
            values = []
            for feat in features:
                if feat in level_data.columns:
                    val = level_data[feat].mean()
                    # 정규화 (0-1)
                    max_val = data[feat].max()
                    if max_val > 0:
                        values.append(val / max_val)
                    else:
                        values.append(0)
                else:
                    values.append(0)

            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=f'레벨 {level}',
                   color=colors[level])
            ax.fill(angles, values, alpha=0.15, color=colors[level])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, size=10)
        ax.set_ylim(0, 1)
        ax.set_title('계층별 변수 프로파일 (정규화)', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        return self._fig_to_html(fig, '계층별 변수 프로파일')

    def _create_boxplots(self, data: pd.DataFrame, features: List[str], target_var: str) -> str:
        """계층별 박스플롯"""
        n_features = len(features)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        colors = ['#ffc107', '#ff9800', '#4caf50']

        for idx, feat in enumerate(features):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # 계층별 데이터 준비
            plot_data = []
            labels = []
            for level in sorted(data['level'].unique()):
                level_data = data[data['level'] == level][feat].dropna()
                if len(level_data) > 0:
                    plot_data.append(level_data)
                    labels.append(f'레벨 {level}')

            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title(feat, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        # 남은 subplot 제거
        for idx in range(len(features), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        return self._fig_to_html(fig, '계층별 변수 분포')

    def _create_correlation_heatmap(self, data: pd.DataFrame, features: List[str]) -> str:
        """변수 간 상관관계 히트맵"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # 수치형 데이터만 선택
        numeric_features = [f for f in features if data[f].dtype in ['int64', 'float64']][:10]
        corr = data[numeric_features].corr()

        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('변수 간 상관관계', fontsize=14, fontweight='bold', pad=20)

        return self._fig_to_html(fig, '변수 간 상관관계')

    def _fig_to_html(self, fig, title: str) -> str:
        """Figure를 HTML로 변환"""
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)

        return f'''
        <div class="chart-container">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{image_base64}" />
        </div>
        '''

    def _generate_model_comparison(self, results: Dict, summary_df: pd.DataFrame) -> str:
        """모델 성능 비교"""
        html = '<div class="section"><h2>🏆 모델 성능 비교</h2>'
        html += summary_df.to_html(index=False, classes='table', border=0)
        html += '</div>'
        return html

    def _generate_feature_importance(self, results: Dict) -> str:
        """특징 중요도"""
        html = '<div class="section"><h2>🔍 특징 중요도 분석</h2>'

        for model_name, result in results.items():
            if 'feature_importance' not in result or 'error' in result:
                continue

            importance = sorted(
                result['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            max_importance = max([imp for _, imp in importance]) if importance else 1

            html += f'<h3>{result["model_name"]}</h3>'
            html += '<div class="feature-importance">'

            for feature, imp in importance:
                width_pct = (imp / max_importance) * 100 if max_importance > 0 else 0
                html += f'''
                <div class="feature-bar">
                    <div class="label">{feature}</div>
                    <div class="bar" style="width: {width_pct}%;" data-value="{imp:.4f}"></div>
                </div>
                '''

            html += '</div>'

        html += '</div>'
        return html

    def _generate_detailed_analysis(self, data: pd.DataFrame, target_var: str) -> str:
        """상세 분석"""
        html = '<div class="section"><h2>📋 계층별 상세 통계</h2>'

        # 계층별 통계 테이블
        stats_data = []
        target_col = f'{target_var}_score'

        for level in sorted(data['level'].unique()):
            level_data = data[data['level'] == level]
            level_name = level_data['level_name'].iloc[0]

            stats_data.append({
                '계층': f'레벨 {level}',
                '이름': level_name,
                '샘플 수': len(level_data),
                f'{target_var} 평균': level_data[target_col].mean(),
                f'{target_var} 표준편차': level_data[target_col].std(),
                f'{target_var} 최소': level_data[target_col].min(),
                f'{target_var} 최대': level_data[target_col].max()
            })

        stats_df = pd.DataFrame(stats_data)
        html += stats_df.to_html(index=False, classes='table', border=0, float_format='%.3f')

        html += '</div>'
        return html
