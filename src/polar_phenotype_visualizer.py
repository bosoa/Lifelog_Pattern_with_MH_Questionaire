"""
극좌표 변환 페노타입 시각화 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class PolarPhenotypeVisualizer:
    """극좌표 변환 데이터의 페노타입 시각화 클래스"""

    def __init__(self, data_dir: str = "hierarchical_data_polar",
                 output_dir: str = "model_results_polar"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_polar_scatter_plot(self, target_var: str, sample_size: int = 5000):
        """
        극좌표 r-θ 산점도 생성

        Args:
            target_var: 타겟 변수 ('anxiety', 'depression', 'stress')
            sample_size: 샘플링할 데이터 개수
        """
        print(f"\n{'='*60}")
        print(f"📊 {target_var.upper()} 극좌표 페노타입 시각화")
        print(f"{'='*60}\n")

        # 데이터 로드
        data_file = self.data_dir / f"{target_var}_binary_classification_polar.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {data_file}")

        df = pd.read_csv(data_file)
        print(f"📂 데이터 로드: {len(df):,}행")

        # 샘플링 (시각화 성능을 위해)
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"🎲 샘플링: {sample_size:,}행")
        else:
            df_sample = df

        # r-θ 특성 추출
        r_cols = [col for col in df_sample.columns if col.startswith('r_') and not col.startswith('r_theta')]
        theta_cols = [col for col in df_sample.columns if col.startswith('theta_')]

        print(f"📈 r 특성: {len(r_cols)}개")
        print(f"📐 θ 특성: {len(theta_cols)}개")

        # 서브플롯 생성
        n_pairs = min(len(r_cols), len(theta_cols))
        fig = make_subplots(
            rows=2, cols=min(3, n_pairs),
            subplot_titles=[f"특성 페어 {i+1}" for i in range(min(6, n_pairs))],
            specs=[[{'type': 'polar'}] * min(3, n_pairs)] * 2
        )

        # 색상 매핑
        color_map = {0: '#3498db', 1: '#e74c3c'}
        label_map = {0: '미발생', 1: '발생'}

        # 각 특성 페어에 대해 극좌표 플롯 생성
        for idx in range(min(6, n_pairs)):
            row = (idx // 3) + 1
            col = (idx % 3) + 1

            r_col = r_cols[idx]
            theta_col = theta_cols[idx]

            # 클래스별로 나누어서 플롯
            for class_val in [0, 1]:
                df_class = df_sample[df_sample[f'{target_var}_binary'] == class_val]

                fig.add_trace(
                    go.Scatterpolar(
                        r=df_class[r_col],
                        theta=np.degrees(df_class[theta_col]),  # 라디안을 도(degree)로 변환
                        mode='markers',
                        name=label_map[class_val],
                        marker=dict(
                            size=4,
                            color=color_map[class_val],
                            opacity=0.6
                        ),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )

        # 레이아웃 설정
        fig.update_layout(
            title=f"{target_var.upper()} - 극좌표 페노타입 분포",
            height=800,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def create_comprehensive_visualization(self, target_var: str):
        """종합 시각화 HTML 생성"""
        print(f"\n{'='*60}")
        print(f"🎨 {target_var.upper()} 종합 시각화 생성")
        print(f"{'='*60}\n")

        # 데이터 로드
        data_file = self.data_dir / f"{target_var}_binary_classification_polar.csv"
        df = pd.read_csv(data_file)

        # 샘플링
        sample_size = 3000
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

        # r-θ 특성 추출
        r_cols = [col for col in df_sample.columns if col.startswith('r_') and not col.startswith('r_theta')][:3]
        theta_cols = [col for col in df_sample.columns if col.startswith('theta_')][:3]

        html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{target_var.upper()} - 극좌표 페노타입 시각화</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}

        h1 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-align: center;
        }}

        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 1.1em;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}

        .stat-card h3 {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }}

        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}

        .plot-container {{
            margin: 30px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
        }}

        .plot-title {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
            font-weight: 600;
        }}

        .description {{
            background: #e8eaf6;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .back-btn {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border-radius: 8px;
            text-decoration: none;
            margin-top: 30px;
            font-weight: 600;
        }}

        .back-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌐 {target_var.upper()} - 극좌표 페노타입 시각화</h1>
        <p class="subtitle">Polar Coordinate Phenotype Visualization</p>

        <div class="stats">
            <div class="stat-card">
                <h3>전체 샘플</h3>
                <div class="value">{len(df):,}</div>
            </div>
            <div class="stat-card">
                <h3>시각화 샘플</h3>
                <div class="value">{len(df_sample):,}</div>
            </div>
            <div class="stat-card">
                <h3>발생 비율</h3>
                <div class="value">{df[f'{target_var}_binary'].sum() / len(df) * 100:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>특성 페어</h3>
                <div class="value">{len(r_cols)}</div>
            </div>
        </div>

        <div class="description">
            <h3 style="color: #667eea; margin-bottom: 10px;">📊 극좌표 변환이란?</h3>
            <p>카테시안 좌표계 (x, y)의 두 특성을 극좌표계 (r, θ)로 변환합니다:</p>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li><strong>r (반지름)</strong>: 두 특성의 결합된 크기 = √(x² + y²)</li>
                <li><strong>θ (각도)</strong>: 두 특성 간의 상대적 관계 = atan2(y, x)</li>
            </ul>
            <p style="margin-top: 10px;">이 변환을 통해 비선형 관계와 특성 간 상호작용을 더 잘 포착할 수 있습니다.</p>
        </div>
"""

        # 각 특성 페어에 대한 플롯 생성
        for idx, (r_col, theta_col) in enumerate(zip(r_cols, theta_cols)):
            pair_name = r_col.replace('r_', '').replace('_', ' + ').upper()

            # 클래스별 데이터 분리
            df_class0 = df_sample[df_sample[f'{target_var}_binary'] == 0]
            df_class1 = df_sample[df_sample[f'{target_var}_binary'] == 1]

            plot_id = f"plot{idx + 1}"

            html_content += f"""
        <div class="plot-container">
            <h2 class="plot-title">특성 페어 {idx + 1}: {pair_name}</h2>
            <div id="{plot_id}"></div>
        </div>

        <script>
            var trace0 = {{
                type: 'scatterpolar',
                r: {df_class0[r_col].tolist()},
                theta: {(np.degrees(df_class0[theta_col])).tolist()},
                mode: 'markers',
                name: '미발생',
                marker: {{
                    size: 4,
                    color: '#3498db',
                    opacity: 0.6
                }}
            }};

            var trace1 = {{
                type: 'scatterpolar',
                r: {df_class1[r_col].tolist()},
                theta: {(np.degrees(df_class1[theta_col])).tolist()},
                mode: 'markers',
                name: '발생',
                marker: {{
                    size: 4,
                    color: '#e74c3c',
                    opacity: 0.6
                }}
            }};

            var data = [trace0, trace1];

            var layout = {{
                polar: {{
                    radialaxis: {{
                        title: 'r (크기)',
                        showgrid: true
                    }},
                    angularaxis: {{
                        title: 'θ (각도)',
                        showgrid: true,
                        direction: 'counterclockwise'
                    }}
                }},
                showlegend: true,
                legend: {{
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: -0.2,
                    xanchor: 'center',
                    x: 0.5
                }},
                height: 600,
                margin: {{t: 50, b: 50}}
            }};

            Plotly.newPlot('{plot_id}', data, layout, {{responsive: true}});
        </script>
"""

        html_content += """
        <div class="description">
            <h3 style="color: #667eea; margin-bottom: 10px;">💡 시각화 해석</h3>
            <ul style="margin-left: 20px;">
                <li><strong>파란색 점</strong>: 증상 미발생 그룹</li>
                <li><strong>빨간색 점</strong>: 증상 발생 그룹</li>
                <li><strong>r 축</strong>: 원점으로부터의 거리 (특성들의 종합 크기)</li>
                <li><strong>θ 축</strong>: 각도 (특성들 간의 상대적 관계)</li>
            </ul>
            <p style="margin-top: 10px;">두 그룹이 공간적으로 분리되어 있을수록 예측 모델의 성능이 우수합니다.</p>
        </div>

        <div style="text-align: center;">
            <a href="index.html" class="back-btn">← 대시보드로 돌아가기</a>
        </div>
    </div>
</body>
</html>
"""

        # 저장
        output_file = self.output_dir / f"{target_var}_polar_phenotype_visualization.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ 저장 완료: {output_file.name}\n")

        return str(output_file)


def main():
    """메인 실행 함수"""
    visualizer = PolarPhenotypeVisualizer()

    target_vars = ['anxiety', 'depression', 'stress']

    print(f"\n{'='*60}")
    print(f"🚀 극좌표 페노타입 시각화 시작")
    print(f"{'='*60}\n")

    for target in target_vars:
        try:
            visualizer.create_comprehensive_visualization(target)
        except Exception as e:
            print(f"❌ {target} 시각화 중 오류: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"✅ 극좌표 페노타입 시각화 완료")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
