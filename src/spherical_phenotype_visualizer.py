"""
구면좌표 변환 페노타입 시각화 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path


class SphericalPhenotypeVisualizer:
    """구면좌표 변환 데이터의 페노타입 시각화 클래스"""

    def __init__(self, data_dir: str = "hierarchical_data_sphere",
                 output_dir: str = "model_results_sphere"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_comprehensive_visualization(self, target_var: str):
        """종합 시각화 HTML 생성"""
        print(f"\n{'='*60}")
        print(f"🎨 {target_var.upper()} 구면좌표 시각화 생성")
        print(f"{'='*60}\n")

        # 데이터 로드
        data_file = self.data_dir / f"{target_var}_binary_classification_sphere.csv"
        df = pd.read_csv(data_file)

        # 샘플링
        sample_size = 2000
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

        # r-θ-φ 특성 추출
        r_cols = [col for col in df_sample.columns if col.startswith('r_')][:2]
        theta_cols = [col for col in df_sample.columns if col.startswith('theta_')][:2]
        phi_cols = [col for col in df_sample.columns if col.startswith('phi_')][:2]

        print(f"📊 r 특성: {len(r_cols)}개")
        print(f"📐 θ 특성: {len(theta_cols)}개")
        print(f"🎯 φ 특성: {len(phi_cols)}개")

        html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{target_var.upper()} - 구면좌표 페노타입 시각화</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
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
            color: #11998e;
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
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
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
            color: #11998e;
            margin-bottom: 15px;
            font-size: 1.5em;
            font-weight: 600;
        }}

        .description {{
            background: #e0f2f1;
            border-left: 4px solid #11998e;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .back-btn {{
            display: inline-block;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 12px 30px;
            border-radius: 8px;
            text-decoration: none;
            margin-top: 30px;
            font-weight: 600;
        }}

        .back-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(17, 153, 142, 0.4);
        }}

        .formula {{
            background: white;
            border: 2px solid #11998e;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }}

        .formula code {{
            color: #11998e;
            font-weight: bold;
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌐 {target_var.upper()} - 구면좌표 페노타입 시각화</h1>
        <p class="subtitle">Spherical Coordinate Phenotype Visualization (3D)</p>

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
                <h3>특성 트리플</h3>
                <div class="value">{len(r_cols)}</div>
            </div>
        </div>

        <div class="description">
            <h3 style="color: #11998e; margin-bottom: 10px;">📊 구면좌표 변환이란?</h3>
            <p>카테시안 좌표계 (x, y, z)의 세 특성을 구면좌표계 (r, θ, φ)로 변환합니다:</p>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li><strong>r (반지름)</strong>: 세 특성의 결합된 크기 = √(x² + y² + z²)</li>
                <li><strong>θ (방위각)</strong>: xy 평면에서의 각도 = atan2(y, x)</li>
                <li><strong>φ (극각)</strong>: z축으로부터의 각도 = arccos(z / r)</li>
            </ul>
            <p style="margin-top: 10px;">3차원 변환을 통해 더 복잡한 특성 간 관계를 포착할 수 있습니다.</p>
        </div>

        <div class="formula">
            <h3 style="color: #11998e; margin-bottom: 15px;">변환 공식</h3>
            <code>r = √(x² + y² + z²)</code><br><br>
            <code>θ = atan2(y, x)</code><br><br>
            <code>φ = arccos(z / r)</code>
        </div>
"""

        # 각 특성 트리플에 대한 3D 플롯 생성
        for idx, (r_col, theta_col, phi_col) in enumerate(zip(r_cols, theta_cols, phi_cols)):
            triple_name = r_col.replace('r_', '').replace('_', ' + ').upper()

            # 구면좌표를 카테시안 좌표로 변환하여 3D 플롯 생성
            # x = r * sin(φ) * cos(θ)
            # y = r * sin(φ) * sin(θ)
            # z = r * cos(φ)

            df_sample['x'] = df_sample[r_col] * np.sin(df_sample[phi_col]) * np.cos(df_sample[theta_col])
            df_sample['y'] = df_sample[r_col] * np.sin(df_sample[phi_col]) * np.sin(df_sample[theta_col])
            df_sample['z'] = df_sample[r_col] * np.cos(df_sample[phi_col])

            # 클래스별 데이터 분리
            df_class0 = df_sample[df_sample[f'{target_var}_binary'] == 0]
            df_class1 = df_sample[df_sample[f'{target_var}_binary'] == 1]

            plot_id = f"plot3d{idx + 1}"

            html_content += f"""
        <div class="plot-container">
            <h2 class="plot-title">특성 트리플 {idx + 1}: {triple_name[:80]}</h2>
            <p style="color: #666; margin-bottom: 15px;">3D 인터랙티브 시각화 - 마우스로 회전 및 줌 가능</p>
            <div id="{plot_id}"></div>
        </div>

        <script>
            var trace0 = {{
                type: 'scatter3d',
                x: {df_class0['x'].tolist()},
                y: {df_class0['y'].tolist()},
                z: {df_class0['z'].tolist()},
                mode: 'markers',
                name: '미발생',
                marker: {{
                    size: 3,
                    color: '#3498db',
                    opacity: 0.6,
                    line: {{
                        width: 0
                    }}
                }}
            }};

            var trace1 = {{
                type: 'scatter3d',
                x: {df_class1['x'].tolist()},
                y: {df_class1['y'].tolist()},
                z: {df_class1['z'].tolist()},
                mode: 'markers',
                name: '발생',
                marker: {{
                    size: 3,
                    color: '#e74c3c',
                    opacity: 0.6,
                    line: {{
                        width: 0
                    }}
                }}
            }};

            var data = [trace0, trace1];

            var layout = {{
                scene: {{
                    xaxis: {{title: 'X (r·sin(φ)·cos(θ))'}},
                    yaxis: {{title: 'Y (r·sin(φ)·sin(θ))'}},
                    zaxis: {{title: 'Z (r·cos(φ))'}}
                }},
                showlegend: true,
                legend: {{
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: -0.1,
                    xanchor: 'center',
                    x: 0.5
                }},
                height: 700,
                margin: {{t: 50, b: 50}}
            }};

            Plotly.newPlot('{plot_id}', data, layout, {{responsive: true}});
        </script>
"""

        # 극좌표 형식의 2D 플롯도 추가 (r-θ)
        html_content += f"""
        <div class="plot-container">
            <h2 class="plot-title">방위각 분포 (r-θ 평면)</h2>
            <p style="color: #666; margin-bottom: 15px;">구면좌표의 방위각 성분을 2D로 시각화</p>
            <div id="polar_plot"></div>
        </div>

        <script>
            var trace0_polar = {{
                type: 'scatterpolar',
                r: {df_class0[r_cols[0]].tolist()},
                theta: {(np.degrees(df_class0[theta_cols[0]])).tolist()},
                mode: 'markers',
                name: '미발생',
                marker: {{
                    size: 4,
                    color: '#3498db',
                    opacity: 0.6
                }}
            }};

            var trace1_polar = {{
                type: 'scatterpolar',
                r: {df_class1[r_cols[0]].tolist()},
                theta: {(np.degrees(df_class1[theta_cols[0]])).tolist()},
                mode: 'markers',
                name: '발생',
                marker: {{
                    size: 4,
                    color: '#e74c3c',
                    opacity: 0.6
                }}
            }};

            var data_polar = [trace0_polar, trace1_polar];

            var layout_polar = {{
                polar: {{
                    radialaxis: {{
                        title: 'r (반지름)',
                        showgrid: true
                    }},
                    angularaxis: {{
                        title: 'θ (방위각)',
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

            Plotly.newPlot('polar_plot', data_polar, layout_polar, {{responsive: true}});
        </script>
"""

        html_content += """
        <div class="description">
            <h3 style="color: #11998e; margin-bottom: 10px;">💡 시각화 해석</h3>
            <ul style="margin-left: 20px;">
                <li><strong>파란색 점</strong>: 증상 미발생 그룹</li>
                <li><strong>빨간색 점</strong>: 증상 발생 그룹</li>
                <li><strong>3D 플롯</strong>: 구면좌표를 카테시안 좌표로 역변환하여 시각화</li>
                <li><strong>2D 극좌표 플롯</strong>: r-θ 평면만 추출하여 방위각 분포 확인</li>
            </ul>
            <p style="margin-top: 10px;">
                3D 공간에서 두 그룹이 분리되어 있을수록 구면좌표 변환이 효과적으로 패턴을 포착했음을 의미합니다.
            </p>
        </div>

        <div class="description">
            <h3 style="color: #11998e; margin-bottom: 10px;">🔍 극좌표 vs 구면좌표</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                <tr style="background: #f5f5f5;">
                    <th style="padding: 10px; border: 1px solid #ddd;">항목</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">극좌표 (2D)</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">구면좌표 (3D)</th>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>특성 그룹</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">2개 (페어)</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">3개 (트리플)</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>변환 결과</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">r, θ</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">r, θ, φ</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>표현력</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">2개 특성 간 관계</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">3개 특성 간 복합 관계</td>
                </tr>
            </table>
        </div>

        <div style="text-align: center;">
            <a href="index.html" class="back-btn">← 대시보드로 돌아가기</a>
        </div>
    </div>
</body>
</html>
"""

        # 저장
        output_file = self.output_dir / f"{target_var}_spherical_phenotype_visualization.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ 저장 완료: {output_file.name}\n")

        return str(output_file)


def main():
    """메인 실행 함수"""
    visualizer = SphericalPhenotypeVisualizer()

    target_vars = ['anxiety', 'depression', 'stress']

    print(f"\n{'='*60}")
    print(f"🚀 구면좌표 페노타입 시각화 시작")
    print(f"{'='*60}\n")

    for target in target_vars:
        try:
            visualizer.create_comprehensive_visualization(target)
        except Exception as e:
            print(f"❌ {target} 시각화 중 오류: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"✅ 구면좌표 페노타입 시각화 완료")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
