"""
원통좌표 변환 페노타입 시각화 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path


class CylindricalPhenotypeVisualizer:
    """원통좌표 변환 데이터의 페노타입 시각화 클래스"""

    def __init__(self, data_dir: str = "hierarchical_data_cylinder",
                 output_dir: str = "model_results_cylinder"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_comprehensive_visualization(self, target_var: str):
        """종합 시각화 HTML 생성"""
        print(f"\n{'='*60}")
        print(f"🎨 {target_var.upper()} 원통좌표 시각화 생성")
        print(f"{'='*60}\n")

        # 데이터 로드
        data_file = self.data_dir / f"{target_var}_binary_classification_cylinder.csv"
        df = pd.read_csv(data_file)

        # 샘플링
        sample_size = 2000
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

        # ρ-φ-z 특성 추출
        rho_cols = [col for col in df_sample.columns if col.startswith('rho_')][:2]
        phi_cols = [col for col in df_sample.columns if col.startswith('phi_')][:2]
        z_cols = [col for col in df_sample.columns if col.startswith('z_')][:2]

        print(f"📊 ρ 특성: {len(rho_cols)}개")
        print(f"📐 φ 특성: {len(phi_cols)}개")
        print(f"📏 z 특성: {len(z_cols)}개")

        html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{target_var.upper()} - 원통좌표 페노타입 시각화</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
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
            color: #f5576c;
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
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
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
            color: #f5576c;
            margin-bottom: 15px;
            font-size: 1.5em;
            font-weight: 600;
        }}

        .description {{
            background: #fff5f7;
            border-left: 4px solid #f5576c;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}

        .back-btn {{
            display: inline-block;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 12px 30px;
            border-radius: 8px;
            text-decoration: none;
            margin-top: 30px;
            font-weight: 600;
        }}

        .back-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(245, 87, 108, 0.4);
        }}

        .formula {{
            background: white;
            border: 2px solid #f5576c;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }}

        .formula code {{
            color: #f5576c;
            font-weight: bold;
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 {target_var.upper()} - 원통좌표 페노타입 시각화</h1>
        <p class="subtitle">Cylindrical Coordinate Phenotype Visualization</p>

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
                <div class="value">{len(rho_cols)}</div>
            </div>
        </div>

        <div class="description">
            <h3 style="color: #f5576c; margin-bottom: 10px;">📊 원통좌표 변환이란?</h3>
            <p>카테시안 좌표계 (x, y, z)의 세 특성을 원통좌표계 (ρ, φ, z)로 변환합니다:</p>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li><strong>ρ (rho)</strong>: xy 평면에서의 반지름 = √(x² + y²)</li>
                <li><strong>φ (phi)</strong>: xy 평면에서의 각도 = atan2(y, x)</li>
                <li><strong>z</strong>: z축 높이 (변환 없이 유지)</li>
            </ul>
            <p style="margin-top: 10px;">극좌표 + 높이 형태로, 구면좌표보다 직관적이며 물리적 의미가 명확합니다.</p>
        </div>

        <div class="formula">
            <h3 style="color: #f5576c; margin-bottom: 15px;">변환 공식</h3>
            <code>ρ = √(x² + y²)</code> (xy 평면 반지름)<br><br>
            <code>φ = atan2(y, x)</code> (xy 평면 각도)<br><br>
            <code>z = z</code> (높이, 변환 없음)
        </div>
"""

        # 각 특성 트리플에 대한 시각화 생성
        for idx, (rho_col, phi_col, z_col) in enumerate(zip(rho_cols, phi_cols, z_cols)):
            triple_name = rho_col.replace('rho_', '').replace('_', ' + ').upper()

            # 카테시안 좌표로 역변환 (3D 시각화용)
            # x = ρ * cos(φ)
            # y = ρ * sin(φ)
            # z = z (그대로)
            df_sample['x'] = df_sample[rho_col] * np.cos(df_sample[phi_col])
            df_sample['y'] = df_sample[rho_col] * np.sin(df_sample[phi_col])
            df_sample['z_plot'] = df_sample[z_col]

            # 클래스별 데이터 분리
            df_class0 = df_sample[df_sample[f'{target_var}_binary'] == 0]
            df_class1 = df_sample[df_sample[f'{target_var}_binary'] == 1]

            plot_id_3d = f"plot3d{idx + 1}"
            plot_id_polar = f"polar{idx + 1}"
            plot_id_z = f"z_dist{idx + 1}"

            html_content += f"""
        <div class="plot-container">
            <h2 class="plot-title">특성 트리플 {idx + 1}: {triple_name[:80]}</h2>

            <!-- 3D 원통좌표 시각화 -->
            <h3 style="color: #f5576c; margin: 20px 0 10px 0;">1. 3D 원통좌표 공간</h3>
            <p style="color: #666; margin-bottom: 15px;">원통좌표를 카테시안으로 변환하여 3D 시각화 (회전 가능)</p>
            <div id="{plot_id_3d}"></div>
        </div>

        <script>
            var trace0_3d = {{
                type: 'scatter3d',
                x: {df_class0['x'].tolist()},
                y: {df_class0['y'].tolist()},
                z: {df_class0['z_plot'].tolist()},
                mode: 'markers',
                name: '미발생',
                marker: {{
                    size: 3,
                    color: '#3498db',
                    opacity: 0.6,
                    line: {{width: 0}}
                }}
            }};

            var trace1_3d = {{
                type: 'scatter3d',
                x: {df_class1['x'].tolist()},
                y: {df_class1['y'].tolist()},
                z: {df_class1['z_plot'].tolist()},
                mode: 'markers',
                name: '발생',
                marker: {{
                    size: 3,
                    color: '#e74c3c',
                    opacity: 0.6,
                    line: {{width: 0}}
                }}
            }};

            var data_3d = [trace0_3d, trace1_3d];

            var layout_3d = {{
                scene: {{
                    xaxis: {{title: 'X (ρ·cos(φ))'}},
                    yaxis: {{title: 'Y (ρ·sin(φ))'}},
                    zaxis: {{title: 'Z (높이)'}}
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

            Plotly.newPlot('{plot_id_3d}', data_3d, layout_3d, {{responsive: true}});
        </script>

        <div class="plot-container">
            <!-- ρ-φ 극좌표 플롯 -->
            <h3 style="color: #f5576c; margin: 20px 0 10px 0;">2. ρ-φ 평면 (극좌표 투영)</h3>
            <p style="color: #666; margin-bottom: 15px;">xy 평면에 대한 극좌표 표현 (높이 정보 제외)</p>
            <div id="{plot_id_polar}"></div>
        </div>

        <script>
            var trace0_polar = {{
                type: 'scatterpolar',
                r: {df_class0[rho_col].tolist()},
                theta: {(np.degrees(df_class0[phi_col])).tolist()},
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
                r: {df_class1[rho_col].tolist()},
                theta: {(np.degrees(df_class1[phi_col])).tolist()},
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
                        title: 'ρ (xy 평면 반지름)',
                        showgrid: true
                    }},
                    angularaxis: {{
                        title: 'φ (각도)',
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

            Plotly.newPlot('{plot_id_polar}', data_polar, layout_polar, {{responsive: true}});
        </script>

        <div class="plot-container">
            <!-- z 분포 히스토그램 -->
            <h3 style="color: #f5576c; margin: 20px 0 10px 0;">3. Z축 (높이) 분포</h3>
            <p style="color: #666; margin-bottom: 15px;">원통좌표의 높이 성분 분포 비교</p>
            <div id="{plot_id_z}"></div>
        </div>

        <script>
            var trace0_z = {{
                type: 'histogram',
                x: {df_class0[z_col].tolist()},
                name: '미발생',
                opacity: 0.6,
                marker: {{
                    color: '#3498db'
                }},
                nbinsx: 30
            }};

            var trace1_z = {{
                type: 'histogram',
                x: {df_class1[z_col].tolist()},
                name: '발생',
                opacity: 0.6,
                marker: {{
                    color: '#e74c3c'
                }},
                nbinsx: 30
            }};

            var data_z = [trace0_z, trace1_z];

            var layout_z = {{
                barmode: 'overlay',
                xaxis: {{title: 'Z (높이)'}},
                yaxis: {{title: '빈도'}},
                showlegend: true,
                legend: {{
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: -0.2,
                    xanchor: 'center',
                    x: 0.5
                }},
                height: 400,
                margin: {{t: 50, b: 50}}
            }};

            Plotly.newPlot('{plot_id_z}', data_z, layout_z, {{responsive: true}});
        </script>
"""

        html_content += """
        <div class="description">
            <h3 style="color: #f5576c; margin-bottom: 10px;">💡 시각화 해석</h3>
            <ul style="margin-left: 20px;">
                <li><strong>파란색</strong>: 증상 미발생 그룹</li>
                <li><strong>빨간색</strong>: 증상 발생 그룹</li>
                <li><strong>3D 플롯</strong>: 원통좌표의 전체 구조 (ρ, φ, z)</li>
                <li><strong>ρ-φ 극좌표</strong>: xy 평면에서의 분포 (높이 무시)</li>
                <li><strong>z 분포</strong>: 높이 성분의 클래스별 차이</li>
            </ul>
            <p style="margin-top: 10px;">
                두 그룹이 3차원 공간에서 분리되어 있을수록 원통좌표 변환이 효과적으로 패턴을 포착했음을 의미합니다.
            </p>
        </div>

        <div class="description">
            <h3 style="color: #f5576c; margin-bottom: 10px;">🔍 원통좌표의 장점</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                <tr style="background: #f5f5f5;">
                    <th style="padding: 10px; border: 1px solid #ddd;">특징</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">설명</th>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>직관성</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">극좌표 + 높이로 물리적 의미가 명확</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>균형성</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">극좌표와 구면좌표의 중간 성능</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>분리성</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">평면(ρ, φ)과 높이(z)를 독립적으로 분석 가능</td>
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
        output_file = self.output_dir / f"{target_var}_cylindrical_phenotype_visualization.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ 저장 완료: {output_file.name}\n")

        return str(output_file)


def main():
    """메인 실행 함수"""
    visualizer = CylindricalPhenotypeVisualizer()

    target_vars = ['anxiety', 'depression', 'stress']

    print(f"\n{'='*60}")
    print(f"🚀 원통좌표 페노타입 시각화 시작")
    print(f"{'='*60}\n")

    for target in target_vars:
        try:
            visualizer.create_comprehensive_visualization(target)
        except Exception as e:
            print(f"❌ {target} 시각화 중 오류: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"✅ 원통좌표 페노타입 시각화 완료")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
