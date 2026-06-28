"""
원통좌표 고유 형태 시각화 모듈
카테시안 변환 없이 원통 형태 그대로 시각화
"""
import pandas as pd
import numpy as np
from pathlib import Path


class CylindricalNativeVisualizer:
    """원통좌표를 원통 형태 그대로 시각화하는 클래스"""

    def __init__(self, data_dir: str = "hierarchical_data_cylinder",
                 output_dir: str = "model_results_cylinder"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_native_cylinder_visualization(self, target_var: str):
        """원통 형태 그대로 시각화 HTML 생성"""
        print(f"\n{'='*60}")
        print(f"🎨 {target_var.upper()} 원통 형태 시각화 생성")
        print(f"{'='*60}\n")

        # 데이터 로드
        data_file = self.data_dir / f"{target_var}_binary_classification_cylinder.csv"
        df = pd.read_csv(data_file)

        # 샘플링
        sample_size = 1500
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

        # ρ-φ-z 특성 추출
        rho_cols = [col for col in df_sample.columns if col.startswith('rho_')][:1]
        phi_cols = [col for col in df_sample.columns if col.startswith('phi_')][:1]
        z_cols = [col for col in df_sample.columns if col.startswith('z_')][:1]

        if not (rho_cols and phi_cols and z_cols):
            print("❌ 원통좌표 특성을 찾을 수 없습니다.")
            return None

        rho_col = rho_cols[0]
        phi_col = phi_cols[0]
        z_col = z_cols[0]

        print(f"📊 사용 특성: {rho_col}, {phi_col}, {z_col}")

        # 클래스별 데이터 분리
        df_class0 = df_sample[df_sample[f'{target_var}_binary'] == 0]
        df_class1 = df_sample[df_sample[f'{target_var}_binary'] == 1]

        # 원통 표면 메쉬 생성 (시각화 가이드용)
        max_rho = df_sample[rho_col].max()
        min_z = df_sample[z_col].min()
        max_z = df_sample[z_col].max()

        # 원통 표면을 위한 φ와 z 격자
        phi_mesh = np.linspace(0, 2*np.pi, 50)
        z_mesh = np.linspace(min_z, max_z, 30)
        phi_grid, z_grid = np.meshgrid(phi_mesh, z_mesh)

        # 원통 표면을 카테시안 좌표로 (표시용)
        rho_surface = max_rho * 0.8  # 최대값의 80%를 원통 반지름으로
        x_surface = rho_surface * np.cos(phi_grid)
        y_surface = rho_surface * np.sin(phi_grid)

        html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{target_var.upper()} - 원통 형태 시각화</title>
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 {target_var.upper()} - 원통 형태 시각화</h1>
        <p class="subtitle">Native Cylindrical Coordinate Visualization</p>

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
        </div>

        <div class="description">
            <h3 style="color: #f5576c; margin-bottom: 10px;">🔄 원통 형태 시각화란?</h3>
            <p>원통좌표 (ρ, φ, z)를 카테시안으로 변환하지 않고 <strong>원통 모양 그대로</strong> 표현합니다:</p>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li><strong>ρ (반지름)</strong>: 원통 중심축으로부터의 거리 → 점의 위치가 중심에서 멀어질수록 값이 큼</li>
                <li><strong>φ (각도)</strong>: 원통 둘레 방향 → 0°~360° 회전</li>
                <li><strong>z (높이)</strong>: 원통의 수직 높이 → 위아래 위치</li>
            </ul>
            <p style="margin-top: 10px;">원통 표면의 투명한 메쉬는 좌표계 가이드이며, 데이터 포인트의 위치가 원통좌표 값을 직접 나타냅니다.</p>
        </div>

        <div class="plot-container">
            <h2 class="plot-title">원통 형태 3D 시각화</h2>
            <p style="color: #666; margin-bottom: 15px;">
                마우스로 회전하여 다양한 각도에서 원통 구조를 확인할 수 있습니다.
                데이터 포인트의 색상: 파란색 = 미발생, 빨간색 = 발생
            </p>
            <div id="cylinder_plot"></div>
        </div>

        <script>
            // 원통 표면 메쉬 (투명)
            var cylinder_surface = {{
                type: 'surface',
                x: {x_surface.tolist()},
                y: {y_surface.tolist()},
                z: {z_grid.tolist()},
                opacity: 0.1,
                colorscale: [[0, 'lightgray'], [1, 'lightgray']],
                showscale: false,
                name: '원통 표면',
                hoverinfo: 'skip'
            }};

            // 미발생 그룹 데이터 (원통좌표 → 카테시안)
            var x0 = {(df_class0[rho_col] * np.cos(df_class0[phi_col])).tolist()};
            var y0 = {(df_class0[rho_col] * np.sin(df_class0[phi_col])).tolist()};
            var z0 = {df_class0[z_col].tolist()};
            var rho0 = {df_class0[rho_col].tolist()};
            var phi0 = {(np.degrees(df_class0[phi_col])).tolist()};

            var trace0 = {{
                type: 'scatter3d',
                x: x0,
                y: y0,
                z: z0,
                mode: 'markers',
                name: '미발생',
                marker: {{
                    size: 4,
                    color: '#3498db',
                    opacity: 0.7,
                    line: {{
                        width: 0.5,
                        color: '#2980b9'
                    }}
                }},
                text: rho0.map((r, i) => `ρ: ${{r.toFixed(2)}}<br>φ: ${{phi0[i].toFixed(1)}}°<br>z: ${{z0[i].toFixed(2)}}`),
                hoverinfo: 'text+name'
            }};

            // 발생 그룹 데이터 (원통좌표 → 카테시안)
            var x1 = {(df_class1[rho_col] * np.cos(df_class1[phi_col])).tolist()};
            var y1 = {(df_class1[rho_col] * np.sin(df_class1[phi_col])).tolist()};
            var z1 = {df_class1[z_col].tolist()};
            var rho1 = {df_class1[rho_col].tolist()};
            var phi1 = {(np.degrees(df_class1[phi_col])).tolist()};

            var trace1 = {{
                type: 'scatter3d',
                x: x1,
                y: y1,
                z: z1,
                mode: 'markers',
                name: '발생',
                marker: {{
                    size: 4,
                    color: '#e74c3c',
                    opacity: 0.7,
                    line: {{
                        width: 0.5,
                        color: '#c0392b'
                    }}
                }},
                text: rho1.map((r, i) => `ρ: ${{r.toFixed(2)}}<br>φ: ${{phi1[i].toFixed(1)}}°<br>z: ${{z1[i].toFixed(2)}}`),
                hoverinfo: 'text+name'
            }};

            var data = [cylinder_surface, trace0, trace1];

            var layout = {{
                scene: {{
                    xaxis: {{
                        title: 'X (ρ·cos(φ))',
                        showgrid: true,
                        zeroline: true
                    }},
                    yaxis: {{
                        title: 'Y (ρ·sin(φ))',
                        showgrid: true,
                        zeroline: true
                    }},
                    zaxis: {{
                        title: 'Z (높이)',
                        showgrid: true
                    }},
                    camera: {{
                        eye: {{x: 1.5, y: 1.5, z: 1.2}}
                    }},
                    aspectmode: 'cube'
                }},
                showlegend: true,
                legend: {{
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: -0.1,
                    xanchor: 'center',
                    x: 0.5
                }},
                height: 800,
                margin: {{t: 50, b: 50}}
            }};

            Plotly.newPlot('cylinder_plot', data, layout, {{responsive: true}});
        </script>

        <div class="description">
            <h3 style="color: #f5576c; margin-bottom: 10px;">💡 시각화 해석 가이드</h3>
            <ul style="margin-left: 20px;">
                <li><strong>원통 중심축</strong>: z축이 원통의 중심을 관통합니다</li>
                <li><strong>점의 위치</strong>:
                    <ul style="margin-left: 20px; margin-top: 5px;">
                        <li>중심에서의 거리 = ρ (반지름)</li>
                        <li>원통 둘레 위치 = φ (각도, 0°~360°)</li>
                        <li>수직 위치 = z (높이)</li>
                    </ul>
                </li>
                <li><strong>원통 표면</strong>: 투명한 회색 메쉬는 좌표계 경계를 나타냄</li>
                <li><strong>클래스 분리</strong>: 원통 공간에서 두 그룹이 분리되어 있을수록 예측 모델 성능이 우수</li>
            </ul>
        </div>

        <div class="description">
            <h3 style="color: #f5576c; margin-bottom: 10px;">🔍 원통좌표의 장점</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                <tr style="background: #f5f5f5;">
                    <th style="padding: 10px; border: 1px solid #ddd;">특징</th>
                    <th style="padding: 10px; border: 1px solid #ddd;">설명</th>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>물리적 직관성</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">원통 모양으로 자연스럽게 이해 가능</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>회전 대칭성</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">φ 방향으로 회전해도 구조 유지</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>독립적 분석</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">ρ(크기), φ(방향), z(높이)를 각각 분석 가능</td>
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
        output_file = self.output_dir / f"{target_var}_cylindrical_native_visualization.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ 저장 완료: {output_file.name}\n")

        return str(output_file)


def main():
    """메인 실행 함수"""
    visualizer = CylindricalNativeVisualizer()

    target_vars = ['anxiety', 'depression', 'stress']

    print(f"\n{'='*60}")
    print(f"🚀 원통 형태 시각화 시작")
    print(f"{'='*60}\n")

    for target in target_vars:
        try:
            visualizer.create_native_cylinder_visualization(target)
        except Exception as e:
            print(f"❌ {target} 시각화 중 오류: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"✅ 원통 형태 시각화 완료")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
