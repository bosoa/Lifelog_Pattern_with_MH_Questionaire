"""
최고 성능 모델(표준화 구면좌표)의 이벤트 발생/미발생 차이 시각화
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_best_model_data():
    """가장 성능이 좋은 모델 데이터 로드 (Stress - 표준화 구면좌표)"""
    data_path = Path('data_splits_standardized_sphere/stress_train_standardized_spherical.csv')

    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

    df = pd.read_csv(data_path)
    print(f"✅ 데이터 로드 완료: {len(df):,}행 x {len(df.columns)}열")
    print(f"   컬럼: {list(df.columns)}")

    return df

def extract_representative_samples(df, target_col='stress_binary', n_samples=50):
    """
    이벤트 발생/미발생 그룹의 대표 샘플 추출

    각 그룹의 중심에 가까운 샘플들 선택
    """
    # 이벤트 발생/미발생 분리
    event_occurred = df[df[target_col] == 1].copy()
    event_not_occurred = df[df[target_col] == 0].copy()

    print(f"\n📊 데이터 분포:")
    print(f"   이벤트 발생: {len(event_occurred):,}개 ({len(event_occurred)/len(df)*100:.1f}%)")
    print(f"   이벤트 미발생: {len(event_not_occurred):,}개 ({len(event_not_occurred)/len(df)*100:.1f}%)")

    # 구면좌표 특성 추출
    coord_cols = [col for col in df.columns if col.startswith('r_') or col.startswith('theta_') or col.startswith('phi_')]
    print(f"\n🌍 구면좌표 특성: {len(coord_cols)}개")
    print(f"   {coord_cols[:5]}...")

    # 각 그룹의 중심 계산
    center_occurred = event_occurred[coord_cols].mean()
    center_not_occurred = event_not_occurred[coord_cols].mean()

    # 중심으로부터의 거리 계산
    event_occurred['distance_to_center'] = np.sqrt(
        ((event_occurred[coord_cols] - center_occurred) ** 2).sum(axis=1)
    )
    event_not_occurred['distance_to_center'] = np.sqrt(
        ((event_not_occurred[coord_cols] - center_not_occurred) ** 2).sum(axis=1)
    )

    # 중심에 가까운 샘플 선택
    repr_occurred = event_occurred.nsmallest(n_samples, 'distance_to_center')
    repr_not_occurred = event_not_occurred.nsmallest(n_samples, 'distance_to_center')

    print(f"\n✅ 대표 샘플 추출 완료:")
    print(f"   이벤트 발생: {len(repr_occurred)}개")
    print(f"   이벤트 미발생: {len(repr_not_occurred)}개")

    return repr_occurred, repr_not_occurred, coord_cols

def create_3d_spherical_visualization(repr_occurred, repr_not_occurred, coord_cols):
    """3D 구면좌표 시각화"""

    # 첫 번째 구면좌표 트리플 사용 (r, theta, phi)
    r_cols = [col for col in coord_cols if col.startswith('r_')]
    theta_cols = [col for col in coord_cols if col.startswith('theta_')]
    phi_cols = [col for col in coord_cols if col.startswith('phi_')]

    if len(r_cols) == 0 or len(theta_cols) == 0 or len(phi_cols) == 0:
        print("⚠️ 구면좌표 컬럼을 찾을 수 없습니다.")
        return None

    # 첫 번째 트리플 사용
    r_col = r_cols[0]
    theta_col = theta_cols[0]
    phi_col = phi_cols[0]

    print(f"\n📐 사용할 좌표: {r_col}, {theta_col}, {phi_col}")

    # 그림 생성
    fig = plt.figure(figsize=(20, 15))

    # 1. 3D 산점도
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    # 구면좌표를 카테시안으로 변환
    def spherical_to_cartesian(r, theta, phi):
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return x, y, z

    # 이벤트 발생 그룹
    x_occ, y_occ, z_occ = spherical_to_cartesian(
        repr_occurred[r_col].values,
        repr_occurred[theta_col].values,
        repr_occurred[phi_col].values
    )

    # 이벤트 미발생 그룹
    x_not, y_not, z_not = spherical_to_cartesian(
        repr_not_occurred[r_col].values,
        repr_not_occurred[theta_col].values,
        repr_not_occurred[phi_col].values
    )

    ax1.scatter(x_occ, y_occ, z_occ, c='red', marker='o', s=50, alpha=0.6, label='이벤트 발생')
    ax1.scatter(x_not, y_not, z_not, c='blue', marker='^', s=50, alpha=0.6, label='이벤트 미발생')

    ax1.set_xlabel('X (카테시안)', fontsize=10)
    ax1.set_ylabel('Y (카테시안)', fontsize=10)
    ax1.set_zlabel('Z (카테시안)', fontsize=10)
    ax1.set_title('3D 공간 분포 (구면 → 카테시안 변환)', fontsize=12, fontweight='bold')
    ax1.legend()

    # 2. 반지름(r) 분포
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist([repr_occurred[r_col].values, repr_not_occurred[r_col].values],
             bins=30, label=['이벤트 발생', '이벤트 미발생'],
             color=['red', 'blue'], alpha=0.6)
    ax2.set_xlabel('반지름 (r)', fontsize=10)
    ax2.set_ylabel('빈도', fontsize=10)
    ax2.set_title('반지름(r) 분포 비교', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. 방위각(theta) 분포
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist([repr_occurred[theta_col].values, repr_not_occurred[theta_col].values],
             bins=30, label=['이벤트 발생', '이벤트 미발생'],
             color=['red', 'blue'], alpha=0.6)
    ax3.set_xlabel('방위각 θ (라디안)', fontsize=10)
    ax3.set_ylabel('빈도', fontsize=10)
    ax3.set_title('방위각(θ) 분포 비교', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. 극각(phi) 분포
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist([repr_occurred[phi_col].values, repr_not_occurred[phi_col].values],
             bins=30, label=['이벤트 발생', '이벤트 미발생'],
             color=['red', 'blue'], alpha=0.6)
    ax4.set_xlabel('극각 φ (라디안)', fontsize=10)
    ax4.set_ylabel('빈도', fontsize=10)
    ax4.set_title('극각(φ) 분포 비교', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. 박스플롯 비교
    ax5 = fig.add_subplot(2, 3, 5)
    data_box = [
        repr_occurred[r_col].values,
        repr_not_occurred[r_col].values,
        repr_occurred[theta_col].values,
        repr_not_occurred[theta_col].values,
        repr_occurred[phi_col].values,
        repr_not_occurred[phi_col].values
    ]
    positions = [1, 2, 4, 5, 7, 8]
    bp = ax5.boxplot(data_box, positions=positions, widths=0.6,
                     patch_artist=True, labels=[
                         'r (발생)', 'r (미발생)',
                         'θ (발생)', 'θ (미발생)',
                         'φ (발생)', 'φ (미발생)'
                     ])

    # 색상 설정
    colors = ['red', 'blue', 'red', 'blue', 'red', 'blue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax5.set_ylabel('값', fontsize=10)
    ax5.set_title('구면좌표 성분 박스플롯', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 6. 통계 요약 테이블
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    stats_data = []
    for coord, name in [(r_col, 'r'), (theta_col, 'θ'), (phi_col, 'φ')]:
        mean_occ = repr_occurred[coord].mean()
        mean_not = repr_not_occurred[coord].mean()
        std_occ = repr_occurred[coord].std()
        std_not = repr_not_occurred[coord].std()

        stats_data.append([
            name,
            f"{mean_occ:.3f} ± {std_occ:.3f}",
            f"{mean_not:.3f} ± {std_not:.3f}",
            f"{mean_occ - mean_not:+.3f}"
        ])

    table = ax6.table(cellText=stats_data,
                      colLabels=['좌표', '이벤트 발생\n(평균 ± 표준편차)', '이벤트 미발생\n(평균 ± 표준편차)', '차이'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 헤더 스타일
    for i in range(4):
        table[(0, i)].set_facecolor('#667eea')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 데이터 행 스타일
    for i in range(1, 4):
        for j in range(4):
            if j == 3:  # 차이 컬럼
                diff_val = float(stats_data[i-1][3])
                if diff_val > 0:
                    table[(i, j)].set_facecolor('#ffcccc')
                else:
                    table[(i, j)].set_facecolor('#ccccff')

    ax6.set_title('통계 요약', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    return fig

def create_html_report(repr_occurred, repr_not_occurred, coord_cols, output_dir='visualization_results'):
    """HTML 리포트 생성"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 시각화 생성
    print("\n📊 시각화 생성 중...")
    fig = create_3d_spherical_visualization(repr_occurred, repr_not_occurred, coord_cols)

    if fig is None:
        print("❌ 시각화 생성 실패")
        return

    # 이미지 저장
    img_file = output_path / 'coordinate_comparison.png'
    fig.savefig(img_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 이미지 저장: {img_file}")

    # 추가 분석
    r_cols = [col for col in coord_cols if col.startswith('r_')]
    theta_cols = [col for col in coord_cols if col.startswith('theta_')]
    phi_cols = [col for col in coord_cols if col.startswith('phi_')]

    # 통계적 차이 계산
    differences = []
    for coord in coord_cols:
        mean_occ = repr_occurred[coord].mean()
        mean_not = repr_not_occurred[coord].mean()
        diff = mean_occ - mean_not
        diff_pct = (diff / mean_not * 100) if mean_not != 0 else 0

        differences.append({
            'coordinate': coord,
            'mean_occurred': mean_occ,
            'mean_not_occurred': mean_not,
            'difference': diff,
            'difference_pct': diff_pct
        })

    # 가장 큰 차이를 보이는 좌표
    top_diffs = sorted(differences, key=lambda x: abs(x['difference']), reverse=True)[:5]

    # HTML 생성
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>좌표계 이벤트 발생/미발생 차이 분석</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section-title {{
            font-size: 2em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}

        .highlight-box {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-left: 5px solid #667eea;
            padding: 25px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .comparison-box {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}

        .event-card {{
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}

        .event-occurred {{
            background: linear-gradient(135deg, #ffcccc 0%, #ff9999 100%);
            border-left: 5px solid #e74c3c;
        }}

        .event-not-occurred {{
            background: linear-gradient(135deg, #ccccff 0%, #9999ff 100%);
            border-left: 5px solid #3498db;
        }}

        .metric {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}

        .metric-label {{
            color: #555;
            font-size: 0.9em;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: bold;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}

        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}

        tr:hover {{
            background-color: #e3f2fd;
            transition: background-color 0.3s;
        }}

        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}

        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        ul {{
            margin-left: 20px;
        }}

        li {{
            margin: 10px 0;
            line-height: 1.8;
        }}

        .positive {{
            color: #e74c3c;
            font-weight: bold;
        }}

        .negative {{
            color: #3498db;
            font-weight: bold;
        }}

        footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌍 좌표계 이벤트 발생/미발생 차이 분석</h1>
            <p>표준화 구면좌표 기반 Stress 예측 (C-index: 0.7216)</p>
            <p style="font-size: 0.9em; margin-top: 10px;">분석 일자: 2026-06-27</p>
        </header>

        <div class="content">
            <!-- 요약 -->
            <div class="section">
                <h2 class="section-title">📊 분석 개요</h2>

                <div class="highlight-box">
                    <h3>🎯 분석 목표</h3>
                    <p>최고 성능 모델(표준화 구면좌표, C-index 0.7216)에서 Stress 이벤트 발생/미발생 그룹의
                    좌표계 차이를 시각화하여 예측 모델의 작동 원리를 이해합니다.</p>
                </div>

                <div class="comparison-box">
                    <div class="event-card event-occurred">
                        <h3>🔴 이벤트 발생</h3>
                        <div class="metric">{len(repr_occurred)}</div>
                        <div class="metric-label">대표 샘플 수</div>
                        <p style="margin-top: 10px;">Stress 점수 ≥ 4점 (고위험군)</p>
                    </div>

                    <div class="event-card event-not-occurred">
                        <h3>🔵 이벤트 미발생</h3>
                        <div class="metric">{len(repr_not_occurred)}</div>
                        <div class="metric-label">대표 샘플 수</div>
                        <p style="margin-top: 10px;">Stress 점수 < 4점 (정상군)</p>
                    </div>
                </div>
            </div>

            <!-- 시각화 -->
            <div class="section">
                <h2 class="section-title">📈 좌표계 시각화</h2>

                <div class="chart-container">
                    <img src="coordinate_comparison.png" alt="Coordinate Comparison">
                </div>

                <div class="highlight-box">
                    <h3>📐 구면좌표계 설명</h3>
                    <ul>
                        <li><strong>r (반지름)</strong>: 원점으로부터의 거리 - 전체 활동량/건강 상태의 크기</li>
                        <li><strong>θ (방위각)</strong>: xy 평면에서의 각도 - 특성 간 균형/관계</li>
                        <li><strong>φ (극각)</strong>: z축으로부터의 각도 - 수직적 패턴 차이</li>
                    </ul>
                </div>
            </div>

            <!-- 주요 차이점 -->
            <div class="section">
                <h2 class="section-title">🔍 주요 차이점</h2>

                <table>
                    <thead>
                        <tr>
                            <th>좌표</th>
                            <th>이벤트 발생 평균</th>
                            <th>이벤트 미발생 평균</th>
                            <th>차이</th>
                            <th>차이율 (%)</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for diff in top_diffs:
        diff_class = 'positive' if diff['difference'] > 0 else 'negative'
        arrow = '⬆️' if diff['difference'] > 0 else '⬇️'

        html_content += f"""
                        <tr>
                            <td><strong>{diff['coordinate']}</strong></td>
                            <td>{diff['mean_occurred']:.4f}</td>
                            <td>{diff['mean_not_occurred']:.4f}</td>
                            <td class="{diff_class}">{diff['difference']:+.4f} {arrow}</td>
                            <td class="{diff_class}">{diff['difference_pct']:+.2f}%</td>
                        </tr>
"""

    html_content += """
                    </tbody>
                </table>
            </div>

            <!-- 핵심 인사이트 -->
            <div class="section">
                <h2 class="section-title">💡 핵심 인사이트</h2>

                <div class="highlight-box">
                    <h3>🎯 이벤트 발생 그룹의 특징</h3>
                    <ul>
                        <li>구면좌표 공간에서 특정 영역에 집중적으로 분포</li>
                        <li>반지름(r), 방위각(θ), 극각(φ)의 조합으로 고위험 패턴 형성</li>
                        <li>표준화를 통해 스케일 불균형 제거 → 진정한 관계 패턴 포착</li>
                    </ul>
                </div>

                <div class="highlight-box">
                    <h3>🔬 좌표계 변환의 효과</h3>
                    <ul>
                        <li><strong>원본 카테시안 좌표</strong>: 선형 분리 어려움 (C-index ~0.54)</li>
                        <li><strong>표준화 구면좌표</strong>: 비선형 패턴 명확히 분리 (C-index 0.72) ⬆️ +33%</li>
                        <li>3D 구조 활용으로 복잡한 생체신호 간 상호작용 표현</li>
                    </ul>
                </div>

                <div class="highlight-box">
                    <h3>📊 실무적 의미</h3>
                    <ul>
                        <li><strong>조기 경보</strong>: 특정 구면좌표 영역 진입 시 스트레스 위험 증가</li>
                        <li><strong>개인화 모니터링</strong>: 개인별 구면좌표 궤적 추적으로 변화 감지</li>
                        <li><strong>개입 전략</strong>: 좌표 성분별 맞춤 건강 관리 가능</li>
                    </ul>
                </div>
            </div>

            <!-- 기술적 세부사항 -->
            <div class="section">
                <h2 class="section-title">⚙️ 기술적 세부사항</h2>

                <div class="highlight-box">
                    <h3>데이터 및 방법론</h3>
                    <ul>
                        <li><strong>모델</strong>: 표준화 구면좌표 + Cox Proportional Hazards</li>
                        <li><strong>성능</strong>: C-index 0.7216 (Stress 예측)</li>
                        <li><strong>데이터</strong>: Stress 학습 데이터 10,276행</li>
                        <li><strong>샘플 추출</strong>: 각 그룹 중심에 가까운 50개 대표 샘플</li>
                        <li><strong>좌표계</strong>: 표준화(StandardScaler) → 구면좌표 변환</li>
                    </ul>
                </div>

                <div class="highlight-box">
                    <h3>구면좌표 변환 공식</h3>
                    <p><strong>카테시안 → 구면좌표</strong></p>
                    <ul>
                        <li>r = √(x² + y² + z²)</li>
                        <li>θ = arctan2(y, x)</li>
                        <li>φ = arccos(z / r)</li>
                    </ul>
                    <p style="margin-top: 10px;"><strong>역변환 (시각화용)</strong></p>
                    <ul>
                        <li>x = r × sin(φ) × cos(θ)</li>
                        <li>y = r × sin(φ) × sin(θ)</li>
                        <li>z = r × cos(φ)</li>
                    </ul>
                </div>
            </div>

            <!-- 관련 자료 -->
            <div class="section">
                <h2 class="section-title">📚 관련 자료</h2>

                <div class="highlight-box">
                    <ul>
                        <li><a href="../model_results/standardization_comprehensive_report.html" target="_blank">
                            표준화 종합 분석 리포트</a></li>
                        <li><a href="../model_results_standardized_sphere/stress_survival_analysis_standardized_spherical_report.html" target="_blank">
                            Stress 생존 분석 상세 리포트</a></li>
                        <li><a href="../STANDARDIZATION_EFFECT_INSIGHTS.md" target="_blank">
                            표준화 효과 인사이트 문서</a></li>
                    </ul>
                </div>
            </div>
        </div>

        <footer>
            <p><strong>KLOSDOM Lifelog Pattern Data Generation Project</strong></p>
            <p>분석 수행: Claude Code (Anthropic) | 날짜: 2026-06-27</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Contact: bosoagalaxy@gmail.com
            </p>
        </footer>
    </div>
</body>
</html>
"""

    # HTML 저장
    html_file = output_path / 'coordinate_event_comparison.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML 리포트 저장: {html_file}")

    return html_file

def main():
    """메인 실행 함수"""
    print("\n" + "="*70)
    print("🌍 좌표계 이벤트 발생/미발생 차이 시각화")
    print("="*70 + "\n")

    # 데이터 로드
    df = load_best_model_data()

    # 대표 샘플 추출
    repr_occurred, repr_not_occurred, coord_cols = extract_representative_samples(df)

    # HTML 리포트 생성
    html_file = create_html_report(repr_occurred, repr_not_occurred, coord_cols)

    print("\n" + "="*70)
    print("✅ 시각화 완료")
    print("="*70)
    print(f"\n📄 생성된 파일:")
    print(f"   - visualization_results/coordinate_event_comparison.html")
    print(f"   - visualization_results/coordinate_comparison.png")
    print("\n💡 다음 단계: model_results/index.html에 연결하기")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
