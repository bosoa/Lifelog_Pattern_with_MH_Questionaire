"""
피노타입 3D 회전 동영상 생성
가장 명확하게 구분되는 스트레스 피노타입 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from pathlib import Path

print("=" * 70)
print("3D 피노타입 회전 동영상 생성")
print("=" * 70)

# 데이터 로드
base_dir = Path("/Users/jihwanpark/claude/KLOSDOM/Lifelog_Pattern_Data_Generation")
data_file = base_dir / "data_splits_standardized_sphere" / "stress_test_standardized_spherical.csv"

print(f"\n데이터 로드 중: {data_file}")
df = pd.read_csv(data_file)

print(f"✓ 데이터 로드 완료: {len(df)} 샘플")

# 좌표 컬럼 찾기
r_cols = [c for c in df.columns if c.startswith('r_')]
theta_cols = [c for c in df.columns if c.startswith('theta_')]
phi_cols = [c for c in df.columns if c.startswith('phi_')]

print(f"\n좌표 컬럼:")
print(f"  - Radius (r): {len(r_cols)} 개")
print(f"  - Azimuthal (θ): {len(theta_cols)} 개")
print(f"  - Polar (φ): {len(phi_cols)} 개")

# 첫 번째 좌표 트리플 사용
r_col = r_cols[0]
theta_col = theta_cols[0]
phi_col = phi_cols[0]

print(f"\n사용 좌표:")
print(f"  - {r_col}")
print(f"  - {theta_col}")
print(f"  - {phi_col}")

# 이벤트 발생 여부로 피노타입 구분
df['phenotype'] = df['stress_binary'].apply(lambda x: 'Stress Event' if x == 1 else 'Normal')

# 통계
event_count = (df['stress_binary'] == 1).sum()
normal_count = (df['stress_binary'] == 0).sum()

print(f"\n피노타입 분포:")
print(f"  - Stress Event: {event_count} ({event_count/len(df)*100:.2f}%)")
print(f"  - Normal: {normal_count} ({normal_count/len(df)*100:.2f}%)")

# 샘플링 (시각화를 위해)
np.random.seed(42)

# 이벤트는 모두 포함, 정상은 일부만
stress_samples = df[df['stress_binary'] == 1]
normal_samples = df[df['stress_binary'] == 0].sample(min(500, normal_count), random_state=42)

df_plot = pd.concat([stress_samples, normal_samples], ignore_index=True)

print(f"\n시각화 샘플:")
print(f"  - Stress Event: {len(stress_samples)}")
print(f"  - Normal: {len(normal_samples)}")
print(f"  - 총: {len(df_plot)}")

# 3D 좌표 추출
r = df_plot[r_col].values
theta = df_plot[theta_col].values
phi = df_plot[phi_col].values

# Cartesian 좌표로 변환 (시각화용)
x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

# 색상 설정
colors = df_plot['phenotype'].map({'Stress Event': '#ff6b6b', 'Normal': '#95e1d3'}).values
sizes = df_plot['phenotype'].map({'Stress Event': 50, 'Normal': 20}).values

print("\n" + "=" * 70)
print("3D 회전 애니메이션 생성 중...")
print("=" * 70)

# Figure 생성
fig = plt.figure(figsize=(12, 9), facecolor='white')
ax = fig.add_subplot(111, projection='3d', facecolor='#f8f9fa')

# 초기 산점도
scatter = ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.6, edgecolors='white', linewidth=0.5)

# 제목 및 레이블
ax.set_xlabel('X (r·sin(φ)·cos(θ))', fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel('Y (r·sin(φ)·sin(θ))', fontsize=11, fontweight='bold', labelpad=10)
ax.set_zlabel('Z (r·cos(φ))', fontsize=11, fontweight='bold', labelpad=10)

title = ax.text2D(0.5, 0.95,
                 'Stress Phenotype Separation in Standardized Spherical Coordinates',
                 transform=ax.transAxes,
                 ha='center', va='top',
                 fontsize=14, fontweight='bold')

subtitle = ax.text2D(0.5, 0.91,
                     f'Stress Events: {len(stress_samples)} | Normal: {len(normal_samples)}',
                     transform=ax.transAxes,
                     ha='center', va='top',
                     fontsize=10, color='#666')

# 범례
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#ff6b6b', label=f'Stress Event (n={len(stress_samples)})', alpha=0.6),
    Patch(facecolor='#95e1d3', label=f'Normal (n={len(normal_samples)})', alpha=0.6)
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

# 그리드 및 배경 스타일
ax.grid(True, alpha=0.3)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# 초기 시야각
initial_elev = 20
initial_azim = 45
ax.view_init(elev=initial_elev, azim=initial_azim)

print("\n애니메이션 설정:")
print(f"  - 초기 시야각: elev={initial_elev}°, azim={initial_azim}°")
print(f"  - 회전 범위: 360° (5초)")
print(f"  - FPS: 30")
print(f"  - 총 프레임: 150")

# 애니메이션 업데이트 함수
def update(frame, ax, initial_elev, initial_azim):
    # 360도 회전 (150 프레임에 걸쳐)
    azim = initial_azim + (frame * 360 / 150)

    # 약간의 상하 움직임 추가 (더 역동적)
    elev = initial_elev + 10 * np.sin(frame * 2 * np.pi / 150)

    ax.view_init(elev=elev, azim=azim)
    return ax,

# 애니메이션 생성
print("\n애니메이션 렌더링 중...")
anim = animation.FuncAnimation(
    fig,
    update,
    frames=150,  # 5초 * 30 FPS
    fargs=(ax, initial_elev, initial_azim),
    interval=33,  # ~30 FPS
    blit=False
)

# 저장
output_file = Path("/Users/jihwanpark/claude/KLOSDOM/Lifelog_Pattern_Data_Generation/draft_for_paper/phenotype_3d_rotation.mp4")

print(f"\n동영상 저장 중: {output_file}")
print("(ffmpeg 사용, 몇 초 소요될 수 있습니다...)")

try:
    # MP4로 저장 (h264 코덱)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='KLOSDOM'), bitrate=1800)
    anim.save(str(output_file), writer=writer, dpi=100)

    print(f"\n✓ 동영상 저장 완료!")
    print(f"  파일: {output_file}")
    print(f"  크기: {output_file.stat().st_size / 1024:.1f} KB")

except Exception as e:
    print(f"\n⚠ MP4 저장 실패: {e}")
    print("\nGIF로 대체 저장 시도 중...")

    # GIF로 저장 (대체)
    output_file_gif = Path("/Users/jihwanpark/claude/KLOSDOM/Lifelog_Pattern_Data_Generation/draft_for_paper/phenotype_3d_rotation.gif")
    anim.save(str(output_file_gif), writer='pillow', fps=20, dpi=80)

    print(f"\n✓ GIF 저장 완료!")
    print(f"  파일: {output_file_gif}")
    print(f"  크기: {output_file_gif.stat().st_size / 1024:.1f} KB")

plt.close()

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
