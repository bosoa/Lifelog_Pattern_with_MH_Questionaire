"""
Table 1 생성: 데이터셋 기본 특성
참고 이미지와 유사한 전문적인 형식
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

print("=" * 70)
print("Table 1: 데이터셋 기본 특성 생성")
print("=" * 70)

# 데이터셋 정보
data = {
    '특성 (Characteristics)': [
        '전체 샘플 수 (명)',
        '',
        '타겟별 분포',
        '  - Anxiety',
        '    • Train',
        '    • Validation',
        '    • Test',
        '    • 이벤트 비율 (%)',
        '',
        '  - Depression',
        '    • Train',
        '    • Validation',
        '    • Test',
        '    • 이벤트 비율 (%)',
        '',
        '  - Stress',
        '    • Train',
        '    • Validation',
        '    • Test',
        '    • 이벤트 비율 (%)',
        '',
        '특징 변수',
        '  - 수면 지표 (개)',
        '  - 심혈관 지표 (개)',
        '  - 활동 지표 (개)',
        '  - 대사 지표 (개)',
        '  - 기타 (개)',
        '  - 총 특징 변수 (개)',
        '',
        '데이터 전처리',
        '  - 결측치 처리',
        '  - 이상치 처리',
        '  - 표준화 방법',
    ],
    '값': [
        '281,138',
        '',
        '',
        '',
        '176,160 (62.7%)',
        '50,332 (17.9%)',
        '25,166 (8.9%)',
        '6.6',
        '',
        '',
        '10,358 (3.7%)',
        '2,960 (1.1%)',
        '1,480 (0.5%)',
        '47.4',
        '',
        '',
        '10,276 (3.7%)',
        '2,937 (1.0%)',
        '1,469 (0.5%)',
        '24.6',
        '',
        '',
        '3 (total_sleep, rem_sleep, light_sleep)',
        '2 (heart_beat, hrv)',
        '2 (walk, stick_sensor)',
        '2 (blood_sugar, body_temperature)',
        '1 (skin_temperature, oxygen_saturation)',
        '10',
        '',
        '',
        '중앙값 대체 (Median imputation)',
        'IQR 기반 클리핑 (Q1-1.5×IQR, Q3+1.5×IQR)',
        'StandardScaler (μ=0, σ=1)',
    ]
}

df = pd.DataFrame(data)

# 그림 생성
fig, ax = plt.subplots(figsize=(14, 11))
ax.axis('tight')
ax.axis('off')

# 테이블 생성
table = ax.table(
    cellText=df.values,
    colLabels=['특성 (Characteristics)', '값'],
    cellLoc='left',
    loc='center',
    colWidths=[0.6, 0.4]
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# 헤더 스타일
for i in range(2):
    cell = table[(0, i)]
    cell.set_facecolor('#667eea')
    cell.set_text_props(weight='bold', color='white', ha='center')
    cell.set_height(0.08)

# 행 스타일링
section_rows = [0, 2, 9, 16, 21, 29]  # 섹션 시작 행
subsection_rows = [3, 10, 17, 22]  # 하위섹션 행

for i in range(len(df)):
    for j in range(2):
        cell = table[(i+1, j)]

        # 섹션 헤더
        if i in section_rows:
            cell.set_facecolor('#e8eaf6')
            if j == 0:
                cell.set_text_props(weight='bold', fontsize=11)

        # 하위섹션
        elif i in subsection_rows:
            cell.set_facecolor('#f3f4f9')
            if j == 0:
                cell.set_text_props(weight='bold', fontsize=10)

        # 일반 행
        else:
            if i % 2 == 0:
                cell.set_facecolor('#ffffff')
            else:
                cell.set_facecolor('#fafafa')

        # 빈 행
        if df.iloc[i, 0] == '':
            cell.set_facecolor('#ffffff')
            cell.set_height(0.02)

        # 들여쓰기된 항목
        if df.iloc[i, 0].startswith('  -') or df.iloc[i, 0].startswith('    •'):
            if j == 0:
                cell.set_text_props(fontsize=9)

# 제목
title = fig.text(0.5, 0.97, 'Table 1: 데이터셋의 기본 특성 (N=281,138)',
                ha='center', va='top',
                fontsize=14, fontweight='bold')

subtitle = fig.text(0.5, 0.94, 'Basic Characteristics of the Dataset (N=281,138)',
                   ha='center', va='top',
                   fontsize=11, color='#666')

# 각주
footnote = fig.text(0.1, 0.02,
                   '주: KLOSDOM (Korea Lifelog Observatory for Sustainable Dementia Outcome Management) 전처리 데이터셋 (version 20260622)\n' +
                   '이벤트: 셀프체크 점수 ≥ 4점으로 정의',
                   ha='left', va='bottom',
                   fontsize=8, color='#666', style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 0.93])

# 저장
output_dir = Path("/Users/jihwanpark/claude/KLOSDOM/Lifelog_Pattern_Data_Generation/draft_for_paper/figures/phenotyping")
output_file = output_dir / 'table1_dataset_characteristics.png'

plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Table 1 이미지 저장: {output_file}")
print(f"  크기: {output_file.stat().st_size / 1024:.1f} KB")

plt.close()

# CSV로도 저장
csv_file = output_dir / 'table1_dataset_characteristics.csv'
df.to_csv(csv_file, index=False, encoding='utf-8-sig')
print(f"✓ Table 1 CSV 저장: {csv_file}")

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
