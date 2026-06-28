"""
정신건강 설문 데이터 상세 탐색 및 전처리
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from mh_data_loader import MentalHealthDataLoader

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def explore_mental_health_scales(df: pd.DataFrame, wave: int = 1) -> dict:
    """
    정신건강 척도 탐색

    Args:
        df: 설문 데이터
        wave: 회차

    Returns:
        척도별 통계 정보
    """
    print(f"\n{'='*80}")
    print(f"Wave {wave} 정신건강 척도 분석")
    print(f"{'='*80}")

    results = {}

    # PHQ-9 (우울)
    phq_cols = [col for col in df.columns if col.startswith('phq') and col != 'PHQ9_Score']
    if phq_cols:
        print(f"\nPHQ-9 (우울 척도) - {len(phq_cols)}개 문항")
        print(f"  컬럼: {phq_cols}")

        # 각 문항별 응답 분포
        for col in phq_cols:
            valid_count = df[col].notna().sum()
            print(f"  {col}: 응답 {valid_count}명 ({valid_count/len(df)*100:.1f}%)")

        # 총점 계산
        if 'PHQ9_Score' in df.columns:
            scores = df['PHQ9_Score'].dropna()
            print(f"\n  PHQ9 총점 통계:")
            print(f"    - 평균: {scores.mean():.2f} (SD={scores.std():.2f})")
            print(f"    - 중앙값: {scores.median():.2f}")
            print(f"    - 범위: {scores.min():.0f} ~ {scores.max():.0f}")

            # 심각도 분포
            severity = pd.cut(scores, bins=[-1, 4, 9, 19, 27],
                            labels=['정상(0-4)', '경미(5-9)', '중등도(10-19)', '심각(20-27)'])
            print(f"\n  심각도 분포:")
            for cat, count in severity.value_counts().sort_index().items():
                print(f"    {cat}: {count}명 ({count/len(scores)*100:.1f}%)")

            results['PHQ9'] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'severity_dist': severity.value_counts().to_dict()
            }

    # GAD-7 (불안)
    gad_cols = [col for col in df.columns if col.startswith('gad') and col != 'GAD7_Score']
    if gad_cols:
        print(f"\nGAD-7 (불안 척도) - {len(gad_cols)}개 문항")
        print(f"  컬럼: {gad_cols}")

        if 'GAD7_Score' in df.columns:
            scores = df['GAD7_Score'].dropna()
            print(f"\n  GAD7 총점 통계:")
            print(f"    - 평균: {scores.mean():.2f} (SD={scores.std():.2f})")
            print(f"    - 중앙값: {scores.median():.2f}")
            print(f"    - 범위: {scores.min():.0f} ~ {scores.max():.0f}")

            # 심각도 분포
            severity = pd.cut(scores, bins=[-1, 4, 21],
                            labels=['정상(0-4)', '불안(5+)'])
            print(f"\n  심각도 분포:")
            for cat, count in severity.value_counts().sort_index().items():
                print(f"    {cat}: {count}명 ({count/len(scores)*100:.1f}%)")

            results['GAD7'] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'severity_dist': severity.value_counts().to_dict()
            }

    # ISI (불면증)
    isi_cols = [col for col in df.columns if col.startswith('isi') and col != 'ISI_Score']
    if isi_cols:
        print(f"\nISI (불면증 척도) - {len(isi_cols)}개 문항")
        print(f"  컬럼: {isi_cols}")

        if 'ISI_Score' in df.columns:
            scores = df['ISI_Score'].dropna()
            print(f"\n  ISI 총점 통계:")
            print(f"    - 평균: {scores.mean():.2f} (SD={scores.std():.2f})")
            print(f"    - 중앙값: {scores.median():.2f}")
            print(f"    - 범위: {scores.min():.0f} ~ {scores.max():.0f}")

            # 심각도 분포
            severity = pd.cut(scores, bins=[-1, 7, 14, 21, 28],
                            labels=['정상(0-7)', '경미(8-14)', '중등도(15-21)', '심각(22-28)'])
            print(f"\n  심각도 분포:")
            for cat, count in severity.value_counts().sort_index().items():
                print(f"    {cat}: {count}명 ({count/len(scores)*100:.1f}%)")

            results['ISI'] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'severity_dist': severity.value_counts().to_dict()
            }

    # Loneliness (외로움)
    if 'Loneliness_Score' in df.columns:
        scores = df['Loneliness_Score'].dropna()
        print(f"\n외로움 척도:")
        print(f"  - 평균: {scores.mean():.2f} (SD={scores.std():.2f})")
        print(f"  - 중앙값: {scores.median():.2f}")

        # 심각도 분포
        severity = (scores >= 2).map({True: '외로움 있음(2+)', False: '외로움 없음(0-1)'})
        print(f"\n  분포:")
        for cat, count in severity.value_counts().items():
            print(f"    {cat}: {count}명 ({count/len(scores)*100:.1f}%)")

        results['Loneliness'] = {
            'mean': scores.mean(),
            'std': scores.std()
        }

    # WHOQOL (삶의 질)
    if 'WHOQOL_Score' in df.columns:
        scores = df['WHOQOL_Score'].dropna()
        print(f"\nWHOQOL (삶의 질):")
        print(f"  - 평균: {scores.mean():.2f} (SD={scores.std():.2f})")
        print(f"  - 중앙값: {scores.median():.2f}")

        results['WHOQOL'] = {
            'mean': scores.mean(),
            'std': scores.std()
        }

    return results


def analyze_pss_data(df: pd.DataFrame) -> dict:
    """PSS-10 데이터 분석"""
    print(f"\n{'='*80}")
    print("PSS-10 (스트레스 척도) 분석")
    print(f"{'='*80}")

    pss_cols = [col for col in df.columns if col.startswith('pss') and col != 'id']

    if pss_cols:
        print(f"\nPSS 문항: {len(pss_cols)}개")
        print(f"  {pss_cols}")

        # 총점 계산 (역코딩 고려해야 함)
        # PSS-10: 4, 5, 7, 8번은 역코딩
        df_pss = df.copy()

        # 각 문항이 숫자인지 확인
        for col in pss_cols:
            if df[col].dtype == 'object':
                # 텍스트를 숫자로 변환 (응답 범주 매핑 필요)
                print(f"\n{col} 응답 유형: {df[col].value_counts().head()}")

        # 숫자 컬럼만 합산
        numeric_cols = [col for col in pss_cols if df[col].dtype in ['int64', 'float64']]
        if numeric_cols:
            total = df[numeric_cols].sum(axis=1)
            print(f"\nPSS 총점 통계 ({len(numeric_cols)}개 문항 기준):")
            print(f"  - 평균: {total.mean():.2f} (SD={total.std():.2f})")
            print(f"  - 중앙값: {total.median():.2f}")
            print(f"  - 범위: {total.min():.0f} ~ {total.max():.0f}")

            return {
                'mean': total.mean(),
                'std': total.std(),
                'total_scores': total
            }

    return {}


def match_wave1_wave2(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
    """
    1회차와 2회차 매칭

    Returns:
        (matched_df1, matched_df2, matched_ids)
    """
    print(f"\n{'='*80}")
    print("1회차-2회차 매칭 분석")
    print(f"{'='*80}")

    # ID 컬럼 확인
    id_col = 'ID' if 'ID' in df1.columns else 'id'

    # 공통 ID 찾기
    ids_wave1 = set(df1[id_col].dropna())
    ids_wave2 = set(df2[id_col].dropna())

    matched_ids = ids_wave1 & ids_wave2
    only_wave1 = ids_wave1 - ids_wave2
    only_wave2 = ids_wave2 - ids_wave1

    print(f"\n매칭 결과:")
    print(f"  - 1회차 전용: {len(only_wave1)}명 ({len(only_wave1)/len(ids_wave1)*100:.1f}%)")
    print(f"  - 2회차 전용: {len(only_wave2)}명 ({len(only_wave2)/len(ids_wave2)*100:.1f}%)")
    print(f"  - 양쪽 모두: {len(matched_ids)}명 ({len(matched_ids)/len(ids_wave1)*100:.1f}%)")

    # 매칭된 데이터만 추출
    matched_df1 = df1[df1[id_col].isin(matched_ids)].copy()
    matched_df2 = df2[df2[id_col].isin(matched_ids)].copy()

    # ID로 정렬
    matched_df1 = matched_df1.sort_values(id_col).reset_index(drop=True)
    matched_df2 = matched_df2.sort_values(id_col).reset_index(drop=True)

    return matched_df1, matched_df2, list(matched_ids)


def save_preprocessed_data(df: pd.DataFrame, filename: str, output_dir: str = "analysis_results"):
    """전처리된 데이터 저장"""
    output_path = Path(output_dir) / "01_data_exploration" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\n데이터 저장 완료: {output_path}")
    print(f"  - 행: {len(df)}, 열: {len(df.columns)}")


def main():
    """메인 실행 함수"""
    print("="*80)
    print("정신건강 설문 데이터 상세 탐색")
    print("="*80)

    # 데이터 로더 초기화
    loader = MentalHealthDataLoader()

    # 데이터 로드
    df_wave1 = loader.load_questionnaire_data(wave=1)
    df_wave2 = loader.load_questionnaire_data(wave=2)
    pss_wave1 = loader.load_pss_data(wave=1)

    # 1회차 척도 분석
    results_wave1 = explore_mental_health_scales(df_wave1, wave=1)

    # 2회차 척도 분석
    results_wave2 = explore_mental_health_scales(df_wave2, wave=2)

    # PSS 분석
    pss_results = analyze_pss_data(pss_wave1)

    # 1회차-2회차 매칭
    matched_df1, matched_df2, matched_ids = match_wave1_wave2(df_wave1, df_wave2)

    # 전처리된 데이터 저장
    print(f"\n{'='*80}")
    print("전처리 데이터 저장")
    print(f"{'='*80}")

    save_preprocessed_data(df_wave1, "wave1_full.csv")
    save_preprocessed_data(df_wave2, "wave2_full.csv")
    save_preprocessed_data(matched_df1, "wave1_matched.csv")
    save_preprocessed_data(matched_df2, "wave2_matched.csv")
    save_preprocessed_data(pss_wave1, "pss_wave1.csv")

    # 매칭 ID 리스트 저장
    matched_ids_df = pd.DataFrame({'ID': matched_ids})
    save_preprocessed_data(matched_ids_df, "matched_ids.csv")

    # 라이프로그 매칭
    print(f"\n{'='*80}")
    print("라이프로그 데이터 매칭")
    print(f"{'='*80}")

    try:
        lifelog_ids = loader.get_lifelog_ids()
        matched_lifelog = loader.match_questionnaire_lifelog(df_wave1, lifelog_ids)
        save_preprocessed_data(matched_lifelog, "wave1_lifelog_matched.csv")

        # 매칭된 ID 저장
        lifelog_matched_ids = pd.DataFrame({'ID': matched_lifelog['ID'].unique()})
        save_preprocessed_data(lifelog_matched_ids, "lifelog_matched_ids.csv")

    except Exception as e:
        print(f"라이프로그 매칭 오류: {e}")

    print(f"\n{'='*80}")
    print("데이터 탐색 완료!")
    print(f"{'='*80}")

    return {
        'wave1': df_wave1,
        'wave2': df_wave2,
        'pss': pss_wave1,
        'matched_wave1': matched_df1,
        'matched_wave2': matched_df2,
        'results_wave1': results_wave1,
        'results_wave2': results_wave2,
        'pss_results': pss_results
    }


if __name__ == "__main__":
    results = main()
