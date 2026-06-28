"""
정신건강 설문 데이터 로더 및 전처리 모듈
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class MentalHealthDataLoader:
    """정신건강 설문 데이터를 로드하고 전처리하는 클래스"""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.mh_dir = self.base_dir / "MentalHealth_Questionaire"
        self.lifelog_dir = self.base_dir / "KLOSDOM_Preprocessed_Dataset"

    def load_questionnaire_data(self, wave: int = 1) -> pd.DataFrame:
        """
        정신건강 설문 데이터 로드

        Args:
            wave: 회차 (1 또는 2)

        Returns:
            설문 데이터프레임
        """
        if wave == 1:
            file_path = self.mh_dir / "(PSS추가)전체대상자_천안설문_1회차_250905.xlsx"
            sheet_name = "전체대상자_천안설문_1회차_250905"
        elif wave == 2:
            file_path = self.mh_dir / "(PSS추가)전체대상자_천안설문_2회차_250905.xlsx"
            sheet_name = "전체대상자_천안설문_2회차_250905"
        else:
            raise ValueError("Wave must be 1 or 2")

        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Wave {wave} 데이터 로드 완료: {len(df)}명, {len(df.columns)}개 변수")
        return df

    def load_pss_data(self, wave: int = 1) -> pd.DataFrame:
        """PSS-10 데이터 로드"""
        if wave == 1:
            file_path = self.mh_dir / "(PSS추가)전체대상자_천안설문_1회차_250905.xlsx"
            sheet_name = "Sheet1"
        elif wave == 2:
            file_path = self.mh_dir / "(PSS추가)전체대상자_천안설문_2회차_250905.xlsx"
            sheet_name = "Sheet2"
        else:
            raise ValueError("Wave must be 1 or 2")

        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Wave {wave} PSS 데이터 로드 완료: {len(df)}명")
        return df

    def load_codebook(self) -> pd.DataFrame:
        """코딩북 로드"""
        file_path = self.mh_dir / "250822_코딩북.xlsx"
        df = pd.read_excel(file_path, sheet_name="Sheet1")
        print(f"코딩북 로드 완료: {len(df)}개 변수")
        return df

    def load_subject_list(self) -> pd.DataFrame:
        """대상자 리스트 로드"""
        file_path = self.mh_dir / "대상자 리스트(가입일)_1022.csv"
        df = pd.read_csv(file_path)
        print(f"대상자 리스트 로드 완료: {len(df)}명")
        return df

    def calculate_scale_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        주요 척도 점수 계산

        PHQ-9, GAD-7, ISI 등의 총점을 계산
        """
        df_processed = df.copy()

        # PHQ-9 (우울) 계산 - phq 관련 컬럼 찾기
        phq_cols = [col for col in df.columns if 'phq' in col.lower() and col.lower() != 'phq9_score']
        if phq_cols:
            df_processed['PHQ9_Score_Calculated'] = df[phq_cols].sum(axis=1, skipna=False)
            print(f"PHQ-9 점수 계산 완료 (컬럼: {len(phq_cols)}개)")

        # GAD-7 (불안) 계산
        gad_cols = [col for col in df.columns if 'gad' in col.lower() and col.lower() != 'gad7_score']
        if gad_cols:
            df_processed['GAD7_Score_Calculated'] = df[gad_cols].sum(axis=1, skipna=False)
            print(f"GAD-7 점수 계산 완료 (컬럼: {len(gad_cols)}개)")

        # ISI (불면증) 계산
        isi_cols = [col for col in df.columns if 'isi' in col.lower() and col.lower() != 'isi_score']
        if isi_cols:
            df_processed['ISI_Score_Calculated'] = df[isi_cols].sum(axis=1, skipna=False)
            print(f"ISI 점수 계산 완료 (컬럼: {len(isi_cols)}개)")

        return df_processed

    def get_lifelog_ids(self) -> List[str]:
        """라이프로그 데이터의 ID 목록 추출"""
        # 첫 번째 파일에서 ID 추출
        sample_file = list(self.lifelog_dir.glob("*.csv"))[0]
        df = pd.read_csv(sample_file, nrows=0)

        # ID가 컬럼인지 확인
        if 'ID' in df.columns or 'id' in df.columns:
            df_full = pd.read_csv(sample_file)
            id_col = 'ID' if 'ID' in df.columns else 'id'
            ids = df_full[id_col].unique().tolist()
            print(f"라이프로그 ID {len(ids)}개 추출 완료")
            return ids
        else:
            print("라이프로그 데이터에서 ID 컬럼을 찾을 수 없습니다")
            return []

    def match_questionnaire_lifelog(self,
                                    quest_df: pd.DataFrame,
                                    lifelog_ids: List[str]) -> pd.DataFrame:
        """
        설문 데이터와 라이프로그 데이터 ID 매칭

        Args:
            quest_df: 설문 데이터프레임
            lifelog_ids: 라이프로그 ID 리스트

        Returns:
            매칭된 데이터프레임
        """
        # ID 컬럼 찾기
        id_col = 'ID' if 'ID' in quest_df.columns else 'id'

        # 매칭
        matched = quest_df[quest_df[id_col].isin(lifelog_ids)].copy()
        match_rate = len(matched) / len(quest_df) * 100

        print(f"매칭 완료: {len(matched)}명 / {len(quest_df)}명 ({match_rate:.1f}%)")

        return matched

    def get_missing_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 요약 통계"""
        missing = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum().values,
            'missing_pct': (df.isnull().sum() / len(df) * 100).values
        })
        missing = missing[missing['missing_count'] > 0].sort_values('missing_pct', ascending=False)
        return missing

    def categorize_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        척도별 심각도 범주화

        PHQ-9, GAD-7, ISI 점수를 컷오프에 따라 범주화
        """
        df_cat = df.copy()

        # PHQ-9 우울 범주
        if 'PHQ9_Score' in df.columns:
            df_cat['PHQ9_Category'] = pd.cut(
                df['PHQ9_Score'],
                bins=[-1, 4, 9, 19, 27],
                labels=['정상', '경미', '중등도', '심각']
            )

        # GAD-7 불안 범주
        if 'GAD7_Score' in df.columns:
            df_cat['GAD7_Category'] = pd.cut(
                df['GAD7_Score'],
                bins=[-1, 4, 21],
                labels=['정상', '불안']
            )

        # ISI 불면증 범주
        if 'ISI_Score' in df.columns:
            df_cat['ISI_Category'] = pd.cut(
                df['ISI_Score'],
                bins=[-1, 7, 14, 21, 28],
                labels=['정상', '경미', '중등도', '심각']
            )

        return df_cat


def main():
    """데이터 로드 및 기본 탐색"""
    loader = MentalHealthDataLoader()

    print("="*80)
    print("정신건강 설문 데이터 로드")
    print("="*80)

    # 1회차 데이터 로드
    df_wave1 = loader.load_questionnaire_data(wave=1)
    pss_wave1 = loader.load_pss_data(wave=1)

    # 2회차 데이터 로드
    df_wave2 = loader.load_questionnaire_data(wave=2)

    # 코딩북 로드
    codebook = loader.load_codebook()

    # 대상자 리스트
    subject_list = loader.load_subject_list()

    print("\n" + "="*80)
    print("데이터 기본 정보")
    print("="*80)
    print(f"\n1회차 설문:")
    print(f"  - 참여자 수: {len(df_wave1)}")
    print(f"  - 변수 수: {len(df_wave1.columns)}")
    print(f"  - 주요 컬럼: {list(df_wave1.columns[:10])}")

    print(f"\n2회차 설문:")
    print(f"  - 참여자 수: {len(df_wave2)}")
    print(f"  - 추적률: {len(df_wave2)/len(df_wave1)*100:.1f}%")

    print(f"\nPSS-10 데이터:")
    print(f"  - 1회차: {len(pss_wave1)}명")
    print(f"  - 컬럼: {list(pss_wave1.columns)}")

    # 결측치 분석
    print("\n" + "="*80)
    print("결측치 분석 (상위 10개)")
    print("="*80)
    missing = loader.get_missing_summary(df_wave1)
    print(missing.head(10).to_string(index=False))

    # 라이프로그 매칭 확인
    print("\n" + "="*80)
    print("라이프로그 데이터 매칭 가능성")
    print("="*80)
    try:
        lifelog_ids = loader.get_lifelog_ids()
        if lifelog_ids:
            matched_wave1 = loader.match_questionnaire_lifelog(df_wave1, lifelog_ids)
            print(f"1회차 매칭 가능: {len(matched_wave1)}명")
    except Exception as e:
        print(f"라이프로그 매칭 확인 중 오류: {e}")

    return loader, df_wave1, df_wave2, pss_wave1, codebook


if __name__ == "__main__":
    main()
