"""
데이터 로드 및 전처리 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import os


class KLOSDOMDataLoader:
    """KLOSDOM 데이터셋 로더 - 동적 파일 탐지"""

    def __init__(self, data_dir: str = "KLOSDOM_Preprocessed_Dataset"):
        self.data_dir = Path(data_dir)

        # 디렉토리가 존재하는지 확인
        if not self.data_dir.exists():
            raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {self.data_dir}")

        # 파일 목록 동적으로 스캔
        self.sensor_files = self._scan_sensor_files()
        self.survey_files = self._scan_survey_files()

        # 파일 개수 검증
        self._validate_files()

    def _scan_sensor_files(self) -> Dict[str, str]:
        """센서 데이터 파일 (whole_a) 동적 스캔"""
        sensor_files = {}

        # 센서 파일 패턴 정의 (whole_a01 ~ whole_a18)
        sensor_mapping = {
            'a01': 'hrv',
            'a02': 'walk',
            'a03': 'stick_sensor',
            'a04': 'deep_sleep',
            'a05': 'rem_sleep',
            'a06': 'oxygen_saturation',
            'a07': 'screen_time',
            'a08': 'heart_beat',
            'a09': 'body_temperature',
            'a10': 'light_sleep',
            'a11': 'moving_distance',
            'a12': 'wakeup_time',
            'a13': 'bed_time',
            'a14': 'lux_sensor',
            'a15': 'total_sleep',
            'a16': 'skin_temperature',
            'a17': 'blood_sugar',
            'a18': 'blood_pressure',
        }

        # 디렉토리에서 파일 검색
        for filename in os.listdir(self.data_dir):
            if filename.startswith('whole_a') and filename.endswith('.csv'):
                # 파일명에서 코드 추출 (예: whole_a01_hrv_20260622.csv -> a01)
                parts = filename.split('_')
                if len(parts) >= 2:
                    code = parts[1]  # a01, a02, ...
                    if code in sensor_mapping:
                        sensor_files[sensor_mapping[code]] = filename

        return sensor_files

    def _scan_survey_files(self) -> Dict[str, str]:
        """설문 데이터 파일 (whole_e) 동적 스캔"""
        survey_files = {}

        # 설문 파일 패턴 정의 (whole_e01 ~ whole_e04)
        survey_mapping = {
            'e01': 'anxiety',
            'e02': 'depression',
            'e03': 'sleep_quality',
            'e04': 'stress',
        }

        # 디렉토리에서 파일 검색
        for filename in os.listdir(self.data_dir):
            if filename.startswith('whole_e') and filename.endswith('.csv'):
                # 파일명에서 코드 추출 (예: whole_e01_anxiety_20260622.csv -> e01)
                parts = filename.split('_')
                if len(parts) >= 2:
                    code = parts[1]  # e01, e02, ...
                    if code in survey_mapping:
                        survey_files[survey_mapping[code]] = filename

        return survey_files

    def _validate_files(self):
        """파일 개수 및 존재 여부 검증"""
        # 센서 파일 18개 확인
        if len(self.sensor_files) != 18:
            print(f"⚠️  경고: 센서 데이터 파일이 {len(self.sensor_files)}개 발견되었습니다. (예상: 18개)")
            print(f"   찾은 파일: {sorted(self.sensor_files.keys())}")

        # 설문 파일 4개 확인
        if len(self.survey_files) != 4:
            print(f"⚠️  경고: 설문 데이터 파일이 {len(self.survey_files)}개 발견되었습니다. (예상: 4개)")
            print(f"   찾은 파일: {sorted(self.survey_files.keys())}")

        # 파일 존재 여부 확인
        for name, filename in {**self.sensor_files, **self.survey_files}.items():
            filepath = self.data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

        # 검증 성공 메시지
        if len(self.sensor_files) == 18 and len(self.survey_files) == 4:
            print(f"✅ 데이터 파일 검증 완료: 센서 {len(self.sensor_files)}개, 설문 {len(self.survey_files)}개")

    def load_sensor_data(self) -> Dict[str, pd.DataFrame]:
        """모든 센서 데이터 로드"""
        sensor_data = {}
        for name, filename in self.sensor_files.items():
            filepath = self.data_dir / filename
            df = pd.read_csv(filepath)
            # 0을 NaN으로 변환
            df = df.replace(0, np.nan)
            sensor_data[name] = df
        return sensor_data

    def load_survey_data(self) -> Dict[str, pd.DataFrame]:
        """모든 설문 데이터 로드"""
        survey_data = {}
        for name, filename in self.survey_files.items():
            filepath = self.data_dir / filename
            df = pd.read_csv(filepath)
            # 0을 NaN으로 변환
            df = df.replace(0, np.nan)
            survey_data[name] = df
        return survey_data

    def wide_to_long(self, df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        """Wide format을 Long format으로 변환"""
        df_long = df.melt(
            id_vars=['ID'],
            var_name='date',
            value_name=value_name
        )
        df_long['date'] = pd.to_datetime(df_long['date'])
        return df_long

    def prepare_pca_data(
        self,
        target_variable: str = 'anxiety',
        min_data_points: int = 10
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        PCA 분석을 위한 데이터 준비

        Args:
            target_variable: 종속변수 ('anxiety', 'depression', 'stress')
            min_data_points: 최소 데이터 포인트 수

        Returns:
            X: 센서 데이터 (독립변수)
            y: 타겟 변수 (종속변수)
            feature_names: 특징 이름 리스트
        """
        # 센서 데이터와 설문 데이터 로드
        sensor_data = self.load_sensor_data()
        survey_data = self.load_survey_data()

        # 타겟 변수 선택
        if target_variable not in survey_data:
            raise ValueError(f"타겟 변수 '{target_variable}'가 존재하지 않습니다.")

        target_df = survey_data[target_variable]

        # Long format으로 변환
        target_long = self.wide_to_long(target_df, f'{target_variable}_score')

        sensor_long_dict = {}
        for name, df in sensor_data.items():
            sensor_long_dict[name] = self.wide_to_long(df, name)

        # 모든 센서 데이터 병합
        merged_data = target_long.copy()
        for name, df in sensor_long_dict.items():
            merged_data = merged_data.merge(
                df,
                on=['ID', 'date'],
                how='left'
            )

        # 결측치 및 무한대 값 제거 (타겟 변수가 있는 행만 유지)
        target_col = f'{target_variable}_score'
        merged_data = merged_data.dropna(subset=[target_col])
        merged_data = merged_data[np.isfinite(merged_data[target_col])]

        # 사용자별로 충분한 데이터가 있는지 확인
        user_counts = merged_data.groupby('ID').size()
        valid_users = user_counts[user_counts >= min_data_points].index
        merged_data = merged_data[merged_data['ID'].isin(valid_users)]

        # 특징과 타겟 분리
        feature_cols = list(sensor_data.keys())
        X = merged_data[feature_cols]
        y = merged_data[f'{target_variable}_score']

        # 다단계 결측치 처리
        # 1단계: 각 컬럼의 평균값으로 채우기
        X = X.fillna(X.mean())

        # 2단계: 여전히 NaN이 있는 경우 (컬럼 전체가 NaN인 경우) 0으로 채우기
        X = X.fillna(0)

        # 3단계: 혹시 모를 inf 값도 0으로 변환
        X = X.replace([np.inf, -np.inf], 0)

        # 4단계: 최종 확인 - NaN이 있는 행 제거 (안전장치)
        if X.isna().any().any():
            print(f"⚠️ 경고: {X.isna().sum().sum()}개의 NaN 값을 발견했습니다. 해당 행을 제거합니다.")
            valid_idx = ~X.isna().any(axis=1)
            X = X[valid_idx]
            y = y[valid_idx]

        return X, y, feature_cols

    def get_dataset_info(self) -> Dict:
        """데이터셋 기본 정보 반환"""
        sensor_data = self.load_sensor_data()
        survey_data = self.load_survey_data()

        # 첫 번째 파일에서 사용자 수와 날짜 수 확인
        first_sensor = list(sensor_data.values())[0]
        n_users = len(first_sensor) - 1  # 헤더 제외
        n_dates = len(first_sensor.columns) - 1  # ID 컬럼 제외

        return {
            'n_users': n_users,
            'n_dates': n_dates,
            'n_sensors': len(sensor_data),
            'n_surveys': len(survey_data),
            'sensor_names': list(sensor_data.keys()),
            'survey_names': list(survey_data.keys()),
        }
