"""
극좌표 변환 모듈
데이터를 극좌표 시스템으로 변환하여 새로운 특성 생성
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class PolarTransformer:
    """극좌표 변환 클래스"""

    def __init__(self):
        self.feature_pairs = []
        self.polar_features = []

    def transform_pair_to_polar(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        두 특성을 극좌표로 변환

        Args:
            x: 첫 번째 특성 (카테시안 x 좌표)
            y: 두 번째 특성 (카테시안 y 좌표)

        Returns:
            r: 반지름 (magnitude)
            theta: 각도 (라디안)
        """
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        return r, theta

    def transform_dataframe(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        데이터프레임의 수치형 특성을 극좌표로 변환

        Args:
            df: 원본 데이터프레임
            exclude_cols: 변환에서 제외할 컬럼 리스트

        Returns:
            극좌표 변환된 데이터프레임
        """
        if exclude_cols is None:
            exclude_cols = []

        # 제외 컬럼에 추가할 기본 컬럼들 (실제로 데이터에 존재하는 것만)
        potential_exclude = ['level', 'level_name', 'anxiety_score', 'depression_score',
                            'stress_score', 'anxiety_binary', 'depression_binary',
                            'stress_binary', 'anxiety_label', 'depression_label', 'stress_label']

        # 실제로 데이터프레임에 존재하는 컬럼만 제외 목록에 추가
        default_exclude = [col for col in potential_exclude if col in df.columns]
        exclude_cols.extend(default_exclude)
        exclude_cols = list(set(exclude_cols))

        # 수치형 컬럼 추출
        numeric_cols = [col for col in df.columns
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        print(f"📊 극좌표 변환 대상: {len(numeric_cols)}개 특성")

        # 결과 데이터프레임 생성 (제외 컬럼은 유지)
        # 실제로 데이터에 존재하는 제외 컬럼만 복사
        existing_exclude_cols = [col for col in exclude_cols if col in df.columns]
        result_df = df[existing_exclude_cols].copy() if existing_exclude_cols else pd.DataFrame(index=df.index)

        # 페어별로 극좌표 변환
        self.feature_pairs = []
        self.polar_features = []

        for i in range(0, len(numeric_cols), 2):
            if i + 1 < len(numeric_cols):
                # 두 특성을 페어로 처리
                col1, col2 = numeric_cols[i], numeric_cols[i+1]

                x = df[col1].values
                y = df[col2].values

                # 극좌표 변환
                r, theta = self.transform_pair_to_polar(x, y)

                # 새로운 특성 이름
                pair_name = f"{col1}_{col2}"
                r_name = f"r_{pair_name}"
                theta_name = f"theta_{pair_name}"

                result_df[r_name] = r
                result_df[theta_name] = theta

                self.feature_pairs.append((col1, col2))
                self.polar_features.extend([r_name, theta_name])

            else:
                # 홀수 개의 특성인 경우 마지막 특성은 그대로 유지
                col = numeric_cols[i]
                result_df[col] = df[col].values
                self.polar_features.append(col)

        print(f"✅ 극좌표 변환 완료: {len(self.feature_pairs)}개 페어 -> {len(self.polar_features)}개 극좌표 특성")

        return result_df

    def transform_splits(self, data_splits_dir: str = "data_splits",
                        output_dir: str = "data_splits_polar",
                        target_vars: List[str] = None):
        """
        train/val/test 분할 데이터를 모두 극좌표로 변환

        Args:
            data_splits_dir: 원본 데이터 분할 디렉토리
            output_dir: 출력 디렉토리
            target_vars: 타겟 변수 리스트
        """
        if target_vars is None:
            target_vars = ['anxiety', 'depression', 'stress']

        data_splits_path = Path(data_splits_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"🔄 극좌표 변환 시작")
        print(f"{'='*60}\n")

        for target in target_vars:
            print(f"\n📌 {target.upper()} 데이터 변환 중...")

            for split in ['train', 'val', 'test']:
                input_file = data_splits_path / f"{target}_{split}.csv"
                output_file = output_path / f"{target}_{split}_polar.csv"

                if not input_file.exists():
                    print(f"  ⚠️  파일 없음: {input_file.name}")
                    continue

                # 데이터 로드
                df = pd.read_csv(input_file)
                print(f"  📂 {split}: {len(df):,}행 x {len(df.columns)}열")

                # 극좌표 변환
                df_polar = self.transform_dataframe(df)

                # 저장
                df_polar.to_csv(output_file, index=False)
                print(f"  ✅ 저장 완료: {output_file.name} ({len(df_polar):,}행 x {len(df_polar.columns)}열)")

        print(f"\n{'='*60}")
        print(f"✅ 모든 데이터 극좌표 변환 완료")
        print(f"{'='*60}\n")

        # 변환 요약 저장
        summary = {
            'target_variables': target_vars,
            'n_feature_pairs': len(self.feature_pairs),
            'n_polar_features': len(self.polar_features),
            'feature_pairs': [f"{p[0]} + {p[1]}" for p in self.feature_pairs],
            'polar_features': self.polar_features
        }

        summary_df = pd.DataFrame({
            'Feature_Pair': [f"{p[0]} + {p[1]}" for p in self.feature_pairs],
            'R_Feature': [f"r_{p[0]}_{p[1]}" for p in self.feature_pairs],
            'Theta_Feature': [f"theta_{p[0]}_{p[1]}" for p in self.feature_pairs]
        })

        summary_df.to_csv(output_path / "polar_transformation_summary.csv", index=False)
        print(f"📊 변환 요약 저장: polar_transformation_summary.csv\n")

        return summary


def main():
    """메인 실행 함수"""
    transformer = PolarTransformer()
    summary = transformer.transform_splits()

    print("📋 변환 요약:")
    print(f"  - 특성 페어 수: {summary['n_feature_pairs']}")
    print(f"  - 극좌표 특성 수: {summary['n_polar_features']}")


if __name__ == "__main__":
    main()
