"""
구면좌표 변환 모듈
데이터를 구면좌표 시스템(3D)으로 변환하여 새로운 특성 생성
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class SphericalTransformer:
    """구면좌표 변환 클래스"""

    def __init__(self):
        self.feature_triples = []
        self.spherical_features = []

    def transform_triple_to_spherical(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        세 특성을 구면좌표로 변환

        Args:
            x: 첫 번째 특성 (카테시안 x 좌표)
            y: 두 번째 특성 (카테시안 y 좌표)
            z: 세 번째 특성 (카테시안 z 좌표)

        Returns:
            r: 반지름 (magnitude)
            theta: 방위각 (azimuthal angle, 라디안)
            phi: 극각 (polar angle, 라디안)
        """
        # r = sqrt(x^2 + y^2 + z^2)
        r = np.sqrt(x**2 + y**2 + z**2)

        # theta = arctan2(y, x) - xy 평면에서의 각도
        theta = np.arctan2(y, x)

        # phi = arccos(z / r) - z축으로부터의 각도
        # r이 0인 경우를 처리
        phi = np.where(r > 1e-10, np.arccos(np.clip(z / r, -1.0, 1.0)), 0.0)

        return r, theta, phi

    def transform_dataframe(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        데이터프레임의 수치형 특성을 구면좌표로 변환

        Args:
            df: 원본 데이터프레임
            exclude_cols: 변환에서 제외할 컬럼 리스트

        Returns:
            구면좌표 변환된 데이터프레임
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

        print(f"📊 구면좌표 변환 대상: {len(numeric_cols)}개 특성")

        # 결과 데이터프레임 생성 (제외 컬럼은 유지)
        existing_exclude_cols = [col for col in exclude_cols if col in df.columns]
        result_df = df[existing_exclude_cols].copy() if existing_exclude_cols else pd.DataFrame(index=df.index)

        # 3개씩 묶어서 구면좌표 변환
        self.feature_triples = []
        self.spherical_features = []

        for i in range(0, len(numeric_cols), 3):
            if i + 2 < len(numeric_cols):
                # 세 특성을 트리플로 처리
                col1, col2, col3 = numeric_cols[i], numeric_cols[i+1], numeric_cols[i+2]

                x = df[col1].values
                y = df[col2].values
                z = df[col3].values

                # 구면좌표 변환
                r, theta, phi = self.transform_triple_to_spherical(x, y, z)

                # 새로운 특성 이름
                triple_name = f"{col1}_{col2}_{col3}"
                r_name = f"r_{triple_name}"
                theta_name = f"theta_{triple_name}"
                phi_name = f"phi_{triple_name}"

                result_df[r_name] = r
                result_df[theta_name] = theta
                result_df[phi_name] = phi

                self.feature_triples.append((col1, col2, col3))
                self.spherical_features.extend([r_name, theta_name, phi_name])

            elif i + 1 < len(numeric_cols):
                # 남은 특성이 2개인 경우 - 극좌표 변환으로 폴백
                col1, col2 = numeric_cols[i], numeric_cols[i+1]
                x = df[col1].values
                y = df[col2].values

                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)

                pair_name = f"{col1}_{col2}"
                r_name = f"r_{pair_name}"
                theta_name = f"theta_{pair_name}"

                result_df[r_name] = r
                result_df[theta_name] = theta

                self.spherical_features.extend([r_name, theta_name])
                print(f"  ⚠️  마지막 2개 특성은 극좌표로 변환: {col1}, {col2}")

            else:
                # 남은 특성이 1개인 경우 그대로 유지
                col = numeric_cols[i]
                result_df[col] = df[col].values
                self.spherical_features.append(col)
                print(f"  ⚠️  마지막 1개 특성은 그대로 유지: {col}")

        print(f"✅ 구면좌표 변환 완료: {len(self.feature_triples)}개 트리플 -> {len(self.spherical_features)}개 구면좌표 특성")

        return result_df

    def transform_splits(self, data_splits_dir: str = "data_splits",
                        output_dir: str = "data_splits_sphere",
                        target_vars: List[str] = None):
        """
        train/val/test 분할 데이터를 모두 구면좌표로 변환

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
        print(f"🌐 구면좌표 변환 시작")
        print(f"{'='*60}\n")

        for target in target_vars:
            print(f"\n📌 {target.upper()} 데이터 변환 중...")

            for split in ['train', 'val', 'test']:
                input_file = data_splits_path / f"{target}_{split}.csv"
                output_file = output_path / f"{target}_{split}_sphere.csv"

                if not input_file.exists():
                    print(f"  ⚠️  파일 없음: {input_file.name}")
                    continue

                # 데이터 로드
                df = pd.read_csv(input_file)
                print(f"  📂 {split}: {len(df):,}행 x {len(df.columns)}열")

                # 구면좌표 변환
                df_sphere = self.transform_dataframe(df)

                # 저장
                df_sphere.to_csv(output_file, index=False)
                print(f"  ✅ 저장 완료: {output_file.name} ({len(df_sphere):,}행 x {len(df_sphere.columns)}열)")

        print(f"\n{'='*60}")
        print(f"✅ 모든 데이터 구면좌표 변환 완료")
        print(f"{'='*60}\n")

        # 변환 요약 저장
        summary = {
            'target_variables': target_vars,
            'n_feature_triples': len(self.feature_triples),
            'n_spherical_features': len(self.spherical_features),
            'feature_triples': [f"{t[0]} + {t[1]} + {t[2]}" for t in self.feature_triples],
            'spherical_features': self.spherical_features
        }

        summary_df = pd.DataFrame({
            'Feature_Triple': [f"{t[0]} + {t[1]} + {t[2]}" for t in self.feature_triples],
            'R_Feature': [f"r_{t[0]}_{t[1]}_{t[2]}" for t in self.feature_triples],
            'Theta_Feature': [f"theta_{t[0]}_{t[1]}_{t[2]}" for t in self.feature_triples],
            'Phi_Feature': [f"phi_{t[0]}_{t[1]}_{t[2]}" for t in self.feature_triples]
        })

        summary_df.to_csv(output_path / "spherical_transformation_summary.csv", index=False)
        print(f"📊 변환 요약 저장: spherical_transformation_summary.csv\n")

        return summary


def main():
    """메인 실행 함수"""
    transformer = SphericalTransformer()
    summary = transformer.transform_splits()

    print("📋 변환 요약:")
    print(f"  - 특성 트리플 수: {summary['n_feature_triples']}")
    print(f"  - 구면좌표 특성 수: {summary['n_spherical_features']}")


if __name__ == "__main__":
    main()
