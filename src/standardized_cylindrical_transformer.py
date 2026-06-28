"""
표준화 기반 원통좌표 변환 모듈
데이터를 표준화한 후 원통좌표 시스템(3D)으로 변환하여 성능 개선
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


class StandardizedCylindricalTransformer:
    """표준화 기반 원통좌표 변환 클래스"""

    def __init__(self):
        self.feature_triples = []
        self.cylindrical_features = []
        self.scalers = {}
        self.is_fitted = False

    def transform_triple_to_cylindrical(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        세 특성을 원통좌표로 변환

        Args:
            x: 첫 번째 특성 (표준화된 카테시안 x 좌표)
            y: 두 번째 특성 (표준화된 카테시안 y 좌표)
            z: 세 번째 특성 (표준화된 카테시안 z 좌표)

        Returns:
            rho: xy 평면에서의 반지름
            phi: 방위각 (라디안)
            z: 높이 (그대로 유지)
        """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        return rho, phi, z

    def fit(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> 'StandardizedCylindricalTransformer':
        """데이터에 scaler를 fit"""
        if exclude_cols is None:
            exclude_cols = []

        potential_exclude = ['level', 'level_name', 'anxiety_score', 'depression_score',
                            'stress_score', 'anxiety_binary', 'depression_binary',
                            'stress_binary', 'anxiety_label', 'depression_label', 'stress_label']

        default_exclude = [col for col in potential_exclude if col in df.columns]
        exclude_cols.extend(default_exclude)
        exclude_cols = list(set(exclude_cols))

        numeric_cols = [col for col in df.columns
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        print(f"📊 표준화 대상: {len(numeric_cols)}개 특성")

        for col in numeric_cols:
            scaler = StandardScaler()
            scaler.fit(df[[col]])
            self.scalers[col] = scaler

        self.is_fitted = True
        print(f"✅ Scaler 학습 완료: {len(self.scalers)}개 특성")

        return self

    def transform_dataframe(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """데이터프레임을 표준화 후 원통좌표로 변환"""
        if not self.is_fitted:
            raise ValueError("Scaler가 아직 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        if exclude_cols is None:
            exclude_cols = []

        potential_exclude = ['level', 'level_name', 'anxiety_score', 'depression_score',
                            'stress_score', 'anxiety_binary', 'depression_binary',
                            'stress_binary', 'anxiety_label', 'depression_label', 'stress_label']

        default_exclude = [col for col in potential_exclude if col in df.columns]
        exclude_cols.extend(default_exclude)
        exclude_cols = list(set(exclude_cols))

        numeric_cols = [col for col in df.columns
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        print(f"📊 원통좌표 변환 대상: {len(numeric_cols)}개 특성")

        # 1단계: 표준화
        df_standardized = df.copy()
        for col in numeric_cols:
            if col in self.scalers:
                df_standardized[col] = self.scalers[col].transform(df[[col]])

        print(f"✅ 표준화 완료")

        # 결과 데이터프레임 생성
        existing_exclude_cols = [col for col in exclude_cols if col in df.columns]
        result_df = df[existing_exclude_cols].copy() if existing_exclude_cols else pd.DataFrame(index=df.index)

        # 2단계: 3개씩 묶어서 원통좌표 변환
        self.feature_triples = []
        self.cylindrical_features = []

        for i in range(0, len(numeric_cols), 3):
            if i + 2 < len(numeric_cols):
                col1, col2, col3 = numeric_cols[i], numeric_cols[i+1], numeric_cols[i+2]

                x = df_standardized[col1].values
                y = df_standardized[col2].values
                z = df_standardized[col3].values

                rho, phi, z_cyl = self.transform_triple_to_cylindrical(x, y, z)

                triple_name = f"{col1}_{col2}_{col3}"
                rho_name = f"rho_{triple_name}"
                phi_name = f"phi_{triple_name}"
                z_name = f"z_{triple_name}"

                result_df[rho_name] = rho
                result_df[phi_name] = phi
                result_df[z_name] = z_cyl

                self.feature_triples.append((col1, col2, col3))
                self.cylindrical_features.extend([rho_name, phi_name, z_name])

            else:
                # 남은 특성들은 표준화된 값 유지
                for j in range(i, len(numeric_cols)):
                    col = numeric_cols[j]
                    result_df[col] = df_standardized[col].values
                    self.cylindrical_features.append(col)

        print(f"✅ 원통좌표 변환 완료: {len(self.feature_triples)}개 트리플 -> {len(self.cylindrical_features)}개 특성")

        return result_df

    def fit_transform_dataframe(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """fit과 transform을 한번에 수행"""
        self.fit(df, exclude_cols)
        return self.transform_dataframe(df, exclude_cols)

    def transform_splits(self, data_splits_dir: str = "data_splits",
                        output_dir: str = "data_splits_standardized_cylinder",
                        target_vars: List[str] = None,
                        save_scalers: bool = True):
        """train/val/test 분할 데이터를 모두 표준화 후 원통좌표로 변환"""
        if target_vars is None:
            target_vars = ['anxiety', 'depression', 'stress']

        data_splits_path = Path(data_splits_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"🔄 표준화 기반 원통좌표 변환 시작")
        print(f"{'='*60}\n")

        for target in target_vars:
            print(f"\n📌 {target.upper()} 데이터 변환 중...")

            train_file = data_splits_path / f"{target}_train.csv"
            if not train_file.exists():
                print(f"  ⚠️  Train 파일 없음: {train_file.name}")
                continue

            df_train = pd.read_csv(train_file)
            print(f"  📂 Train: {len(df_train):,}행 x {len(df_train.columns)}열")

            df_train_cylindrical = self.fit_transform_dataframe(df_train)

            output_file = output_path / f"{target}_train_standardized_cylindrical.csv"
            df_train_cylindrical.to_csv(output_file, index=False)
            print(f"  ✅ 저장 완료: {output_file.name} ({len(df_train_cylindrical):,}행 x {len(df_train_cylindrical.columns)}열)")

            if save_scalers:
                scaler_file = output_path / f"{target}_scalers.pkl"
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.scalers, f)
                print(f"  💾 Scaler 저장: {scaler_file.name}")

            for split in ['val', 'test']:
                input_file = data_splits_path / f"{target}_{split}.csv"
                output_file = output_path / f"{target}_{split}_standardized_cylindrical.csv"

                if not input_file.exists():
                    print(f"  ⚠️  파일 없음: {input_file.name}")
                    continue

                df = pd.read_csv(input_file)
                print(f"  📂 {split}: {len(df):,}행 x {len(df.columns)}열")

                df_cylindrical = self.transform_dataframe(df)

                df_cylindrical.to_csv(output_file, index=False)
                print(f"  ✅ 저장 완료: {output_file.name} ({len(df_cylindrical):,}행 x {len(df_cylindrical.columns)}열)")

        print(f"\n{'='*60}")
        print(f"✅ 모든 데이터 표준화 기반 원통좌표 변환 완료")
        print(f"{'='*60}\n")

        summary_df = pd.DataFrame({
            'Feature_Triple': [f"{t[0]} + {t[1]} + {t[2]}" for t in self.feature_triples],
            'Rho_Feature': [f"rho_{t[0]}_{t[1]}_{t[2]}" for t in self.feature_triples],
            'Phi_Feature': [f"phi_{t[0]}_{t[1]}_{t[2]}" for t in self.feature_triples],
            'Z_Feature': [f"z_{t[0]}_{t[1]}_{t[2]}" for t in self.feature_triples]
        })

        summary_df.to_csv(output_path / "standardized_cylindrical_transformation_summary.csv", index=False)
        print(f"📊 변환 요약 저장: standardized_cylindrical_transformation_summary.csv\n")

        return {
            'target_variables': target_vars,
            'n_feature_triples': len(self.feature_triples),
            'n_cylindrical_features': len(self.cylindrical_features),
            'feature_triples': [f"{t[0]} + {t[1]} + {t[2]}" for t in self.feature_triples],
            'cylindrical_features': self.cylindrical_features
        }


def main():
    """메인 실행 함수"""
    transformer = StandardizedCylindricalTransformer()
    summary = transformer.transform_splits()

    print("📋 변환 요약:")
    print(f"  - 특성 트리플 수: {summary['n_feature_triples']}")
    print(f"  - 원통좌표 특성 수: {summary['n_cylindrical_features']}")


if __name__ == "__main__":
    main()
