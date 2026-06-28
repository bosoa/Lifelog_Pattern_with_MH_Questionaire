"""
원통좌표 변환 데이터용 이진 분류 데이터 생성 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path


class CylindricalBinaryClassificationConverter:
    """원통좌표 변환 데이터를 이진 분류 형식으로 변환하는 클래스"""

    def __init__(self, input_dir: str = "data_splits_cylinder",
                 output_dir: str = "hierarchical_data_cylinder"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_binary_classification_data(self, target_var: str,
                                          threshold: float = None) -> pd.DataFrame:
        """
        원통좌표 변환된 데이터에서 이진 분류 데이터 생성

        Args:
            target_var: 타겟 변수 ('anxiety', 'depression', 'stress')
            threshold: 이진 분류 임계값 (None이면 중앙값 사용)

        Returns:
            이진 분류 데이터프레임
        """
        print(f"\n{'='*60}")
        print(f"📊 {target_var.upper()} 원통좌표 이진 분류 데이터 생성")
        print(f"{'='*60}\n")

        # Train 데이터 로드
        train_file = self.input_dir / f"{target_var}_train_cylinder.csv"
        if not train_file.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {train_file}")

        df_train = pd.read_csv(train_file)
        print(f"📂 Train 데이터: {len(df_train):,}행")

        # 타겟 스코어 컬럼
        score_col = f'{target_var}_score'
        if score_col not in df_train.columns:
            raise ValueError(f"타겟 스코어 컬럼을 찾을 수 없습니다: {score_col}")

        # 임계값 설정
        if threshold is None:
            threshold = df_train[score_col].median()
        print(f"🎯 이진 분류 임계값: {threshold:.2f}")

        # Val과 Test 데이터도 로드하여 합치기
        all_data = [df_train]

        for split in ['val', 'test']:
            split_file = self.input_dir / f"{target_var}_{split}_cylinder.csv"
            if split_file.exists():
                df_split = pd.read_csv(split_file)
                all_data.append(df_split)
                print(f"📂 {split.capitalize()} 데이터: {len(df_split):,}행")

        # 모든 데이터 합치기
        df_all = pd.concat(all_data, ignore_index=True)
        print(f"\n📊 전체 데이터: {len(df_all):,}행")

        # 이진 분류 레이블 생성
        df_all[f'{target_var}_binary'] = (df_all[score_col] > threshold).astype(int)
        df_all[f'{target_var}_label'] = df_all[f'{target_var}_binary'].map({0: '미발생', 1: '발생'})

        # 클래스 분포 확인
        class_dist = df_all[f'{target_var}_binary'].value_counts()
        print(f"\n📈 클래스 분포:")
        print(f"  - 미발생 (0): {class_dist.get(0, 0):,}개 ({class_dist.get(0, 0)/len(df_all)*100:.1f}%)")
        print(f"  - 발생 (1): {class_dist.get(1, 0):,}개 ({class_dist.get(1, 0)/len(df_all)*100:.1f}%)")

        # 저장
        output_file = self.output_dir / f"{target_var}_binary_classification_cylinder.csv"
        df_all.to_csv(output_file, index=False)
        print(f"\n✅ 저장 완료: {output_file.name}")

        return df_all

    def create_all_binary_classification_data(self):
        """모든 타겟 변수에 대해 이진 분류 데이터 생성"""
        target_vars = ['anxiety', 'depression', 'stress']
        results = {}

        print(f"\n{'='*60}")
        print(f"🚀 원통좌표 이진 분류 데이터 생성 시작")
        print(f"{'='*60}\n")

        for target_var in target_vars:
            try:
                df = self.create_binary_classification_data(target_var)
                results[target_var] = {
                    'n_samples': len(df),
                    'n_positive': (df[f'{target_var}_binary'] == 1).sum(),
                    'n_negative': (df[f'{target_var}_binary'] == 0).sum(),
                }
            except Exception as e:
                print(f"❌ {target_var} 처리 중 오류: {e}")
                results[target_var] = None

        # 요약 저장
        summary_data = []
        for target, result in results.items():
            if result:
                summary_data.append({
                    'target_variable': target,
                    'total_samples': result['n_samples'],
                    'positive_class': result['n_positive'],
                    'negative_class': result['n_negative'],
                    'positive_ratio': result['n_positive'] / result['n_samples'] * 100
                })

        summary_df = pd.DataFrame(summary_data)
        summary_file = self.output_dir / "binary_classification_summary_cylinder.csv"
        summary_df.to_csv(summary_file, index=False)

        print(f"\n{'='*60}")
        print(f"✅ 원통좌표 이진 분류 데이터 생성 완료")
        print(f"{'='*60}\n")
        print(summary_df.to_string(index=False))

        return results


def main():
    """메인 실행 함수"""
    converter = CylindricalBinaryClassificationConverter()
    results = converter.create_all_binary_classification_data()


if __name__ == "__main__":
    main()
