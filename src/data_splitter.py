"""
데이터셋 분할 모듈
Train:Validation:Test = 7:2:1 비율로 분할 및 저장
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import glob


class DataSplitter:
    """데이터를 Train/Validation/Test로 분할하는 클래스"""

    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.2,
                 test_ratio: float = 0.1, random_state: int = 42):
        """
        Args:
            train_ratio: 학습 데이터 비율 (기본값: 0.7)
            val_ratio: 검증 데이터 비율 (기본값: 0.2)
            test_ratio: 테스트 데이터 비율 (기본값: 0.1)
            random_state: 랜덤 시드 (재현성 보장)
        """
        # 비율 검증
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"비율의 합이 1.0이 아닙니다: {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    def split_data(self, data: pd.DataFrame, target_col: str,
                   stratify: bool = True) -> tuple:
        """
        데이터를 Train/Validation/Test로 분할

        Args:
            data: 전체 데이터프레임
            target_col: 타겟 컬럼명 (계층화 분할을 위해)
            stratify: 계층화 분할 여부 (클래스 비율 유지)

        Returns:
            (train_df, val_df, test_df) 튜플
        """
        # 1단계: Train + Val vs Test 분할
        stratify_col = data[target_col] if stratify else None

        train_val, test = train_test_split(
            data,
            test_size=self.test_ratio,
            random_state=self.random_state,
            stratify=stratify_col
        )

        # 2단계: Train vs Val 분할
        # val_ratio를 train+val 대비 비율로 조정
        val_ratio_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)

        stratify_col_trainval = train_val[target_col] if stratify else None

        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            random_state=self.random_state,
            stratify=stratify_col_trainval
        )

        return train, val, test

    def save_splits(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   target_var: str, output_dir: str = "data_splits"):
        """분할된 데이터를 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 파일 저장
        train_path = output_path / f"{target_var}_train.csv"
        val_path = output_path / f"{target_var}_val.csv"
        test_path = output_path / f"{target_var}_test.csv"

        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)

        print(f"   ✓ Train: {train_path} ({len(train):,}개)")
        print(f"   ✓ Val:   {val_path} ({len(val):,}개)")
        print(f"   ✓ Test:  {test_path} ({len(test):,}개)")

        return {
            'train_path': train_path,
            'val_path': val_path,
            'test_path': test_path
        }

    def analyze_split_distribution(self, train: pd.DataFrame, val: pd.DataFrame,
                                   test: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """분할된 데이터의 클래스 분포 분석"""
        def calc_dist(df, name):
            if target_col not in df.columns:
                return None

            total = len(df)
            if df[target_col].dtype in [int, np.int64, np.int32]:
                # 이진 분류
                positive = (df[target_col] == 1).sum()
                return {
                    'Split': name,
                    'Total': total,
                    'Positive': positive,
                    'Negative': total - positive,
                    'Positive %': f"{(positive/total)*100:.1f}%"
                }
            else:
                # 연속형
                return {
                    'Split': name,
                    'Total': total,
                    'Mean': df[target_col].mean(),
                    'Std': df[target_col].std(),
                    'Min': df[target_col].min(),
                    'Max': df[target_col].max()
                }

        stats = []
        for df, name in [(train, 'Train'), (val, 'Validation'), (test, 'Test')]:
            stat = calc_dist(df, name)
            if stat:
                stats.append(stat)

        return pd.DataFrame(stats) if stats else None

    def split_all_datasets(self, input_pattern: str = "hierarchical_data/*_binary_classification.csv",
                          output_dir: str = "data_splits"):
        """모든 이진 분류 데이터를 분할"""
        print("\n" + "="*70)
        print("데이터셋 분할 (Train:Val:Test = 7:2:1)")
        print(f"Random Seed: {self.random_state} (재현성 보장)")
        print("="*70)

        # 파일 찾기
        files = glob.glob(input_pattern)

        if not files:
            print(f"\n⚠️  파일을 찾을 수 없습니다: {input_pattern}")
            return {}

        results = {}

        for filepath in files:
            filename = Path(filepath).stem
            target = filename.split('_')[0]

            print(f"\n처리 중: {target.upper()}")
            print(f"입력 파일: {filepath}")

            # 데이터 로드
            data = pd.read_csv(filepath)
            print(f"   전체 샘플: {len(data):,}개")

            # 타겟 컬럼 확인
            binary_col = f'{target}_binary'
            if binary_col not in data.columns:
                print(f"   ⚠️  이진 분류 컬럼({binary_col})이 없습니다. 스킵합니다.")
                continue

            # 분할 (계층화 분할)
            train, val, test = self.split_data(data, binary_col, stratify=True)

            print(f"\n   📊 분할 결과:")
            print(f"      Train: {len(train):,}개 ({len(train)/len(data)*100:.1f}%)")
            print(f"      Val:   {len(val):,}개 ({len(val)/len(data)*100:.1f}%)")
            print(f"      Test:  {len(test):,}개 ({len(test)/len(data)*100:.1f}%)")

            # 클래스 분포 확인
            dist_stats = self.analyze_split_distribution(train, val, test, binary_col)
            if dist_stats is not None:
                print(f"\n   📊 클래스 분포 (발생률):")
                for _, row in dist_stats.iterrows():
                    print(f"      {row['Split']}: {row['Positive %']}")

            # 저장
            print(f"\n   💾 저장 중...")
            paths = self.save_splits(train, val, test, target, output_dir)

            results[target] = {
                'train': train,
                'val': val,
                'test': test,
                'paths': paths,
                'distribution': dist_stats
            }

        print("\n" + "="*70)
        print("✅ 데이터셋 분할 완료!")
        print(f"   📁 저장 위치: {output_dir}")
        print("="*70 + "\n")

        return results

    def create_split_summary(self, results: dict) -> pd.DataFrame:
        """분할 요약 리포트"""
        summary_data = []

        for target, result in results.items():
            train = result['train']
            val = result['val']
            test = result['test']

            total = len(train) + len(val) + len(test)

            summary_data.append({
                '종속변수': target.upper(),
                '전체': total,
                'Train': len(train),
                'Val': len(val),
                'Test': len(test),
                'Train %': f"{len(train)/total*100:.1f}%",
                'Val %': f"{len(val)/total*100:.1f}%",
                'Test %': f"{len(test)/total*100:.1f}%"
            })

        return pd.DataFrame(summary_data)


def main():
    """실행 예시"""
    # 데이터 분할기 생성 (7:2:1)
    splitter = DataSplitter(
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_state=42
    )

    # 모든 이진 분류 데이터 분할
    results = splitter.split_all_datasets()

    if results:
        # 요약 리포트
        summary = splitter.create_split_summary(results)
        print("\n📊 분할 요약:")
        print(summary.to_string(index=False))
        print()

        # 요약 저장
        summary_path = "data_splits/split_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"✅ 요약 리포트 저장: {summary_path}\n")


if __name__ == "__main__":
    main()
