"""
종속변수 이진 분류 변환 모듈
6점 이상: 발생 (1), 6점 미만: 미발생 (0)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob


class BinaryClassificationConverter:
    """종속변수를 이진 분류로 변환하는 클래스"""

    def __init__(self, threshold: float = 4.0):
        """
        Args:
            threshold: 발생/미발생 구분 기준점 (기본값: 4.0)
        """
        self.threshold = threshold

    def convert_to_binary(self, data: pd.DataFrame, target_var: str) -> pd.DataFrame:
        """
        연속형 점수를 이진 분류로 변환

        Args:
            data: 계층화 데이터
            target_var: 타겟 변수명 (anxiety, depression, stress)

        Returns:
            이진 분류 컬럼이 추가된 데이터프레임
        """
        score_col = f'{target_var}_score'
        binary_col = f'{target_var}_binary'
        label_col = f'{target_var}_label'

        # 이진 분류 생성 (발생: 1, 미발생: 0)
        data[binary_col] = (data[score_col] >= self.threshold).astype(int)

        # 레이블 생성 (발생/미발생)
        data[label_col] = data[binary_col].map({1: '발생', 0: '미발생'})

        return data

    def analyze_distribution(self, data: pd.DataFrame, target_var: str) -> dict:
        """발생/미발생 분포 분석"""
        binary_col = f'{target_var}_binary'

        total = len(data)
        positive = (data[binary_col] == 1).sum()
        negative = (data[binary_col] == 0).sum()

        stats = {
            'target': target_var,
            'threshold': self.threshold,
            'total_samples': total,
            'positive_count': positive,
            'negative_count': negative,
            'positive_ratio': positive / total,
            'negative_ratio': negative / total,
            'class_balance': min(positive, negative) / max(positive, negative)
        }

        return stats

    def analyze_by_level(self, data: pd.DataFrame, target_var: str) -> pd.DataFrame:
        """계층별 발생/미발생 분포 분석"""
        binary_col = f'{target_var}_binary'

        level_stats = []
        for level in sorted(data['level'].unique()):
            level_data = data[data['level'] == level]
            total = len(level_data)
            positive = (level_data[binary_col] == 1).sum()

            level_stats.append({
                'level': level,
                'total': total,
                'positive': positive,
                'negative': total - positive,
                'positive_ratio': positive / total if total > 0 else 0
            })

        return pd.DataFrame(level_stats)

    def save_binary_data(self, data: pd.DataFrame, target_var: str, output_dir: str = "hierarchical_data"):
        """이진 분류 데이터 저장"""
        output_path = Path(output_dir) / f"{target_var}_binary_classification.csv"
        data.to_csv(output_path, index=False)

        print(f"   ✓ 이진 분류 데이터 저장: {output_path}")
        return output_path

    def convert_all_hierarchical_data(self, input_pattern: str = "hierarchical_data/*_hierarchical_data.csv",
                                     output_dir: str = "hierarchical_data"):
        """모든 계층화 데이터를 이진 분류로 변환"""
        print("\n" + "="*70)
        print("종속변수 이진 분류 변환")
        print(f"기준: {self.threshold}점 이상 = 발생, 미만 = 미발생")
        print("="*70)

        # 파일 찾기
        files = glob.glob(input_pattern)

        # 중복 제거
        unique_files = {}
        for filepath in files:
            filename = Path(filepath).stem
            target = filename.split('_')[0]
            if target not in unique_files:
                unique_files[target] = filepath

        results = {}

        for target, filepath in unique_files.items():
            print(f"\n처리 중: {target.upper()}")
            print(f"입력 파일: {filepath}")

            # 데이터 로드
            data = pd.read_csv(filepath)
            print(f"   샘플 수: {len(data):,}")

            # 이진 분류 변환
            data = self.convert_to_binary(data, target)

            # 분포 분석
            stats = self.analyze_distribution(data, target)
            print(f"\n   📊 분포 분석:")
            print(f"      발생: {stats['positive_count']:,}개 ({stats['positive_ratio']:.1%})")
            print(f"      미발생: {stats['negative_count']:,}개 ({stats['negative_ratio']:.1%})")
            print(f"      클래스 균형: {stats['class_balance']:.2f}")

            # 계층별 분석
            level_stats = self.analyze_by_level(data, target)
            print(f"\n   📊 계층별 발생률:")
            for _, row in level_stats.iterrows():
                print(f"      레벨 {int(row['level'])}: {row['positive_ratio']:.1%} "
                      f"({int(row['positive'])}/{int(row['total'])})")

            # 저장
            output_path = self.save_binary_data(data, target, output_dir)

            results[target] = {
                'stats': stats,
                'level_stats': level_stats,
                'output_path': output_path,
                'data': data
            }

        print("\n" + "="*70)
        print("✅ 이진 분류 변환 완료!")
        print(f"   📁 저장 위치: {output_dir}")
        print("="*70 + "\n")

        return results

    def create_summary_report(self, results: dict) -> pd.DataFrame:
        """변환 결과 요약 리포트"""
        summary_data = []

        for target, result in results.items():
            stats = result['stats']
            summary_data.append({
                '종속변수': target.upper(),
                '전체 샘플': stats['total_samples'],
                '발생': stats['positive_count'],
                '미발생': stats['negative_count'],
                '발생률': f"{stats['positive_ratio']:.1%}",
                '클래스 균형': f"{stats['class_balance']:.2f}"
            })

        return pd.DataFrame(summary_data)


def main():
    """실행 예시"""
    # 변환기 생성 (기준점: 6.0)
    converter = BinaryClassificationConverter(threshold=6.0)

    # 모든 계층화 데이터 변환
    results = converter.convert_all_hierarchical_data()

    # 요약 리포트
    summary = converter.create_summary_report(results)
    print("\n📊 변환 요약:")
    print(summary.to_string(index=False))
    print()

    # 요약 저장
    summary_path = "hierarchical_data/binary_classification_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"✅ 요약 리포트 저장: {summary_path}\n")


if __name__ == "__main__":
    main()
