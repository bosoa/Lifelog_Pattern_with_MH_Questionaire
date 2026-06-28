#!/usr/bin/env python3
"""
종속변수의 무한대 값을 최소 척도값(1)으로 변환하는 스크립트
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def fix_infinity_in_file(file_path, target_col, replacement_value=1.0):
    """
    파일에서 종속변수의 무한대 값을 대체

    Args:
        file_path: 파일 경로
        target_col: 종속변수 컬럼명
        replacement_value: 대체할 값 (기본값: 1.0)

    Returns:
        dict: 처리 결과
    """
    print(f"\n{'='*80}")
    print(f"파일 처리: {file_path}")
    print(f"{'='*80}")

    # 파일 읽기
    df = pd.read_csv(file_path)
    print(f"전체 행 수: {len(df):,}")

    # 종속변수 컬럼 확인
    if target_col not in df.columns:
        print(f"⚠️  '{target_col}' 컬럼을 찾을 수 없습니다.")
        return None

    # 무한대 값 개수 확인
    original_data = df[target_col].copy()
    inf_mask = np.isinf(original_data)
    inf_count = inf_mask.sum()

    if inf_count == 0:
        print(f"✅ 무한대 값이 없습니다. 처리 불필요.")
        return {
            'file': file_path,
            'inf_count': 0,
            'fixed': False
        }

    print(f"\n📊 처리 전:")
    print(f"  - 무한대 값: {inf_count:,}개 ({inf_count/len(df)*100:.2f}%)")

    # 무한대 값을 replacement_value로 대체
    df.loc[inf_mask, target_col] = replacement_value

    # 처리 후 확인
    remaining_inf = np.isinf(df[target_col]).sum()

    print(f"\n📊 처리 후:")
    print(f"  - 남은 무한대 값: {remaining_inf:,}개")
    print(f"  - 변환된 값: {replacement_value}")
    print(f"  - 변환된 행 수: {inf_count:,}개")

    # 백업 파일 생성 (원본 보존)
    backup_path = str(file_path).replace('.csv', '_backup.csv')
    if not Path(backup_path).exists():
        original_df = pd.read_csv(file_path)
        original_df.to_csv(backup_path, index=False)
        print(f"\n💾 백업 생성: {backup_path}")

    # 수정된 파일 저장
    df.to_csv(file_path, index=False)
    print(f"✅ 수정된 파일 저장: {file_path}")

    return {
        'file': file_path,
        'inf_count': inf_count,
        'replacement_value': replacement_value,
        'remaining_inf': remaining_inf,
        'fixed': True
    }

def main():
    """메인 실행 함수"""
    print("🔧 무한대 값 수정 시작...\n")

    # 처리할 파일 설정
    file_configs = [
        {
            'pattern': 'hierarchical_data/anxiety_binary_classification.csv',
            'target': 'anxiety_score'
        },
        {
            'pattern': 'hierarchical_data/depression_binary_classification.csv',
            'target': 'depression_score'
        },
        {
            'pattern': 'hierarchical_data/stress_binary_classification.csv',
            'target': 'stress_score'
        },
        {
            'pattern': 'data_splits/anxiety_*.csv',
            'target': 'anxiety_score'
        },
        {
            'pattern': 'data_splits/depression_*.csv',
            'target': 'depression_score'
        },
        {
            'pattern': 'data_splits/stress_*.csv',
            'target': 'stress_score'
        }
    ]

    results = []
    total_fixed = 0

    for config in file_configs:
        pattern = config['pattern']
        target_col = config['target']

        # 패턴에 맞는 파일 찾기
        files = glob.glob(pattern)

        for file_path in files:
            result = fix_infinity_in_file(file_path, target_col, replacement_value=1.0)
            if result:
                results.append(result)
                if result['fixed']:
                    total_fixed += 1

    # 최종 요약
    print("\n" + "="*80)
    print("📋 최종 요약")
    print("="*80)

    print(f"\n처리한 파일 수: {len(results)}개")
    print(f"수정된 파일 수: {total_fixed}개")

    if total_fixed > 0:
        print("\n수정된 파일 목록:")
        for r in results:
            if r['fixed']:
                print(f"  - {Path(r['file']).name}: {r['inf_count']:,}개 → {r['replacement_value']}")

        print("\n✅ 모든 무한대 값이 수정되었습니다!")
        print("   이제 전체 분석을 진행할 수 있습니다.")
    else:
        print("\n✅ 처리가 필요한 무한대 값이 없습니다.")

    return total_fixed > 0

if __name__ == "__main__":
    result = main()
    exit(0 if result else 0)
