#!/usr/bin/env python3
"""
종속변수 데이터에 무한대 값이 있는지 체크하는 스크립트
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def check_infinity_in_file(file_path, target_col):
    """
    파일에서 종속변수 컬럼의 무한대 값 체크

    Args:
        file_path: 체크할 파일 경로
        target_col: 종속변수 컬럼명

    Returns:
        dict: 무한대 값 통계 정보
    """
    print(f"\n{'='*80}")
    print(f"파일 검사: {file_path}")
    print(f"{'='*80}")

    # 파일 읽기
    df = pd.read_csv(file_path)
    print(f"전체 행 수: {len(df):,}")
    print(f"전체 열 수: {len(df.columns)}")

    # 종속변수 컬럼 확인
    if target_col not in df.columns:
        print(f"⚠️  경고: '{target_col}' 컬럼을 찾을 수 없습니다.")
        print(f"사용 가능한 컬럼: {df.columns.tolist()}")
        return None

    # 무한대 값 체크
    target_data = df[target_col]

    # 양의 무한대
    pos_inf_count = np.isinf(target_data) & (target_data > 0)
    pos_inf_sum = pos_inf_count.sum()

    # 음의 무한대
    neg_inf_count = np.isinf(target_data) & (target_data < 0)
    neg_inf_sum = neg_inf_count.sum()

    # 전체 무한대
    total_inf = pos_inf_sum + neg_inf_sum

    # NaN 값
    nan_count = target_data.isna().sum()

    # 유한한 값의 통계
    finite_data = target_data[np.isfinite(target_data)]

    print(f"\n📊 '{target_col}' 컬럼 분석 결과:")
    print(f"  - 양의 무한대 (inf): {pos_inf_sum:,}개")
    print(f"  - 음의 무한대 (-inf): {neg_inf_sum:,}개")
    print(f"  - 전체 무한대: {total_inf:,}개 ({total_inf/len(df)*100:.2f}%)")
    print(f"  - NaN 값: {nan_count:,}개 ({nan_count/len(df)*100:.2f}%)")
    print(f"  - 유한한 값: {len(finite_data):,}개 ({len(finite_data)/len(df)*100:.2f}%)")

    if len(finite_data) > 0:
        print(f"\n📈 유한한 값의 통계:")
        print(f"  - 평균: {finite_data.mean():.4f}")
        print(f"  - 중앙값: {finite_data.median():.4f}")
        print(f"  - 최솟값: {finite_data.min():.4f}")
        print(f"  - 최댓값: {finite_data.max():.4f}")
        print(f"  - 표준편차: {finite_data.std():.4f}")

    # 무한대 값이 있는 행의 인덱스 출력 (처음 10개만)
    if total_inf > 0:
        inf_indices = df[np.isinf(target_data)].index.tolist()
        print(f"\n⚠️  무한대 값이 있는 행 인덱스 (처음 10개): {inf_indices[:10]}")

    return {
        'file': file_path,
        'total_rows': len(df),
        'pos_inf': pos_inf_sum,
        'neg_inf': neg_inf_sum,
        'total_inf': total_inf,
        'nan_count': nan_count,
        'finite_count': len(finite_data)
    }

def main():
    """메인 실행 함수"""
    print("🔍 종속변수 무한대 값 체크 시작...\n")

    # 체크할 파일 목록
    data_configs = [
        {
            'name': 'Binary Classification - Anxiety',
            'file': 'hierarchical_data/anxiety_binary_classification.csv',
            'target': 'anxiety_score'
        },
        {
            'name': 'Binary Classification - Depression',
            'file': 'hierarchical_data/depression_binary_classification.csv',
            'target': 'depression_score'
        },
        {
            'name': 'Binary Classification - Stress',
            'file': 'hierarchical_data/stress_binary_classification.csv',
            'target': 'stress_score'
        },
        {
            'name': 'Train Split - Anxiety',
            'file': 'data_splits/anxiety_train.csv',
            'target': 'anxiety_score'
        },
        {
            'name': 'Validation Split - Anxiety',
            'file': 'data_splits/anxiety_val.csv',
            'target': 'anxiety_score'
        },
        {
            'name': 'Test Split - Anxiety',
            'file': 'data_splits/anxiety_test.csv',
            'target': 'anxiety_score'
        },
        {
            'name': 'Train Split - Depression',
            'file': 'data_splits/depression_train.csv',
            'target': 'depression_score'
        },
        {
            'name': 'Validation Split - Depression',
            'file': 'data_splits/depression_val.csv',
            'target': 'depression_score'
        },
        {
            'name': 'Test Split - Depression',
            'file': 'data_splits/depression_test.csv',
            'target': 'depression_score'
        },
        {
            'name': 'Train Split - Stress',
            'file': 'data_splits/stress_train.csv',
            'target': 'stress_score'
        },
        {
            'name': 'Validation Split - Stress',
            'file': 'data_splits/stress_val.csv',
            'target': 'stress_score'
        },
        {
            'name': 'Test Split - Stress',
            'file': 'data_splits/stress_test.csv',
            'target': 'stress_score'
        }
    ]

    # 결과 저장
    results = []
    has_infinity = False

    # 각 파일 체크
    for config in data_configs:
        file_path = config['file']

        # 파일 존재 확인
        if not os.path.exists(file_path):
            print(f"⚠️  파일을 찾을 수 없습니다: {file_path}")
            continue

        # 무한대 값 체크
        result = check_infinity_in_file(file_path, config['target'])

        if result:
            results.append(result)
            if result['total_inf'] > 0:
                has_infinity = True

    # 최종 요약
    print("\n" + "="*80)
    print("📋 최종 요약")
    print("="*80)

    total_inf_all = sum(r['total_inf'] for r in results)
    total_rows_all = sum(r['total_rows'] for r in results)

    print(f"\n검사한 파일 수: {len(results)}개")
    print(f"전체 데이터 행 수: {total_rows_all:,}개")
    print(f"전체 무한대 값: {total_inf_all:,}개")

    if has_infinity:
        print("\n❌ 무한대 값이 발견되었습니다!")
        print("\n파일별 무한대 개수:")
        for r in results:
            if r['total_inf'] > 0:
                print(f"  - {Path(r['file']).name}: {r['total_inf']:,}개")
        return False
    else:
        print("\n✅ 모든 종속변수에 무한대 값이 없습니다!")
        print("   전체 분석을 진행할 수 있습니다.")
        return True

if __name__ == "__main__":
    result = main()
    exit(0 if result else 1)
