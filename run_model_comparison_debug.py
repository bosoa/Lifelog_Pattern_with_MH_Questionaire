"""
모델 비교 및 HTML 리포트 생성 통합 스크립트 (디버그 버전)
"""
import sys
from pathlib import Path
import glob

sys.path.append('src')

print("=" * 70)
print("Step 1: Importing modules...")
print("=" * 70)

from model_comparison import ModelComparison
from enhanced_html_report import EnhancedHTMLReportGenerator

print("✅ Modules imported successfully\n")


def main():
    """메인 실행 함수"""
    print("\n" + "="*70)
    print("정신건강 예측 모델 비교 시스템")
    print("="*70)

    # 출력 디렉토리 생성
    output_dir = Path("model_results")
    output_dir.mkdir(exist_ok=True)
    print(f"✅ Output directory created: {output_dir}\n")

    # 계층화 데이터 파일 찾기
    hierarchical_files = glob.glob("hierarchical_data/*_hierarchical_data.csv")
    print(f"Found {len(hierarchical_files)} hierarchical data files:")
    for f in hierarchical_files:
        print(f"  - {f}")

    if not hierarchical_files:
        print("\n❌ 계층화 데이터 파일을 찾을 수 없습니다.")
        print("   먼저 'python3 src/hierarchical_data_generator.py'를 실행하세요.")
        return

    # 중복 제거 (anxiety가 2개 있음)
    unique_files = {}
    for filepath in hierarchical_files:
        filename = Path(filepath).stem
        target = filename.split('_')[0]  # anxiety, depression, stress
        if target not in unique_files:
            unique_files[target] = filepath

    print(f"\nProcessing {len(unique_files)} unique targets: {list(unique_files.keys())}\n")

    # 각 종속변수에 대해 모델 비교
    for target, filepath in unique_files.items():
        print(f"\n{'='*70}")
        print(f"분석 대상: {target.upper()}")
        print(f"파일: {filepath}")
        print(f"{'='*70}")

        # 모델 비교
        comparison = ModelComparison(output_dir=str(output_dir))

        try:
            print(f"\n>>> Training models for {target}...")

            # 데이터 로드
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            import pandas as pd

            X, y, feature_names = comparison.load_hierarchical_data(filepath)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

            results = {}
            results['RandomForest'] = comparison.train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
            results['XGBoost'] = comparison.train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test)

            print("\n⏭️  Neural Network 스킵 (시간 절약)")
            comparison.results = results

            # 요약 테이블
            summary = comparison.get_comparison_summary()
            print(f"\n📊 {target.upper()} - 모델 성능 비교:")
            print(summary.to_string(index=False))

            # 계층화 데이터 로드
            print(f"\n>>> Loading hierarchical data for HTML report...")
            import pandas as pd
            hierarchical_data = pd.read_csv(filepath)
            print(f"✅ Loaded {len(hierarchical_data)} rows")

            # HTML 리포트 생성 (계층화 데이터 시각화 포함)
            print(f"\n>>> Generating HTML report...")
            html_generator = EnhancedHTMLReportGenerator()
            report_path = output_dir / f"{target}_model_comparison_report.html"

            html_generator.generate_full_report(
                results=results,
                summary_df=summary,
                hierarchical_data=hierarchical_data,
                target_variable=target,
                output_path=str(report_path)
            )

            print(f"\n✅ {target.upper()} 분석 완료!")
            print(f"   📄 HTML 리포트: {report_path}")

        except Exception as e:
            print(f"\n❌ {target.upper()} 분석 실패: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("✅ 전체 분석 완료!")
    print(f"   📁 결과 디렉토리: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
