"""
모델 비교 및 HTML 리포트 생성 통합 스크립트
"""
import sys
from pathlib import Path
import glob

sys.path.append('src')

from model_comparison import ModelComparison
from enhanced_html_report import EnhancedHTMLReportGenerator


def main():
    """메인 실행 함수"""
    print("\n" + "="*70)
    print("정신건강 예측 모델 비교 시스템")
    print("="*70)

    # 출력 디렉토리 생성
    output_dir = Path("model_results")
    output_dir.mkdir(exist_ok=True)

    # 계층화 데이터 파일 찾기
    hierarchical_files = glob.glob("hierarchical_data/*_hierarchical_data.csv")

    if not hierarchical_files:
        print("\n❌ 계층화 데이터 파일을 찾을 수 없습니다.")
        print("   먼저 'python3 src/hierarchical_data_generator.py'를 실행하세요.")
        return

    # 각 종속변수에 대해 모델 비교
    for filepath in hierarchical_files:
        filename = Path(filepath).stem
        target = filename.split('_')[0]  # anxiety, depression, stress

        print(f"\n{'='*70}")
        print(f"분석 대상: {target.upper()}")
        print(f"{'='*70}")

        # 모델 비교
        comparison = ModelComparison(output_dir=str(output_dir))

        try:
            results = comparison.train_all_models(
                hierarchical_data_path=filepath,
                test_size=0.2
            )

            # 요약 테이블
            summary = comparison.get_comparison_summary()
            print(f"\n📊 {target.upper()} - 모델 성능 비교:")
            print(summary.to_string(index=False))

            # 계층화 데이터 로드
            import pandas as pd
            hierarchical_data = pd.read_csv(filepath)

            # HTML 리포트 생성 (계층화 데이터 시각화 포함)
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
