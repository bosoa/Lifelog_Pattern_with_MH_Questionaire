"""
표준화 기반 극좌표 변환 데이터용 Cox Proportional Hazards 생존 분석 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# 기존 SurvivalAnalyzer 클래스 import
import sys
sys.path.append(str(Path(__file__).parent))
from survival_analysis import SurvivalAnalyzer


class StandardizedPolarSurvivalAnalyzer(SurvivalAnalyzer):
    """표준화 기반 극좌표 변환 데이터용 생존 분석 클래스"""

    def __init__(self, data_splits_dir: str = "data_splits_standardized_polar",
                 output_dir: str = "model_results_standardized_polar"):
        super().__init__()
        self.data_splits_dir = Path(data_splits_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def prepare_data_for_analysis(self, target_var: str):
        """
        표준화 극좌표 데이터를 생존 분석용으로 준비

        Args:
            target_var: 타겟 변수명 (anxiety, depression, stress)

        Returns:
            생존 분석용 데이터프레임
        """
        # Train 데이터 로드
        train_file = self.data_splits_dir / f"{target_var}_train_standardized_polar.csv"
        if not train_file.exists():
            raise FileNotFoundError(f"Train 파일을 찾을 수 없습니다: {train_file}")

        data = pd.read_csv(train_file)
        print(f"📂 데이터 로드: {len(data):,}행 x {len(data.columns)}열")

        # 타겟 변수명 확인
        target_col = f'{target_var}_binary'
        if target_col not in data.columns:
            # anxiety_label 등으로 되어 있을 수 있음
            if f'{target_var}_label' in data.columns:
                data[target_col] = data[f'{target_var}_label']
            else:
                raise ValueError(f"타겟 컬럼을 찾을 수 없습니다: {target_col} 또는 {target_var}_label")

        # 생존 분석 데이터 준비
        survival_data = self.prepare_survival_data(data, target_var)

        return survival_data

    def run_analysis_for_target(self, target_var: str):
        """특정 타겟 변수에 대한 전체 생존 분석 실행"""
        print(f"\n{'='*70}")
        print(f"🔬 {target_var.upper()} 표준화 극좌표 생존 분석 시작")
        print(f"{'='*70}\n")

        # 데이터 준비
        survival_data = self.prepare_data_for_analysis(target_var)
        print(f"✅ 생존 분석 데이터 준비 완료: {len(survival_data):,}행")

        # 특성 컬럼 추출 (극좌표 특성만)
        exclude_cols = ['duration', 'event', 'level', 'level_name',
                       f'{target_var}_score', f'{target_var}_binary', f'{target_var}_label']

        features = [col for col in survival_data.columns if col not in exclude_cols]
        print(f"📊 분석 특성 수: {len(features)}개")
        print(f"   특성 목록: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")

        # Cox PH 모델 학습
        model_results = self.fit_cox_model(survival_data, features, target_var)

        # HTML 리포트 생성
        report_file = self.output_dir / f"{target_var}_survival_analysis_standardized_polar_report.html"

        # 데이터를 원본 형식에 맞게 조정 (리포트 생성을 위해)
        data_for_report = survival_data.copy()
        data_for_report[f'{target_var}_binary'] = data_for_report['event']

        self.generate_html_report(target_var, data_for_report, features, str(report_file))

        print(f"\n✅ 리포트 저장 완료: {report_file.name}\n")

        return model_results


def main():
    """메인 실행 함수"""
    analyzer = StandardizedPolarSurvivalAnalyzer()

    target_vars = ['anxiety', 'depression', 'stress']
    all_results = {}

    print(f"\n{'='*70}")
    print(f"🚀 표준화 기반 극좌표 생존 분석 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for target in target_vars:
        try:
            results = analyzer.run_analysis_for_target(target)
            all_results[target] = results

            if results and 'concordance' in results:
                print(f"   📊 {target.upper()} C-index: {results['concordance']:.4f}")

        except Exception as e:
            print(f"❌ {target} 분석 중 오류: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"✅ 표준화 기반 극좌표 생존 분석 완료")
    print(f"{'='*70}\n")

    # 결과 요약
    print("\n📊 성능 요약:")
    print(f"{'='*70}")
    for target, results in all_results.items():
        if results and 'concordance' in results:
            print(f"{target.upper():15s}: C-index = {results['concordance']:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
