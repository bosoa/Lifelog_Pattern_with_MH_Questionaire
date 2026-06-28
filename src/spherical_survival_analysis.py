"""
구면좌표 변환 데이터용 Cox Proportional Hazards 생존 분석 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

# 기존 SurvivalAnalyzer 클래스 import
import sys
sys.path.append(str(Path(__file__).parent))
from survival_analysis import SurvivalAnalyzer


class SphericalSurvivalAnalyzer(SurvivalAnalyzer):
    """구면좌표 변환 데이터용 생존 분석 클래스"""

    def __init__(self, data_dir: str = "hierarchical_data_sphere",
                 output_dir: str = "model_results_sphere"):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_analysis_for_target(self, target_var: str):
        """특정 타겟 변수에 대한 전체 생존 분석 실행"""
        print(f"\n{'='*70}")
        print(f"🔬 {target_var.upper()} 구면좌표 생존 분석 시작")
        print(f"{'='*70}\n")

        # 데이터 로드
        data_file = self.data_dir / f"{target_var}_binary_classification_sphere.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_file}")

        data = pd.read_csv(data_file)
        print(f"📂 데이터 로드: {len(data):,}행 x {len(data.columns)}열")

        # 생존 분석 데이터 준비
        survival_data = self.prepare_survival_data(data, target_var)
        print(f"✅ 생존 분석 데이터 준비 완료: {len(survival_data):,}행")

        # 특성 컬럼 추출 (구면좌표 특성만)
        exclude_cols = ['duration', 'event', 'level', 'level_name',
                       f'{target_var}_score', f'{target_var}_binary', f'{target_var}_label']

        features = [col for col in survival_data.columns if col not in exclude_cols]
        print(f"📊 분석 특성 수: {len(features)}개")
        print(f"   특성 목록: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")

        # Cox PH 모델 학습
        model_results = self.fit_cox_model(survival_data, features, target_var)

        # HTML 리포트 생성
        report_file = self.output_dir / f"{target_var}_survival_analysis_sphere_report.html"
        self.generate_html_report(target_var, data, features, str(report_file))

        print(f"\n✅ 리포트 저장 완료: {report_file.name}\n")

        return model_results


def main():
    """메인 실행 함수"""
    analyzer = SphericalSurvivalAnalyzer()

    target_vars = ['anxiety', 'depression', 'stress']
    all_results = {}

    print(f"\n{'='*70}")
    print(f"🚀 구면좌표 생존 분석 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for target in target_vars:
        try:
            results = analyzer.run_analysis_for_target(target)
            all_results[target] = results
        except Exception as e:
            print(f"❌ {target} 분석 중 오류: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"✅ 구면좌표 생존 분석 완료")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
