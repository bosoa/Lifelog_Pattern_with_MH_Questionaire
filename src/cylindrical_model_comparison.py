"""
원통좌표 변환 데이터용 모델 비교 시스템
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 기존 ModelComparison 클래스 import
sys.path.append(str(Path(__file__).parent))
from model_comparison import ModelComparison


class CylindricalModelComparison(ModelComparison):
    """원통좌표 변환 데이터용 모델 비교 클래스"""

    def __init__(self, input_dir: str = "data_splits_cylinder",
                 output_dir: str = "model_results_cylinder"):
        super().__init__(output_dir=output_dir)
        self.input_dir = Path(input_dir)

    def run_comparison_for_target(self, target_var: str):
        """특정 타겟 변수에 대한 모델 비교 실행"""
        print(f"\n{'='*70}")
        print(f"🔬 {target_var.upper()} 원통좌표 모델 비교 분석")
        print(f"{'='*70}\n")

        # 데이터 파일
        train_file = self.input_dir / f"{target_var}_train_cylinder.csv"
        val_file = self.input_dir / f"{target_var}_val_cylinder.csv"
        test_file = self.input_dir / f"{target_var}_test_cylinder.csv"

        # 파일 존재 확인
        for file in [train_file, val_file, test_file]:
            if not file.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file}")

        # 데이터 로드
        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        df_test = pd.read_csv(test_file)

        print(f"📂 Train: {len(df_train):,}행")
        print(f"📂 Val:   {len(df_val):,}행")
        print(f"📂 Test:  {len(df_test):,}행")

        # 특성과 타겟 분리
        exclude_cols = ['level', 'level_name', f'{target_var}_score',
                       f'{target_var}_binary', f'{target_var}_label']

        feature_cols = [col for col in df_train.columns if col not in exclude_cols]
        target_col = f'{target_var}_score'

        print(f"\n📊 특성 수: {len(feature_cols)}개")
        print(f"🎯 타겟 변수: {target_col}")

        # 데이터 준비
        X_train = df_train[feature_cols]
        y_train = df_train[target_col]
        X_val = df_val[feature_cols]
        y_val = df_val[target_col]
        X_test = df_test[feature_cols]
        y_test = df_test[target_col]

        # 모델 학습 및 평가
        print(f"\n{'='*70}")
        print(f"🤖 모델 학습 시작")
        print(f"{'='*70}\n")

        results = {}

        # Random Forest
        print(f"🌲 Random Forest 학습 중...")
        try:
            rf_result = self.train_random_forest(X_train, y_train, X_test, y_test)
            val_pred = self.models['RandomForest'].predict(X_val)
            rf_result['val_r2'] = r2_score(y_val, val_pred)
            rf_result['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
            rf_result['val_mae'] = mean_absolute_error(y_val, val_pred)
            rf_result['train_rmse'] = np.sqrt(rf_result['train_mse'])
            rf_result['test_rmse'] = np.sqrt(rf_result['test_mse'])
            results['RandomForest'] = rf_result
            print(f"✅ Random Forest 완료 - Test R²: {rf_result['test_r2']:.4f}")
        except Exception as e:
            print(f"❌ Random Forest 실패: {e}")

        # XGBoost
        try:
            from model_comparison import XGB_AVAILABLE
            if XGB_AVAILABLE:
                print(f"\n🚀 XGBoost 학습 중...")
                xgb_result = self.train_xgboost(X_train, y_train, X_test, y_test)
                val_pred = self.models['XGBoost'].predict(X_val)
                xgb_result['val_r2'] = r2_score(y_val, val_pred)
                xgb_result['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
                xgb_result['val_mae'] = mean_absolute_error(y_val, val_pred)
                xgb_result['train_rmse'] = np.sqrt(xgb_result['train_mse'])
                xgb_result['test_rmse'] = np.sqrt(xgb_result['test_mse'])
                results['XGBoost'] = xgb_result
                print(f"✅ XGBoost 완료 - Test R²: {xgb_result['test_r2']:.4f}")
        except Exception as e:
            print(f"❌ XGBoost 실패: {e}")

        # Neural Network
        try:
            from model_comparison import TORCH_AVAILABLE
            if TORCH_AVAILABLE:
                print(f"\n🧠 Neural Network 학습 중...")
                nn_result = self.train_neural_network(X_train, y_train, X_test, y_test)
                import torch
                X_val_tensor = torch.FloatTensor(X_val.values)
                val_pred = self.models['NeuralNetwork'](X_val_tensor).detach().numpy().flatten()
                nn_result['val_r2'] = r2_score(y_val, val_pred)
                nn_result['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
                nn_result['val_mae'] = mean_absolute_error(y_val, val_pred)
                nn_result['train_rmse'] = np.sqrt(nn_result['train_mse'])
                nn_result['test_rmse'] = np.sqrt(nn_result['test_mse'])
                results['NeuralNetwork'] = nn_result
                print(f"✅ Neural Network 완료 - Test R²: {nn_result['test_r2']:.4f}")
        except Exception as e:
            print(f"❌ Neural Network 실패: {e}")

        # 결과 요약
        print(f"\n{'='*70}")
        print(f"📊 모델 성능 비교")
        print(f"{'='*70}\n")

        summary_data = []
        for model_name, result in results.items():
            summary_data.append({
                'Model': model_name,
                'Train_R2': result['train_r2'],
                'Val_R2': result['val_r2'],
                'Test_R2': result['test_r2'],
                'Test_RMSE': result['test_rmse'],
                'Test_MAE': result['test_mae']
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # 저장
        summary_file = self.output_dir / f"{target_var}_model_comparison_cylinder.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✅ 결과 저장: {summary_file.name}\n")

        return results


def main():
    """메인 실행 함수"""
    comparer = CylindricalModelComparison()

    target_vars = ['anxiety', 'depression', 'stress']
    all_results = {}

    print(f"\n{'='*70}")
    print(f"🚀 원통좌표 모델 비교 분석 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for target in target_vars:
        try:
            results = comparer.run_comparison_for_target(target)
            all_results[target] = results
        except Exception as e:
            print(f"❌ {target} 분석 중 오류: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"✅ 원통좌표 모델 비교 분석 완료")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
