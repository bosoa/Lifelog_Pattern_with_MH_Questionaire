"""
정신건강 예측 모델 비교 시스템
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not available. XGBoost will be skipped.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Neural Network will be skipped.")

try:
    from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  Transformers not available. Time-Series Transformer will be skipped.")


class SimpleNeuralNetwork(nn.Module):
    """간단한 피드포워드 신경망"""

    def __init__(self, input_dim: int):
        super(SimpleNeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


class ModelComparison:
    """모델 성능 비교 클래스"""

    def __init__(self, output_dir: str = "model_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.models = {}
        self.results = {}
        self.predictions = {}

    def load_hierarchical_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """계층화 데이터 로드"""
        print(f"📂 데이터 로드: {Path(filepath).name}")

        # 데이터 로드
        data = pd.read_csv(filepath)

        # 타겟 변수 이름 추출
        target_cols = [col for col in data.columns if '_score' in col]
        if not target_cols:
            raise ValueError("타겟 변수를 찾을 수 없습니다.")

        target_col = target_cols[0]
        target_name = target_col.replace('_score', '')

        # 특징 컬럼 (level, level_name, target 제외)
        feature_cols = [col for col in data.columns
                       if col not in ['level', 'level_name', target_col]]

        X = data[feature_cols]
        y = data[target_col]

        print(f"   ✓ 샘플 수: {len(X):,}")
        print(f"   ✓ 특징 수: {len(feature_cols)}")
        print(f"   ✓ 타겟: {target_name}")

        return X, y, feature_cols

    def train_random_forest(
        self,
        X_train, y_train, X_test, y_test,
        **kwargs
    ) -> Dict:
        """Random Forest 학습"""
        print("\n🌲 Random Forest 학습 중...")

        model = RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            min_samples_split=kwargs.get('min_samples_split', 5),
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # 예측
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # 평가
        results = {
            'model_name': 'Random Forest',
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
        }

        self.models['RandomForest'] = model
        self.predictions['RandomForest'] = {'train': y_pred_train, 'test': y_pred_test}

        print(f"   ✓ Test R²: {results['test_r2']:.4f}")
        print(f"   ✓ Test MAE: {results['test_mae']:.4f}")

        return results

    def train_xgboost(
        self,
        X_train, y_train, X_test, y_test,
        **kwargs
    ) -> Dict:
        """XGBoost 학습"""
        if not XGB_AVAILABLE:
            return {
                'model_name': 'XGBoost',
                'error': 'XGBoost not available'
            }

        print("\n🚀 XGBoost 학습 중...")

        model = xgb.XGBRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # 예측
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # 평가
        results = {
            'model_name': 'XGBoost',
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_importance': dict(zip(X_train.columns, model.feature_importances_))
        }

        self.models['XGBoost'] = model
        self.predictions['XGBoost'] = {'train': y_pred_train, 'test': y_pred_test}

        print(f"   ✓ Test R²: {results['test_r2']:.4f}")
        print(f"   ✓ Test MAE: {results['test_mae']:.4f}")

        return results

    def train_neural_network(
        self,
        X_train, y_train, X_test, y_test,
        **kwargs
    ) -> Dict:
        """Neural Network 학습"""
        if not TORCH_AVAILABLE:
            return {
                'model_name': 'Neural Network',
                'error': 'PyTorch not available'
            }

        print("\n🧠 Neural Network 학습 중...")

        # 데이터 준비
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_test_tensor = torch.FloatTensor(y_test.values).view(-1, 1)

        # 데이터 로더
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # 모델 생성
        model = SimpleNeuralNetwork(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 학습
        epochs = kwargs.get('epochs', 20)
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # 예측
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train_tensor).numpy().flatten()
            y_pred_test = model(X_test_tensor).numpy().flatten()

        # 평가
        results = {
            'model_name': 'Neural Network',
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }

        self.models['NeuralNetwork'] = model
        self.predictions['NeuralNetwork'] = {'train': y_pred_train, 'test': y_pred_test}

        print(f"   ✓ Test R²: {results['test_r2']:.4f}")
        print(f"   ✓ Test MAE: {results['test_mae']:.4f}")

        return results

    def train_all_models(
        self,
        hierarchical_data_path: str,
        test_size: float = 0.2
    ) -> Dict:
        """모든 모델 학습 및 비교"""
        print("\n" + "="*60)
        print("모델 학습 및 비교 시작")
        print("="*60)

        # 데이터 로드
        X, y, feature_names = self.load_hierarchical_data(hierarchical_data_path)

        # 데이터 분할
        print(f"\n📊 데이터 분할 (Train: {100-test_size*100:.0f}%, Test: {test_size*100:.0f}%)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 정규화
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )

        # 모델 학습
        results = {}

        # 1. Random Forest
        results['RandomForest'] = self.train_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test
        )

        # 2. XGBoost
        results['XGBoost'] = self.train_xgboost(
            X_train_scaled, y_train, X_test_scaled, y_test
        )

        # 3. Neural Network
        results['NeuralNetwork'] = self.train_neural_network(
            X_train_scaled, y_train, X_test_scaled, y_test
        )

        # 4. Time-Series Transformer (시계열이 아니므로 스킵)
        print("\n⏭️  Time-Series Transformer: 시계열 데이터가 아니므로 스킵")

        self.results = results

        print("\n" + "="*60)
        print("✅ 모델 학습 완료!")
        print("="*60)

        return results

    def get_comparison_summary(self) -> pd.DataFrame:
        """모델 비교 요약 테이블"""
        summary_data = []

        for model_name, result in self.results.items():
            if 'error' not in result:
                summary_data.append({
                    '모델': result['model_name'],
                    'Train MAE': result['train_mae'],
                    'Test MAE': result['test_mae'],
                    'Train R²': result['train_r2'],
                    'Test R²': result['test_r2'],
                    'Overfit': result['train_r2'] - result['test_r2']
                })

        return pd.DataFrame(summary_data).sort_values('Test R²', ascending=False)


def main():
    """실행 예시"""
    import sys
    sys.path.append('src')
    from data_loader import KLOSDOMDataLoader
    import tempfile
    import os

    loader = KLOSDOMDataLoader()
    comparison = ModelComparison(output_dir="model_results")

    print("\n📊 원본 데이터(1-10점)로 모델 비교 수행")
    print("="*70)

    # 각 타겟별로 원본 데이터 로드 및 모델 학습
    for target in ['anxiety', 'depression', 'stress']:
        print(f"\n{'='*70}")
        print(f"{target.upper()} 모델 비교")
        print(f"{'='*70}")

        # 원본 데이터 로드
        X, y, feature_names = loader.prepare_pca_data(
            target_variable=target,
            min_data_points=10
        )

        print(f"   ✓ 샘플 수: {len(y):,}")
        print(f"   ✓ 점수 범위: {y.min():.0f} ~ {y.max():.0f}")
        print(f"   ✓ 평균: {y.mean():.2f}")

        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data = X.copy()
        data[f'{target}_score'] = y.values
        data.to_csv(temp_file.name, index=False)
        temp_file.close()

        # 모델 학습
        results = comparison.train_all_models(
            hierarchical_data_path=temp_file.name
        )

        # 임시 파일 삭제
        os.remove(temp_file.name)

        # 결과 요약
        summary = comparison.get_comparison_summary()
        print(f"\n📊 {target.upper()} 모델 비교 요약:")
        print(summary.to_string(index=False))

    print(f"\n{'='*70}")
    print("✅ 모든 모델 비교 완료!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
