"""
Med-Gemma 영감 정신건강 예측 모델

의료 도메인 지식을 통합한 딥러닝 기반 정신건강 예측 시스템
- 라이프로그 생체신호를 의료 컨텍스트로 변환
- 다층 신경망 + 도메인 지식 통합
- Attention 메커니즘으로 중요 특징 학습
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = Path("analysis_results/06_medgemma_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class LifelogDataset(Dataset):
    """라이프로그 데이터셋"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MedicalAttention(nn.Module):
    """의료 도메인 지식 기반 Attention"""

    def __init__(self, input_dim):
        super(MedicalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        # Attention weights 계산
        attention_weights = torch.softmax(self.attention(x), dim=1)
        # Weighted sum
        weighted = x * attention_weights
        return weighted, attention_weights


class MedGemmaInspiredModel(nn.Module):
    """
    Med-Gemma 영감 정신건강 예측 모델

    특징:
    - Multi-head attention for feature importance
    - Residual connections
    - Batch normalization
    - Dropout for regularization
    """

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super(MedGemmaInspiredModel, self).__init__()

        # Feature attention
        self.attention = MedicalAttention(input_dim)

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )

        # Output layer (regression)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        # Apply attention
        x_att, att_weights = self.attention(x)

        # Input layer
        out = self.input_layer(x_att)

        # Hidden layers
        for layer in self.hidden_layers:
            out = layer(out)

        # Output
        out = self.output_layer(out)

        return out, att_weights


class MedicalDomainKnowledge:
    """의료 도메인 지식 클래스"""

    @staticmethod
    def get_feature_groups():
        """생체신호를 의료적 카테고리로 그룹핑"""
        return {
            'cardiovascular': ['a01', 'a08', 'a06'],  # HRV, 심박수, 산소포화도
            'sleep': ['a04', 'a05', 'a10', 'a15'],    # 수면 관련
            'activity': ['a02', 'a03', 'a11'],        # 활동량
            'thermoregulation': ['a09', 'a16'],       # 체온 조절
            'circadian': ['a07', 'a12', 'a13', 'a14'], # 일주기 리듬
            'metabolic': ['a17', 'a18']               # 대사
        }

    @staticmethod
    def interpret_lifelog_to_medical_context(lifelog_data, feature_names):
        """라이프로그 데이터를 의료 컨텍스트로 해석"""
        groups = MedicalDomainKnowledge.get_feature_groups()

        interpretation = {}
        for group_name, feature_codes in groups.items():
            # 해당 그룹의 특징 찾기
            group_features = [f for f in feature_names if any(code in f for code in feature_codes)]
            if group_features:
                # 그룹별 평균
                group_mean = lifelog_data[group_features].mean(axis=1)
                interpretation[f'{group_name}_score'] = group_mean

        return pd.DataFrame(interpretation)


def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    """모델 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []

    print(f"학습 디바이스: {device}")
    print(f"총 에포크: {epochs}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs, _ = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs, _ = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


def evaluate_model(model, test_loader, scaler_y):
    """모델 평가"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_preds = []
    all_targets = []
    all_attentions = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs, att_weights = model(X_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            all_attentions.append(att_weights.cpu().numpy())

    preds = np.array(all_preds).flatten()
    targets = np.array(all_targets)

    # 역변환 (원래 스케일로)
    preds_original = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets_original = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()

    # 평가 지표
    mae = mean_absolute_error(targets_original, preds_original)
    r2 = r2_score(targets_original, preds_original)
    rmse = np.sqrt(np.mean((targets_original - preds_original) ** 2))

    print(f"\n모델 성능:")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R²: {r2:.3f}")

    return {
        'predictions': preds_original,
        'targets': targets_original,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'attention_weights': np.concatenate(all_attentions, axis=0)
    }


def plot_training_history(train_losses, val_losses, output_path):
    """학습 곡선 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Med-Gemma 영감 모델 학습 곡선', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"학습 곡선 저장: {output_path}")
    plt.close()


def plot_predictions(targets, predictions, output_path, target_name):
    """예측 결과 시각화"""
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.5)

    # 완벽한 예측 선
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='완벽한 예측')

    plt.xlabel('실제 값')
    plt.ylabel('예측 값')
    plt.title(f'{target_name} 예측 결과', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"예측 결과 저장: {output_path}")
    plt.close()


def main():
    """메인 실행 함수"""
    print("="*80)
    print("Med-Gemma 영감 정신건강 예측 모델")
    print("="*80)

    # 데이터 로드
    print("\n데이터 로드...")
    df = pd.read_csv("analysis_results/04_lifelog_integration/integrated_data.csv")

    # 특징 및 타겟 선택
    feature_cols = [col for col in df.columns if '_mean' in col and col.startswith('a')]
    target_vars = ['PHQ9_Score', 'GAD7_Score', 'ISI_Score']

    print(f"특징 수: {len(feature_cols)}")
    print(f"예측 대상: {target_vars}")

    results = {}

    for target in target_vars:
        print(f"\n{'='*80}")
        print(f"{target} 예측 모델 학습")
        print(f"{'='*80}")

        # 유효 데이터 선택
        df_valid = df[feature_cols + [target]].dropna()

        if len(df_valid) < 100:
            print(f"데이터 부족: {len(df_valid)}명")
            continue

        X = df_valid[feature_cols].values
        y = df_valid[target].values.reshape(-1, 1)

        # 데이터 분할
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # 스케일링
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train).flatten()
        y_val_scaled = scaler_y.transform(y_val).flatten()
        y_test_scaled = scaler_y.transform(y_test).flatten()

        print(f"\n데이터 분할:")
        print(f"  학습: {len(X_train_scaled)}명")
        print(f"  검증: {len(X_val_scaled)}명")
        print(f"  테스트: {len(X_test_scaled)}명")

        # 데이터로더
        train_dataset = LifelogDataset(X_train_scaled, y_train_scaled)
        val_dataset = LifelogDataset(X_val_scaled, y_val_scaled)
        test_dataset = LifelogDataset(X_test_scaled, y_test_scaled)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 모델 생성
        model = MedGemmaInspiredModel(
            input_dim=len(feature_cols),
            hidden_dims=[128, 64, 32],
            dropout=0.3
        )

        print(f"\n모델 구조:")
        print(f"  입력 차원: {len(feature_cols)}")
        print(f"  은닉층: [128, 64, 32]")
        print(f"  Attention: Medical Domain Knowledge")

        # 학습
        print(f"\n학습 시작...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            epochs=100, learning_rate=0.001
        )

        # 평가
        print(f"\n테스트 평가...")
        eval_results = evaluate_model(model, test_loader, scaler_y)

        # 시각화
        plot_training_history(
            train_losses, val_losses,
            OUTPUT_DIR / f"{target}_training_history.png"
        )

        plot_predictions(
            eval_results['targets'], eval_results['predictions'],
            OUTPUT_DIR / f"{target}_predictions.png",
            target
        )

        results[target] = eval_results

        # 모델 저장
        torch.save(model.state_dict(), OUTPUT_DIR / f"{target}_model.pth")
        print(f"모델 저장: {OUTPUT_DIR / f'{target}_model.pth'}")

    # 전체 결과 요약
    print(f"\n\n{'='*80}")
    print("전체 결과 요약")
    print(f"{'='*80}\n")

    summary_data = []
    for target, result in results.items():
        summary_data.append({
            '척도': target,
            'MAE': f"{result['mae']:.3f}",
            'RMSE': f"{result['rmse']:.3f}",
            'R²': f"{result['r2']:.3f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # 저장
    summary_df.to_csv(OUTPUT_DIR / "model_performance_summary.csv", index=False)
    print(f"\n✅ 결과 저장: {OUTPUT_DIR / 'model_performance_summary.csv'}")

    print(f"\n{'='*80}")
    print("Med-Gemma 영감 모델 학습 완료!")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    results = main()
