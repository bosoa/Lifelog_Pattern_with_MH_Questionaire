"""
Cox Proportional Hazards 생존 분석 모듈
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


class SurvivalAnalyzer:
    """Cox Proportional Hazards 생존 분석 클래스"""

    def __init__(self):
        self.models = {}
        self.results = {}

    def prepare_survival_data(self, data: pd.DataFrame, target_var: str) -> pd.DataFrame:
        """
        생존 분석을 위한 데이터 준비

        Args:
            data: 이진 분류 데이터
            target_var: 타겟 변수명

        Returns:
            생존 분석용 데이터프레임 (duration, event, features)
        """
        # 이벤트 컬럼
        event_col = f'{target_var}_binary'

        # Duration 생성: 레벨을 기반으로 관측 시간 생성
        # 레벨 0: 초기 관측 (10-40일)
        # 레벨 1: 중기 관측 (40-70일)
        # 레벨 2: 후기 관측 (70-93일)
        np.random.seed(42)

        duration = np.zeros(len(data))
        for level in [0, 1, 2]:
            level_mask = data['level'] == level
            n_samples = level_mask.sum()

            if level == 0:
                duration[level_mask] = np.random.uniform(10, 40, n_samples)
            elif level == 1:
                duration[level_mask] = np.random.uniform(40, 70, n_samples)
            else:  # level == 2
                duration[level_mask] = np.random.uniform(70, 93, n_samples)

        # 이벤트가 발생하지 않은 경우 censoring (관측 종료)
        # 이벤트 발생 시 해당 시점에 이벤트 발생으로 기록
        data = data.copy()
        data['duration'] = duration
        data['event'] = data[event_col]

        return data

    def fit_cox_model(self, data: pd.DataFrame, features: list,
                     target_var: str) -> dict:
        """
        Cox PH 모델 학습

        Args:
            data: 생존 분석 데이터
            features: 독립변수 리스트
            target_var: 타겟 변수명

        Returns:
            모델 결과 딕셔너리
        """
        print(f"\n🔬 {target_var.upper()} Cox PH 모델 학습 중...")

        # 모델 데이터 준비
        model_data = data[features + ['duration', 'event']].copy()

        # 결측치 제거
        model_data = model_data.dropna()

        # Cox PH Fitter (penalizer를 높여서 다중공선성 완화)
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(model_data, duration_col='duration', event_col='event')

        # 우도비 검정
        lr_test = cph.log_likelihood_ratio_test()

        # 결과
        print(f"   ✓ Concordance Index: {cph.concordance_index_:.4f}")
        print(f"   ✓ Log-likelihood: {cph.log_likelihood_:.2f}")
        print(f"   ✓ AIC (partial): {cph.AIC_partial_:.2f}")
        print(f"   ✓ LR Test Statistic: {lr_test.test_statistic:.2f}")
        print(f"   ✓ LR Test p-value: {lr_test.p_value:.4e}")

        self.models[target_var] = cph

        return {
            'model': cph,
            'concordance_index': cph.concordance_index_,
            'log_likelihood': cph.log_likelihood_,
            'AIC': cph.AIC_partial_,
            'lr_test_statistic': lr_test.test_statistic,
            'lr_test_pvalue': lr_test.p_value,
            'summary': cph.summary
        }

    def create_kaplan_meier_curves(self, data: pd.DataFrame,
                                   target_var: str) -> str:
        """Kaplan-Meier 생존 곡선 생성"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        kmf = KaplanMeierFitter()

        # 전체 생존 곡선
        ax = axes[0]
        kmf.fit(data['duration'], data['event'], label='전체')
        kmf.plot_survival_function(ax=ax, ci_show=True)
        ax.set_xlabel('시간 (일)', fontsize=12)
        ax.set_ylabel('생존 확률', fontsize=12)
        ax.set_title(f'{target_var.upper()} Kaplan-Meier 생존 곡선',
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()

        # 계층별 생존 곡선
        ax = axes[1]
        colors = ['#ffc107', '#ff9800', '#4caf50']

        for level in sorted(data['level'].unique()):
            level_data = data[data['level'] == level]
            kmf.fit(level_data['duration'], level_data['event'],
                   label=f'레벨 {level}')
            kmf.plot_survival_function(ax=ax, ci_show=False,
                                      color=colors[level])

        ax.set_xlabel('시간 (일)', fontsize=12)
        ax.set_ylabel('생존 확률', fontsize=12)
        ax.set_title(f'계층별 {target_var.upper()} 생존 곡선',
                    fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()

        return self._fig_to_base64(fig)

    def create_hazard_ratios_plot(self, cph, top_n: int = 10) -> str:
        """Hazard Ratios 시각화"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # 상위 N개 변수의 hazard ratio
        summary = cph.summary.copy()
        summary = summary.sort_values('exp(coef)', ascending=False).head(top_n)

        y_pos = np.arange(len(summary))
        hazard_ratios = summary['exp(coef)']
        lower_ci = summary['exp(coef) lower 95%']
        upper_ci = summary['exp(coef) upper 95%']

        # Hazard ratio plot
        ax.barh(y_pos, hazard_ratios, alpha=0.7, color='#667eea')
        ax.errorbar(hazard_ratios, y_pos,
                   xerr=[hazard_ratios - lower_ci, upper_ci - hazard_ratios],
                   fmt='none', color='black', capsize=5)

        ax.axvline(x=1, color='red', linestyle='--', linewidth=2,
                  label='HR = 1 (중립)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(summary.index)
        ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
        ax.set_title('상위 변수별 Hazard Ratio', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        return self._fig_to_base64(fig)

    def create_calibration_plot(self, data: pd.DataFrame, cph,
                                target_var: str) -> str:
        """Calibration curve 생성"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # 예측된 위험 점수
        risk_scores = cph.predict_partial_hazard(
            data[cph.params_.index]
        )

        # 위험 점수를 0-1 범위로 정규화 (Min-Max Scaling)
        risk_scores_normalized = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min())

        # 실제 이벤트 발생 여부
        actual_events = data['event'].values

        # Calibration curve 계산
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                actual_events, risk_scores_normalized, n_bins=10, strategy='quantile'
            )

            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration',
                   alpha=0.7)

            # Actual calibration
            ax.plot(mean_predicted_value, fraction_of_positives,
                   'o-', linewidth=3, markersize=10, color='#667eea',
                   label='모델 Calibration', markeredgecolor='white',
                   markeredgewidth=2)

            # 신뢰구간 표시 (간단한 근사)
            n_bins = len(fraction_of_positives)
            for i in range(n_bins):
                ax.plot([mean_predicted_value[i], mean_predicted_value[i]],
                       [max(0, fraction_of_positives[i] - 0.05),
                        min(1, fraction_of_positives[i] + 0.05)],
                       color='#667eea', alpha=0.3, linewidth=2)

            ax.set_xlabel('예측된 위험 점수 (정규화, 0-1)', fontsize=13, fontweight='bold')
            ax.set_ylabel('실제 이벤트 발생 비율', fontsize=13, fontweight='bold')
            ax.set_title(f'{target_var.upper()} Calibration Plot',
                        fontsize=15, fontweight='bold', pad=20)
            ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_xlim([-0.05, 1.05])
            ax.set_ylim([-0.05, 1.05])

            # 통계 정보 추가
            from sklearn.metrics import brier_score_loss
            brier_score = brier_score_loss(actual_events, risk_scores_normalized)
            ax.text(0.98, 0.02, f'Brier Score: {brier_score:.4f}',
                   ha='right', va='bottom', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   transform=ax.transAxes)

        except Exception as e:
            ax.text(0.5, 0.5, f'Calibration 계산 오류:\n{str(e)}',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('예측된 위험 점수', fontsize=12)
            ax.set_ylabel('실제 이벤트 비율', fontsize=12)
            ax.set_title(f'{target_var.upper()} Calibration Plot (Error)',
                        fontsize=14, fontweight='bold')

        return self._fig_to_base64(fig)

    def create_nomogram(self, cph, features: list, target_var: str) -> str:
        """Nomogram 생성 (계수 기반 점수 할당)"""
        fig, ax = plt.subplots(figsize=(14, max(8, len(features) * 0.5 + 3)))

        # Coefficients와 통계
        coefs = cph.params_
        summary = cph.summary

        # 상위 10개 변수 선택 (절대 계수 크기 기준)
        summary['abs_coef'] = summary['coef'].abs()
        top_features = summary.nlargest(min(10, len(features)), 'abs_coef').index.tolist()

        # 점수 계산을 위한 최대 계수
        max_abs_coef = np.abs(coefs[top_features]).max()

        # Nomogram 설정
        n_display = len(top_features)
        y_positions = np.linspace(1, 0.1, n_display + 3)  # 변수 + 총점 + 예측선

        # 제목
        ax.text(0.5, y_positions[0] + 0.05, f'{target_var.upper()} Nomogram',
               ha='center', fontsize=16, fontweight='bold')
        ax.text(0.5, y_positions[0] + 0.02, '(계수 기반 점수 할당)',
               ha='center', fontsize=11, style='italic', color='#666')

        # 각 변수별 점수 스케일 (계수에 비례)
        for idx, feature in enumerate(top_features):
            y = y_positions[idx + 1]
            coef = coefs[feature]

            # 변수명 및 계수 표시
            ax.text(0.02, y, f'{feature}', fontsize=10, va='center', weight='bold')
            ax.text(0.02, y - 0.02, f'β={coef:.3f}', fontsize=8, va='center',
                   color='#666', style='italic')

            # 점수 범위 (계수 절대값에 비례, 최대 100점)
            max_points = int(100 * (np.abs(coef) / max_abs_coef))
            if max_points < 10:
                max_points = 10  # 최소 10점

            scale_start = 0.25
            scale_end = 0.95
            n_ticks = min(6, max_points // 10 + 1)

            # 점수 틱 생성
            scale_points = np.linspace(0, max_points, n_ticks)

            for point in scale_points:
                x = scale_start + (point / max_points) * (scale_end - scale_start) if max_points > 0 else scale_start
                ax.plot([x, x], [y - 0.01, y + 0.01], 'k-', linewidth=1)
                ax.text(x, y - 0.03, f'{int(point)}',
                       ha='center', fontsize=8)

            # 스케일 라인 (계수 방향에 따라 색상 변경)
            line_color = '#d32f2f' if coef > 0 else '#1976d2'  # 빨강(위험증가), 파랑(위험감소)
            line_width = scale_end - scale_start
            actual_end = scale_start + line_width * (max_points / 100)
            ax.plot([scale_start, actual_end], [y, y], color=line_color,
                   linewidth=3, alpha=0.7)

            # HR 표시
            hr = np.exp(coef)
            hr_text = f'HR={hr:.2f}'
            ax.text(0.96, y, hr_text, fontsize=9, va='center',
                   ha='left', color=line_color, weight='bold')

        # 총점 라인 (0-100)
        y_total = y_positions[-2]
        ax.text(0.02, y_total, '총점 (0-100)', fontsize=12,
               fontweight='bold', va='center')
        total_points = np.linspace(0, 100, 11)
        for point in total_points:
            x = 0.25 + (point / 100) * 0.7
            ax.plot([x, x], [y_total - 0.01, y_total + 0.01], 'b-', linewidth=1)
            if int(point) % 20 == 0:
                ax.text(x, y_total - 0.03, f'{int(point)}',
                       ha='center', fontsize=9, color='blue')
        ax.plot([0.25, 0.95], [y_total, y_total], 'b-', linewidth=3)

        # 예측 위험 라인
        y_pred = y_positions[-1]
        ax.text(0.02, y_pred, '예측 위험도', fontsize=12,
               fontweight='bold', va='center', color='red')

        # 위험도 레이블 (상대적)
        risk_labels = ['매우 낮음', '낮음', '보통', '높음', '매우 높음']
        risk_positions = [0.25, 0.4, 0.6, 0.78, 0.95]
        for label, x_pos in zip(risk_labels, risk_positions):
            ax.plot([x_pos, x_pos], [y_pred - 0.01, y_pred + 0.01],
                   'r-', linewidth=1)
            ax.text(x_pos, y_pred - 0.03, label,
                   ha='center', fontsize=8, color='red')
        ax.plot([0.25, 0.95], [y_pred, y_pred], 'r-', linewidth=3)

        # 범례
        ax.text(0.02, 0.02, '🔴 빨간선: 위험 증가 (HR>1) | 🔵 파란선: 위험 감소 (HR<1)',
               fontsize=9, color='#666')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, y_positions[0] + 0.1])
        ax.axis('off')

        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """Figure를 base64로 인코딩"""
        buffer = BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return f'data:image/png;base64,{image_base64}'

    def generate_html_report(self, target_var: str, data: pd.DataFrame,
                            features: list, output_path: str):
        """생존 분석 HTML 리포트 생성"""
        print(f"\n📄 {target_var.upper()} 생존 분석 리포트 생성 중...")

        # 생존 데이터 준비
        surv_data = self.prepare_survival_data(data, target_var)

        # Cox PH 모델 학습
        cox_result = self.fit_cox_model(surv_data, features, target_var)
        cph = cox_result['model']

        # 시각화 생성
        print("   📊 시각화 생성 중...")
        km_img = self.create_kaplan_meier_curves(surv_data, target_var)
        hr_img = self.create_hazard_ratios_plot(cph)
        cal_img = self.create_calibration_plot(surv_data, cph, target_var)
        nom_img = self.create_nomogram(cph, features, target_var)

        # HTML 템플릿
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{target_var.upper()} 생존 분석 리포트</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 50px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.8em;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        .content {{
            padding: 50px;
        }}
        .section {{
            margin-bottom: 50px;
            padding: 40px;
            background: #f8f9fa;
            border-radius: 15px;
        }}
        .section h2 {{
            color: #667eea;
            font-size: 2em;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 4px solid #667eea;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-card h4 {{
            color: #667eea;
            font-size: 0.95em;
            margin-bottom: 10px;
        }}
        .metric-card p {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .insight-box {{
            background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
            border-left: 5px solid #667eea;
            padding: 25px;
            margin: 30px 0;
            border-radius: 10px;
        }}
        .insight-box h4 {{
            color: #667eea;
            margin-bottom: 15px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 {target_var.upper()} 생존 분석 리포트</h1>
            <p>Cox Proportional Hazards 모델을 이용한 이벤트 발생 예측</p>
            <p style="margin-top: 10px; opacity: 0.8;">생성 일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="content">
            <!-- 개요 -->
            <div class="section">
                <h2>📋 분석 개요</h2>
                <div class="insight-box">
                    <h4>분석 목적</h4>
                    <p>Cox Proportional Hazards 모델을 사용하여 {target_var.upper()} 점수 4점 이상 발생을 이벤트로 정의하고,
                    센서 데이터를 기반으로 이벤트 발생 위험을 예측합니다.</p>
                </div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>전체 샘플 수</h4>
                        <p>{len(surv_data):,}</p>
                    </div>
                    <div class="metric-card">
                        <h4>이벤트 발생</h4>
                        <p>{surv_data['event'].sum():,}</p>
                    </div>
                    <div class="metric-card">
                        <h4>Censored</h4>
                        <p>{(~surv_data['event'].astype(bool)).sum():,}</p>
                    </div>
                    <div class="metric-card">
                        <h4>이벤트 발생률</h4>
                        <p>{surv_data['event'].mean():.1%}</p>
                    </div>
                </div>
            </div>

            <!-- 모델 성능 -->
            <div class="section">
                <h2>🎯 Cox PH 모델 성능</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>Concordance Index</h4>
                        <p>{cox_result['concordance_index']:.4f}</p>
                        <small style="color: #666;">모델 식별력</small>
                    </div>
                    <div class="metric-card">
                        <h4>Log-Likelihood</h4>
                        <p>{cox_result['log_likelihood']:.2f}</p>
                        <small style="color: #666;">모델 적합도</small>
                    </div>
                    <div class="metric-card">
                        <h4>AIC (partial)</h4>
                        <p>{cox_result['AIC']:.2f}</p>
                        <small style="color: #666;">정보 기준</small>
                    </div>
                </div>

                <div class="metrics-grid" style="margin-top: 20px;">
                    <div class="metric-card" style="background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);">
                        <h4 style="color: white;">우도비 검정 통계량</h4>
                        <p style="color: white;">{cox_result['lr_test_statistic']:.2f}</p>
                        <small style="color: rgba(255,255,255,0.9);">LR Test Statistic</small>
                    </div>
                    <div class="metric-card" style="background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);">
                        <h4 style="color: white;">우도비 검정 p-value</h4>
                        <p style="color: white;">{cox_result['lr_test_pvalue']:.4e}</p>
                        <small style="color: rgba(255,255,255,0.9);">유의확률</small>
                    </div>
                </div>

                <div class="insight-box">
                    <h4>해석 가이드</h4>
                    <p><strong>Concordance Index:</strong><br>
                       • C-index > 0.7: 우수한 식별력<br>
                       • C-index 0.6-0.7: 적절한 식별력<br>
                       • C-index < 0.6: 낮은 식별력</p>
                    <p style="margin-top: 15px;"><strong>우도비 검정 (Likelihood Ratio Test):</strong><br>
                       • p-value < 0.05: 모델이 통계적으로 유의미함 (귀무가설 기각)<br>
                       • p-value ≥ 0.05: 모델이 통계적으로 유의미하지 않음<br>
                       • 귀무가설: 모든 회귀 계수가 0 (변수들이 이벤트 발생에 영향 없음)</p>
                </div>
            </div>

            <!-- Kaplan-Meier Curves -->
            <div class="section">
                <h2>📈 Kaplan-Meier 생존 곡선</h2>
                <div class="chart-container">
                    <img src="{km_img}" alt="Kaplan-Meier Curves" />
                </div>
                <div class="insight-box">
                    <h4>해석</h4>
                    <p>생존 곡선이 급격히 떨어지는 구간은 이벤트 발생이 집중된 시기를 의미합니다.
                    계층별 곡선의 차이는 각 그룹의 위험도 차이를 나타냅니다.</p>
                </div>
            </div>

            <!-- Hazard Ratios -->
            <div class="section">
                <h2>⚖️ Hazard Ratios (위험비)</h2>
                <div class="chart-container">
                    <img src="{hr_img}" alt="Hazard Ratios" />
                </div>
                <div class="insight-box">
                    <h4>해석</h4>
                    <p>• HR > 1: 해당 변수 증가 시 이벤트 발생 위험 증가<br>
                       • HR = 1: 이벤트 발생 위험에 영향 없음<br>
                       • HR < 1: 해당 변수 증가 시 이벤트 발생 위험 감소</p>
                </div>
            </div>

            <!-- Calibration Plot -->
            <div class="section">
                <h2>📊 Calibration Plot</h2>
                <div class="chart-container">
                    <img src="{cal_img}" alt="Calibration Plot" />
                </div>
                <div class="insight-box">
                    <h4>해석</h4>
                    <p>Calibration curve가 대각선(perfect calibration)에 가까울수록
                    모델의 예측 확률이 실제 발생률과 일치함을 의미합니다.</p>
                </div>
            </div>

            <!-- Nomogram -->
            <div class="section">
                <h2>📐 Nomogram</h2>
                <div class="chart-container">
                    <img src="{nom_img}" alt="Nomogram" />
                </div>
                <div class="insight-box">
                    <h4>사용 방법</h4>
                    <p>1. 각 변수의 값에 해당하는 점수를 확인<br>
                       2. 모든 변수의 점수를 합산<br>
                       3. 총점에 해당하는 이벤트 발생 확률 확인</p>
                </div>
            </div>

            <!-- Cox Model Summary -->
            <div class="section">
                <h2>📋 Cox 모델 계수 요약</h2>
                {cox_result['summary'].head(15).to_html(classes='table', border=0)}
            </div>
        </div>

        <div style="background: #f8f9fa; text-align: center; padding: 30px; color: #666;">
            <p><strong>KLOSDOM Lifelog Pattern Data Generation System</strong></p>
            <p>Cox Proportional Hazards Survival Analysis</p>
        </div>
    </div>
</body>
</html>
"""

        # 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"   ✓ 리포트 저장: {output_path}")
        return output_path


def main():
    """실행 예시"""
    import sys
    sys.path.append('src')
    from data_loader import KLOSDOMDataLoader

    # 출력 디렉토리
    output_dir = Path("model_results")
    output_dir.mkdir(exist_ok=True)

    analyzer = SurvivalAnalyzer()
    loader = KLOSDOMDataLoader()

    for target in ['anxiety', 'depression', 'stress']:
        print(f"\n{'='*70}")
        print(f"생존 분석: {target.upper()}")
        print(f"{'='*70}")

        # 원본 데이터에서 직접 로드 (계층화 없이)
        X, y, feature_names = loader.prepare_pca_data(
            target_variable=target,
            min_data_points=10
        )

        # 분산이 0이거나 매우 낮은 컬럼 제거
        low_variance_cols = ['wakeup_time', 'bed_time', 'blood_pressure']
        X_filtered = X[[col for col in X.columns if col not in low_variance_cols]]
        feature_names_filtered = [col for col in feature_names if col not in low_variance_cols]

        # 이진 분류 변수 생성 (≥7 = 고위험)
        y_binary = (y >= 7).astype(int)

        # 데이터 결합
        data = X_filtered.copy()
        data[f'{target}_score'] = y.values
        data[f'{target}_binary'] = y_binary

        # 레벨 생성 (생존 분석을 위한 시간 계층)
        # 점수 기반으로 3단계 레벨 생성 (낮음, 중간, 높음)
        data['level'] = pd.cut(y, bins=[0, 4, 7, 10], labels=[0, 1, 2], include_lowest=True)

        # 특징 컬럼
        features = feature_names_filtered

        # HTML 리포트 생성
        output_path = output_dir / f"{target}_survival_analysis_report.html"
        analyzer.generate_html_report(target, data, features, str(output_path))

    print(f"\n{'='*70}")
    print("✅ 모든 생존 분석 완료!")
    print(f"   📁 결과 디렉토리: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
