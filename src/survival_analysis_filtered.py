"""
유의미한 변수만 사용한 Cox Proportional Hazards 생존 분석
p-value < 0.05인 변수만 선택
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob

# 기존 SurvivalAnalyzer 클래스 import
from survival_analysis import SurvivalAnalyzer


class FilteredSurvivalAnalyzer(SurvivalAnalyzer):
    """p-value 기반 변수 선택 생존 분석 클래스"""

    def select_significant_features(self, data: pd.DataFrame, features: list,
                                    target_var: str, alpha: float = 0.05) -> tuple:
        """
        p-value < alpha인 유의미한 변수만 선택

        Args:
            data: 생존 분석 데이터
            features: 전체 독립변수 리스트
            target_var: 타겟 변수명
            alpha: 유의수준 (기본값: 0.05)

        Returns:
            (선택된 변수 리스트, 초기 모델 결과, 선택 통계)
        """
        print(f"\n🔍 {target_var.upper()} 변수 선택 중 (p-value < {alpha})...")

        # 생존 데이터 준비
        surv_data = self.prepare_survival_data(data, target_var)

        # 초기 모델 (모든 변수 사용)
        print("\n   📊 초기 모델 (전체 변수):")
        initial_result = self.fit_cox_model(surv_data, features, target_var)
        initial_cph = initial_result['model']

        # p-value 기반 변수 선택
        summary = initial_cph.summary
        significant_vars = summary[summary['p'] < alpha].index.tolist()

        # 통계
        total_vars = len(features)
        selected_vars = len(significant_vars)
        removed_vars = total_vars - selected_vars

        print(f"\n   ✅ 변수 선택 결과:")
        print(f"      전체 변수: {total_vars}개")
        print(f"      선택된 변수: {selected_vars}개 (p < {alpha})")
        print(f"      제거된 변수: {removed_vars}개")
        print(f"      선택률: {selected_vars/total_vars*100:.1f}%")

        if selected_vars == 0:
            print(f"\n   ⚠️  유의미한 변수가 없습니다. 전체 변수를 사용합니다.")
            significant_vars = features

        selection_stats = {
            'total_vars': total_vars,
            'selected_vars': selected_vars,
            'removed_vars': removed_vars,
            'selection_rate': selected_vars / total_vars,
            'selected_features': significant_vars,
            'removed_features': [f for f in features if f not in significant_vars]
        }

        return significant_vars, initial_result, selection_stats

    def generate_filtered_html_report(self, target_var: str, data: pd.DataFrame,
                                     features: list, output_path: str,
                                     alpha: float = 0.05):
        """p-value 필터링된 생존 분석 HTML 리포트 생성"""
        print(f"\n📄 {target_var.upper()} 필터링된 생존 분석 리포트 생성 중...")

        # 변수 선택
        selected_features, initial_result, selection_stats = \
            self.select_significant_features(data, features, target_var, alpha)

        # 생존 데이터 준비
        surv_data = self.prepare_survival_data(data, target_var)

        # 필터링된 모델 학습
        print(f"\n   🔬 필터링된 모델 ({len(selected_features)}개 변수):")
        filtered_result = self.fit_cox_model(surv_data, selected_features, target_var)
        filtered_cph = filtered_result['model']

        # 모델 비교
        print(f"\n   📊 모델 비교:")
        print(f"      초기 C-index: {initial_result['concordance_index']:.4f}")
        print(f"      필터 C-index: {filtered_result['concordance_index']:.4f}")
        print(f"      변화: {filtered_result['concordance_index'] - initial_result['concordance_index']:+.4f}")

        # 시각화 생성
        print("   📊 시각화 생성 중...")
        km_img = self.create_kaplan_meier_curves(surv_data, target_var)
        hr_img = self.create_hazard_ratios_plot(filtered_cph)
        cal_img = self.create_calibration_plot(surv_data, filtered_cph, target_var)
        nom_img = self.create_nomogram(filtered_cph, selected_features, target_var)

        # 변수 선택 시각화
        selection_img = self._create_variable_selection_plot(
            selection_stats, initial_result, target_var, alpha
        )

        # HTML 템플릿
        from datetime import datetime
        html_content = self._create_filtered_html_template(
            target_var=target_var,
            surv_data=surv_data,
            initial_result=initial_result,
            filtered_result=filtered_result,
            selection_stats=selection_stats,
            km_img=km_img,
            hr_img=hr_img,
            cal_img=cal_img,
            nom_img=nom_img,
            selection_img=selection_img,
            alpha=alpha
        )

        # 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"   ✓ 리포트 저장: {output_path}")
        return output_path

    def _create_variable_selection_plot(self, selection_stats: dict,
                                       initial_result: dict,
                                       target_var: str, alpha: float) -> str:
        """변수 선택 시각화"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. 변수 선택 파이 차트
        ax = axes[0]
        sizes = [selection_stats['selected_vars'], selection_stats['removed_vars']]
        colors = ['#4caf50', '#f44336']
        labels = [f"선택됨\n({selection_stats['selected_vars']}개)",
                 f"제거됨\n({selection_stats['removed_vars']}개)"]

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
              startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        ax.set_title(f'변수 선택 결과 (p < {alpha})',
                    fontsize=14, fontweight='bold', pad=20)

        # 2. p-value 분포
        ax = axes[1]
        summary = initial_result['summary']
        p_values = summary['p'].values

        ax.hist(p_values, bins=30, color='#667eea', alpha=0.7, edgecolor='black')
        ax.axvline(x=alpha, color='red', linestyle='--', linewidth=2,
                  label=f'α = {alpha}')
        ax.set_xlabel('p-value', fontsize=12, fontweight='bold')
        ax.set_ylabel('변수 개수', fontsize=12, fontweight='bold')
        ax.set_title('p-value 분포', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. 선택된 변수의 p-value
        ax = axes[2]
        selected_summary = summary[summary['p'] < alpha].copy()
        selected_summary = selected_summary.sort_values('p').head(15)

        y_pos = np.arange(len(selected_summary))
        ax.barh(y_pos, -np.log10(selected_summary['p']), color='#4caf50',
               alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(selected_summary.index, fontsize=9)
        ax.axvline(x=-np.log10(alpha), color='red', linestyle='--',
                  linewidth=2, label=f'α = {alpha}')
        ax.set_xlabel('-log10(p-value)', fontsize=12, fontweight='bold')
        ax.set_title('선택된 변수의 유의성 (상위 15개)',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        return self._fig_to_base64(fig)

    def _create_filtered_html_template(self, **kwargs) -> str:
        """필터링된 분석용 HTML 템플릿"""
        from datetime import datetime

        target_var = kwargs['target_var']
        surv_data = kwargs['surv_data']
        initial_result = kwargs['initial_result']
        filtered_result = kwargs['filtered_result']
        selection_stats = kwargs['selection_stats']
        km_img = kwargs['km_img']
        hr_img = kwargs['hr_img']
        cal_img = kwargs['cal_img']
        nom_img = kwargs['nom_img']
        selection_img = kwargs['selection_img']
        alpha = kwargs['alpha']

        # 성능 향상 계산
        c_index_diff = filtered_result['concordance_index'] - initial_result['concordance_index']
        c_index_improved = c_index_diff > 0

        html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{target_var.upper()} 필터링된 생존 분석 리포트</title>
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
            background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
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
            color: #4caf50;
            font-size: 2em;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 4px solid #4caf50;
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
            color: #4caf50;
            font-size: 0.95em;
            margin-bottom: 10px;
        }}
        .metric-card p {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .metric-card.improved {{
            background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
            color: white;
        }}
        .metric-card.improved h4,
        .metric-card.improved p {{
            color: white;
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
            background: #4caf50;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .insight-box {{
            background: linear-gradient(135deg, #4caf5015 0%, #45a04915 100%);
            border-left: 5px solid #4caf50;
            padding: 25px;
            margin: 30px 0;
            border-radius: 10px;
        }}
        .insight-box h4 {{
            color: #4caf50;
            margin-bottom: 15px;
        }}
        .comparison-table {{
            background: white;
            padding: 20px;
            border-radius: 12px;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            margin: 0 5px;
        }}
        .badge-success {{ background-color: #4caf50; color: white; }}
        .badge-info {{ background-color: #2196f3; color: white; }}
        .badge-warning {{ background-color: #ff9800; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 {target_var.upper()} 필터링된 생존 분석</h1>
            <p>유의미한 변수만 사용한 Cox Proportional Hazards 분석</p>
            <p>변수 선택 기준: p-value < {alpha}</p>
            <p style="margin-top: 10px; opacity: 0.8;">생성 일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="content">
            <!-- 변수 선택 결과 -->
            <div class="section">
                <h2>🔍 변수 선택 결과</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h4>전체 변수</h4>
                        <p>{selection_stats['total_vars']}</p>
                    </div>
                    <div class="metric-card improved">
                        <h4>선택된 변수</h4>
                        <p>{selection_stats['selected_vars']}</p>
                        <small style="color: rgba(255,255,255,0.9);">p < {alpha}</small>
                    </div>
                    <div class="metric-card">
                        <h4>제거된 변수</h4>
                        <p>{selection_stats['removed_vars']}</p>
                    </div>
                    <div class="metric-card">
                        <h4>선택률</h4>
                        <p>{selection_stats['selection_rate']*100:.1f}%</p>
                    </div>
                </div>

                <div class="chart-container">
                    <img src="{selection_img}" alt="Variable Selection" />
                </div>

                <div class="insight-box">
                    <h4>선택된 변수 목록</h4>
                    <p>{'<br>'.join([f'• {var}' for var in selection_stats['selected_features'][:20]])}</p>
                    {f"<p style='margin-top: 10px; color: #666;'>... 외 {len(selection_stats['selected_features']) - 20}개</p>" if len(selection_stats['selected_features']) > 20 else ""}
                </div>
            </div>

            <!-- 모델 비교 -->
            <div class="section">
                <h2>📊 모델 성능 비교</h2>
                <div class="comparison-table">
                    <table>
                        <tr>
                            <th>지표</th>
                            <th>초기 모델 (전체 변수)</th>
                            <th>필터링된 모델 (유의 변수)</th>
                            <th>변화</th>
                        </tr>
                        <tr>
                            <td><strong>변수 개수</strong></td>
                            <td>{selection_stats['total_vars']}개</td>
                            <td>{selection_stats['selected_vars']}개</td>
                            <td><span class="badge badge-info">{selection_stats['removed_vars']}개 감소</span></td>
                        </tr>
                        <tr>
                            <td><strong>Concordance Index</strong></td>
                            <td>{initial_result['concordance_index']:.4f}</td>
                            <td>{filtered_result['concordance_index']:.4f}</td>
                            <td><span class="badge {'badge-success' if c_index_improved else 'badge-warning'}">{c_index_diff:+.4f}</span></td>
                        </tr>
                        <tr>
                            <td><strong>Log-Likelihood</strong></td>
                            <td>{initial_result['log_likelihood']:.2f}</td>
                            <td>{filtered_result['log_likelihood']:.2f}</td>
                            <td>{filtered_result['log_likelihood'] - initial_result['log_likelihood']:+.2f}</td>
                        </tr>
                        <tr>
                            <td><strong>AIC (partial)</strong></td>
                            <td>{initial_result['AIC']:.2f}</td>
                            <td>{filtered_result['AIC']:.2f}</td>
                            <td>{filtered_result['AIC'] - initial_result['AIC']:+.2f}</td>
                        </tr>
                        <tr>
                            <td><strong>LR Test p-value</strong></td>
                            <td>{initial_result['lr_test_pvalue']:.4e}</td>
                            <td>{filtered_result['lr_test_pvalue']:.4e}</td>
                            <td>양쪽 모두 유의</td>
                        </tr>
                    </table>
                </div>

                <div class="insight-box">
                    <h4>해석</h4>
                    <p>{'✅ 변수를 줄이면서도 모델 성능이 향상되었습니다! 더 간결하고 해석 가능한 모델입니다.' if c_index_improved else '⚠️ 변수 개수는 감소했지만 성능은 소폭 하락했습니다. 제거된 변수들도 일부 기여를 했던 것으로 보입니다.'}</p>
                </div>
            </div>

            <!-- Kaplan-Meier Curves -->
            <div class="section">
                <h2>📈 Kaplan-Meier 생존 곡선</h2>
                <div class="chart-container">
                    <img src="{km_img}" alt="Kaplan-Meier Curves" />
                </div>
            </div>

            <!-- Hazard Ratios -->
            <div class="section">
                <h2>⚖️ Hazard Ratios (선택된 변수)</h2>
                <div class="chart-container">
                    <img src="{hr_img}" alt="Hazard Ratios" />
                </div>
            </div>

            <!-- Calibration Plot -->
            <div class="section">
                <h2>📊 Calibration Plot</h2>
                <div class="chart-container">
                    <img src="{cal_img}" alt="Calibration Plot" />
                </div>
            </div>

            <!-- Nomogram -->
            <div class="section">
                <h2>📐 Nomogram</h2>
                <div class="chart-container">
                    <img src="{nom_img}" alt="Nomogram" />
                </div>
            </div>

            <!-- Cox Model Summary -->
            <div class="section">
                <h2>📋 Cox 모델 계수 (선택된 변수)</h2>
                {filtered_result['summary'].to_html(classes='table', border=0)}
            </div>
        </div>

        <div style="background: #f8f9fa; text-align: center; padding: 30px; color: #666;">
            <p><strong>KLOSDOM Lifelog Pattern Data Generation System</strong></p>
            <p>Filtered Cox Proportional Hazards Survival Analysis (p < {alpha})</p>
        </div>
    </div>
</body>
</html>
"""
        return html


def main():
    """실행 예시"""
    import sys
    sys.path.append('src')
    from data_loader import KLOSDOMDataLoader

    # 출력 디렉토리
    output_dir = Path("model_results")
    output_dir.mkdir(exist_ok=True)

    analyzer = FilteredSurvivalAnalyzer()
    loader = KLOSDOMDataLoader()

    for target in ['anxiety', 'depression', 'stress']:
        print(f"\n{'='*70}")
        print(f"필터링된 생존 분석: {target.upper()}")
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
        data['level'] = pd.cut(y, bins=[0, 4, 7, 10], labels=[0, 1, 2], include_lowest=True)

        # 특징 컬럼
        features = feature_names_filtered

        # HTML 리포트 생성 (p < 0.05)
        output_path = output_dir / f"{target}_survival_analysis_filtered_report.html"
        analyzer.generate_filtered_html_report(target, data, features,
                                               str(output_path), alpha=0.05)

    print(f"\n{'='*70}")
    print("✅ 모든 필터링된 생존 분석 완료!")
    print(f"   📁 결과 디렉토리: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
