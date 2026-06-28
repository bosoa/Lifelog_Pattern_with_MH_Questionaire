"""
특성 그룹핑 방법 비교 분석 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# 기존 모듈들 import
sys.path.append(str(Path(__file__).parent))
from intelligent_feature_grouping import IntelligentFeatureGrouping
from survival_analysis import SurvivalAnalyzer


class GroupingMethodComparison:
    """특성 그룹핑 방법 비교 분석 클래스"""

    def __init__(self, data_splits_dir: str = "data_splits",
                 output_dir: str = "model_results_grouping_method_comparison"):
        self.data_splits_dir = Path(data_splits_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.grouper = IntelligentFeatureGrouping()
        self.results = {}

    def transform_with_grouping(self, df: pd.DataFrame, groups: list,
                                coord_type: str = 'polar') -> pd.DataFrame:
        """
        선택된 그룹으로 좌표계 변환 수행

        Args:
            df: 원본 데이터프레임
            groups: 특성 그룹 리스트 [[f1, f2, f3], ...]
            coord_type: 'polar', 'spherical', 'cylindrical'
        """
        result_df = df.copy()

        # 타겟 컬럼 제외
        exclude_patterns = ['level', 'score', 'binary', 'label']
        target_cols = [col for col in df.columns
                      if any(pattern in col for pattern in exclude_patterns)]

        # 결과 데이터프레임에 타겟 컬럼만 먼저 유지
        result_df = df[target_cols].copy()

        for group in groups:
            if len(group) < 2:
                continue

            if coord_type == 'polar' and len(group) >= 2:
                # 극좌표 변환 (2개)
                x = df[group[0]].values
                y = df[group[1]].values

                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)

                pair_name = f"{group[0]}_{group[1]}"
                result_df[f"r_{pair_name}"] = r
                result_df[f"theta_{pair_name}"] = theta

            elif coord_type in ['spherical', 'cylindrical'] and len(group) >= 3:
                # 구면/원통좌표 변환 (3개)
                x = df[group[0]].values
                y = df[group[1]].values
                z = df[group[2]].values

                if coord_type == 'spherical':
                    # 구면좌표
                    r = np.sqrt(x**2 + y**2 + z**2)
                    theta = np.arctan2(y, x)
                    phi = np.where(r > 1e-10, np.arccos(np.clip(z / r, -1.0, 1.0)), 0.0)

                    triple_name = f"{group[0]}_{group[1]}_{group[2]}"
                    result_df[f"r_{triple_name}"] = r
                    result_df[f"theta_{triple_name}"] = theta
                    result_df[f"phi_{triple_name}"] = phi

                else:  # cylindrical
                    # 원통좌표
                    rho = np.sqrt(x**2 + y**2)
                    phi = np.arctan2(y, x)
                    z_cyl = z

                    triple_name = f"{group[0]}_{group[1]}_{group[2]}"
                    result_df[f"rho_{triple_name}"] = rho
                    result_df[f"phi_{triple_name}"] = phi
                    result_df[f"z_{triple_name}"] = z_cyl

        return result_df

    def analyze_grouping_method(self, target_var: str, method: str, coord_type: str):
        """
        특정 그룹핑 방법과 좌표계로 분석 수행

        Args:
            target_var: 타겟 변수 ('anxiety', 'depression', 'stress')
            method: 그룹핑 방법 ('pca', 'correlation', 'hybrid', 'sequential')
            coord_type: 좌표계 ('polar', 'spherical', 'cylindrical')
        """
        print(f"\n{'='*70}")
        print(f"🔬 {target_var.upper()} - {method.upper()} - {coord_type.upper()}")
        print(f"{'='*70}\n")

        # 데이터 로드
        train_file = self.data_splits_dir / f"{target_var}_train.csv"
        df = pd.read_csv(train_file)

        # 제외할 컬럼
        exclude_cols = ['level', 'level_name', f'{target_var}_score',
                       f'{target_var}_binary', f'{target_var}_label']

        # 수치형 특성만 선택
        numeric_cols = [col for col in df.columns
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        # 그룹 선택
        if method == 'sequential':
            # 순차적 그룹핑 (베이스라인)
            if coord_type == 'polar':
                groups = [[numeric_cols[i], numeric_cols[i+1]]
                         for i in range(0, len(numeric_cols)-1, 2)]
            else:  # spherical or cylindrical
                groups = [[numeric_cols[i], numeric_cols[i+1], numeric_cols[i+2]]
                         for i in range(0, len(numeric_cols)-2, 3)]
        else:
            # 지능형 그룹핑
            df_features = df[numeric_cols]
            n_groups = min(3, len(numeric_cols) // 3)

            if method in self.grouper.grouping_methods:
                groups = self.grouper.grouping_methods[method](df_features, n_groups=n_groups)
            else:
                print(f"❌ 알 수 없는 방법: {method}")
                return None

        print(f"📊 선택된 그룹 수: {len(groups)}")
        for i, group in enumerate(groups, 1):
            print(f"  그룹 {i}: {group[:3]}...")  # 첫 3개만 표시

        # 좌표계 변환
        df_transformed = self.transform_with_grouping(df, groups, coord_type)

        # 생존 분석 데이터 준비
        analyzer = SurvivalAnalyzer()
        survival_data = analyzer.prepare_survival_data(df_transformed, target_var)

        # 특성 추출
        feature_cols = [col for col in survival_data.columns
                       if col not in ['duration', 'event', 'level', 'level_name',
                                     f'{target_var}_score', f'{target_var}_binary',
                                     f'{target_var}_label']]

        print(f"📈 변환된 특성 수: {len(feature_cols)}")

        # Cox PH 모델 학습
        try:
            model_results = analyzer.fit_cox_model(survival_data, feature_cols, target_var)

            result = {
                'target': target_var,
                'method': method,
                'coord_type': coord_type,
                'n_groups': len(groups),
                'n_features': len(feature_cols),
                'c_index': model_results.get('concordance_index', 0),
                'log_likelihood': model_results.get('log_likelihood', 0),
                'aic': model_results.get('AIC', 0),
                'groups': [str(g) for g in groups]
            }

            print(f"✅ C-index: {result['c_index']:.4f}")

            return result

        except Exception as e:
            print(f"❌ 분석 실패: {e}")
            return None

    def run_comprehensive_comparison(self):
        """모든 조합에 대한 종합 비교 분석"""
        print(f"\n{'='*70}")
        print(f"🚀 특성 그룹핑 방법 종합 비교 분석 시작")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        target_vars = ['anxiety', 'depression', 'stress']
        methods = ['sequential', 'pca', 'correlation', 'hybrid']
        coord_types = ['polar', 'spherical', 'cylindrical']

        all_results = []

        for target in target_vars:
            for method in methods:
                for coord_type in coord_types:
                    try:
                        result = self.analyze_grouping_method(target, method, coord_type)
                        if result:
                            all_results.append(result)
                    except Exception as e:
                        print(f"❌ {target}-{method}-{coord_type} 실패: {e}")

        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame(all_results)

        # 저장
        results_file = self.output_dir / "comparison_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n✅ 결과 저장: {results_file}")

        return results_df

    def generate_comparison_report(self, results_df: pd.DataFrame):
        """비교 분석 리포트 HTML 생성"""
        print(f"\n📄 비교 리포트 생성 중...")

        # 타겟별, 좌표계별 최고 성능 찾기
        best_by_target_coord = results_df.loc[
            results_df.groupby(['target', 'coord_type'])['c_index'].idxmax()
        ]

        # 방법별 평균 성능
        method_avg = results_df.groupby('method')['c_index'].agg(['mean', 'std'])

        # 좌표계별 평균 성능
        coord_avg = results_df.groupby('coord_type')['c_index'].agg(['mean', 'std'])

        html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>특성 그룹핑 방법 비교 분석</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 40px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        .best {{
            background: #d4edda !important;
            font-weight: bold;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 특성 그룹핑 방법 비교 분석 리포트</h1>
        <p><strong>생성 시각:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>총 분석 수:</strong> {len(results_df)}개 조합</p>

        <h2>📊 방법별 평균 성능</h2>
        <table>
            <thead>
                <tr>
                    <th>그룹핑 방법</th>
                    <th>평균 C-index</th>
                    <th>표준편차</th>
                    <th>순위</th>
                </tr>
            </thead>
            <tbody>
"""

        # 방법별 성능을 C-index 순으로 정렬
        method_sorted = method_avg.sort_values('mean', ascending=False)
        for rank, (method, row) in enumerate(method_sorted.iterrows(), 1):
            best_class = ' class="best"' if rank == 1 else ''
            html_content += f"""
                <tr{best_class}>
                    <td><strong>{method.upper()}</strong></td>
                    <td>{row['mean']:.4f}</td>
                    <td>{row['std']:.4f}</td>
                    <td>{rank}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>

        <h2>🎯 타겟 × 좌표계별 최고 성능</h2>
        <table>
            <thead>
                <tr>
                    <th>타겟</th>
                    <th>좌표계</th>
                    <th>최고 방법</th>
                    <th>C-index</th>
                    <th>특성 수</th>
                </tr>
            </thead>
            <tbody>
"""

        for _, row in best_by_target_coord.iterrows():
            html_content += f"""
                <tr class="best">
                    <td><strong>{row['target'].upper()}</strong></td>
                    <td>{row['coord_type']}</td>
                    <td>{row['method'].upper()}</td>
                    <td>{row['c_index']:.4f}</td>
                    <td>{row['n_features']}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>

        <h2>📈 전체 결과</h2>
        <table>
            <thead>
                <tr>
                    <th>타겟</th>
                    <th>방법</th>
                    <th>좌표계</th>
                    <th>C-index</th>
                    <th>그룹 수</th>
                    <th>특성 수</th>
                </tr>
            </thead>
            <tbody>
"""

        # C-index 순으로 정렬
        results_sorted = results_df.sort_values('c_index', ascending=False)
        for _, row in results_sorted.iterrows():
            html_content += f"""
                <tr>
                    <td>{row['target']}</td>
                    <td>{row['method']}</td>
                    <td>{row['coord_type']}</td>
                    <td>{row['c_index']:.4f}</td>
                    <td>{row['n_groups']}</td>
                    <td>{row['n_features']}</td>
                </tr>
"""

        html_content += """
            </tbody>
        </table>

        <h2>💡 주요 발견</h2>
        <ul>
"""

        # 주요 발견 자동 생성
        best_method = method_sorted.index[0]
        best_method_score = method_sorted.iloc[0]['mean']
        worst_method = method_sorted.index[-1]
        worst_method_score = method_sorted.iloc[-1]['mean']
        improvement = ((best_method_score - worst_method_score) / worst_method_score) * 100

        html_content += f"""
            <li><strong>최고 성능 방법:</strong> {best_method.upper()} (평균 C-index: {best_method_score:.4f})</li>
            <li><strong>성능 향상:</strong> {best_method.upper()}이 {worst_method.upper()} 대비 {improvement:.1f}% 우수</li>
            <li><strong>분석 조합 수:</strong> {len(results_df)}개 (3 타겟 × 4 방법 × 3 좌표계)</li>
        </ul>

        <div style="text-align: center; margin-top: 40px;">
            <a href="../model_results/index.html" style="
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 30px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 600;
            ">← 대시보드로 돌아가기</a>
        </div>
    </div>
</body>
</html>
"""

        # 저장
        report_file = self.output_dir / "comparison_report.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ 리포트 저장: {report_file}")

        return str(report_file)


def main():
    """메인 실행 함수"""
    comparer = GroupingMethodComparison()

    # 종합 비교 분석 실행
    results_df = comparer.run_comprehensive_comparison()

    # 리포트 생성
    comparer.generate_comparison_report(results_df)

    print(f"\n{'='*70}")
    print(f"✅ 특성 그룹핑 방법 비교 분석 완료")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
