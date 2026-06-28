"""
지능형 특성 그룹핑 모듈
PCA와 상관분석을 활용하여 의미 있는 특성 조합을 선택
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class IntelligentFeatureGrouping:
    """지능형 특성 그룹핑 클래스"""

    def __init__(self):
        self.grouping_methods = {
            'pca': self.pca_based_grouping,
            'correlation': self.correlation_based_grouping,
            'hybrid': self.hybrid_grouping
        }
        self.selected_groups = {}

    def pca_based_grouping(self, df: pd.DataFrame, n_groups: int = 3) -> List[List[str]]:
        """
        PCA 기반 특성 그룹핑

        각 주성분에서 가장 큰 기여도를 가진 특성들을 그룹화
        """
        print(f"\n📊 PCA 기반 특성 그룹핑 ({n_groups}개 그룹)")

        # 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        # PCA 수행 (주성분 수는 특성 수와 동일하게)
        pca = PCA()
        pca.fit(X_scaled)

        # 주성분별 특성 기여도
        components = pca.components_

        groups = []
        used_features = set()

        # 설명력이 높은 주성분부터 순회
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        print(f"  누적 설명력: PC1={pca.explained_variance_ratio_[0]:.3f}, " +
              f"PC2={cumulative_variance[1]:.3f}, PC3={cumulative_variance[2]:.3f}")

        for pc_idx in range(min(n_groups * 3, len(components))):
            if len(groups) >= n_groups:
                break

            # 현재 주성분에서 기여도가 높은 특성들 찾기
            loadings = np.abs(components[pc_idx])
            top_indices = loadings.argsort()[::-1]

            # 아직 사용되지 않은 상위 3개 특성 선택
            group = []
            for idx in top_indices:
                feature = df.columns[idx]
                if feature not in used_features and len(group) < 3:
                    group.append(feature)
                    used_features.add(feature)

            if len(group) == 3:  # 3개가 모였을 때만 그룹으로 추가
                groups.append(group)
                print(f"  그룹 {len(groups)} (PC{pc_idx+1}): {group}")

        return groups[:n_groups]

    def correlation_based_grouping(self, df: pd.DataFrame, n_groups: int = 3,
                                   threshold: float = 0.3) -> List[List[str]]:
        """
        상관분석 기반 특성 그룹핑

        상관관계가 높은 특성들을 함께 그룹화
        """
        print(f"\n📊 상관분석 기반 특성 그룹핑 ({n_groups}개 그룹, 임계값={threshold})")

        # 상관계수 행렬
        corr_matrix = df.corr().abs()

        groups = []
        used_features = set()

        # 각 특성의 평균 상관계수 계산 (자기 상관 제외)
        mean_corr = {}
        for col in df.columns:
            corr_values = corr_matrix[col].drop(col)
            mean_corr[col] = corr_values.mean()

        # 평균 상관계수가 높은 순서대로 정렬
        sorted_features = sorted(mean_corr.items(), key=lambda x: x[1], reverse=True)

        for base_feature, _ in sorted_features:
            if len(groups) >= n_groups:
                break

            if base_feature in used_features:
                continue

            # 새 그룹 시작
            group = [base_feature]
            used_features.add(base_feature)

            # 상관관계가 높은 특성 찾기
            correlations = corr_matrix[base_feature].sort_values(ascending=False)

            for feature in correlations.index:
                if feature == base_feature:
                    continue
                if feature not in used_features and len(group) < 3:
                    if correlations[feature] > threshold:
                        group.append(feature)
                        used_features.add(feature)

            # 3개가 되지 않으면 임계값을 낮춰서 추가
            if len(group) < 3:
                for feature in correlations.index:
                    if feature not in used_features and len(group) < 3:
                        group.append(feature)
                        used_features.add(feature)

            if len(group) == 3:
                groups.append(group)
                print(f"  그룹 {len(groups)}: {group}")
                # 그룹 내 평균 상관계수 계산
                group_corr = []
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        group_corr.append(corr_matrix.loc[group[i], group[j]])
                print(f"    평균 상관계수: {np.mean(group_corr):.3f}")

        return groups[:n_groups]

    def hybrid_grouping(self, df: pd.DataFrame, n_groups: int = 3) -> List[List[str]]:
        """
        하이브리드 그룹핑 (PCA + 상관분석)

        PCA로 주요 방향을 찾고, 상관분석으로 그룹 내 응집도 향상
        """
        print(f"\n📊 하이브리드 그룹핑 (PCA + 상관분석, {n_groups}개 그룹)")

        # 1단계: PCA로 후보 그룹 생성
        pca_groups = self.pca_based_grouping(df, n_groups=n_groups * 2)

        # 2단계: 각 그룹의 상관계수 계산
        corr_matrix = df.corr().abs()

        group_scores = []
        for group in pca_groups:
            if len(group) == 3:
                # 그룹 내 평균 상관계수
                corrs = []
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        corrs.append(corr_matrix.loc[group[i], group[j]])
                score = np.mean(corrs)
                group_scores.append((group, score))

        # 상관계수가 높은 순서로 정렬
        group_scores.sort(key=lambda x: x[1], reverse=True)

        # 상위 n_groups 선택
        selected_groups = [group for group, score in group_scores[:n_groups]]

        print(f"\n  선택된 {len(selected_groups)}개 그룹:")
        for i, (group, score) in enumerate(group_scores[:n_groups]):
            print(f"  그룹 {i+1} (상관계수={score:.3f}): {group}")

        return selected_groups

    def select_groups_for_target(self, data_splits_dir: str, target_var: str,
                                 method: str = 'hybrid') -> Dict:
        """
        특정 타겟에 대한 최적 특성 그룹 선택
        """
        print(f"\n{'='*60}")
        print(f"🎯 {target_var.upper()} - {method.upper()} 방법으로 특성 그룹 선택")
        print(f"{'='*60}")

        # Train 데이터 로드
        train_file = Path(data_splits_dir) / f"{target_var}_train.csv"
        df = pd.read_csv(train_file)

        # 제외할 컬럼
        exclude_cols = ['level', 'level_name', f'{target_var}_score',
                       f'{target_var}_binary', f'{target_var}_label']

        # 수치형 특성만 선택
        numeric_cols = [col for col in df.columns
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

        df_features = df[numeric_cols]

        print(f"📂 특성 수: {len(numeric_cols)}개")

        # 그룹핑 수행
        n_groups = min(3, len(numeric_cols) // 3)
        if method in self.grouping_methods:
            groups = self.grouping_methods[method](df_features, n_groups=n_groups)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 결과 저장
        result = {
            'target': target_var,
            'method': method,
            'n_groups': len(groups),
            'groups': groups,
            'all_features': numeric_cols
        }

        self.selected_groups[f"{target_var}_{method}"] = result

        return result

    def save_grouping_report(self, output_dir: str = "feature_grouping_analysis"):
        """그룹핑 결과를 HTML 리포트로 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for key, result in self.selected_groups.items():
            target = result['target']
            method = result['method']
            groups = result['groups']

            html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{target.upper()} - {method.upper()} 특성 그룹핑 결과</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .group {{
            background: #ecf0f1;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .group h3 {{
            color: #3498db;
            margin-top: 0;
        }}
        .feature {{
            background: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }}
        .method-badge {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 5px 15px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{target.upper()} - 특성 그룹핑 결과</h1>
        <p><span class="method-badge">{method.upper()} 방법</span></p>
        <p><strong>총 {len(groups)}개 그룹</strong></p>
"""

            for i, group in enumerate(groups, 1):
                html_content += f"""
        <div class="group">
            <h3>그룹 {i}</h3>
"""
                for feature in group:
                    html_content += f"""            <div class="feature">📊 {feature}</div>\n"""

                html_content += """        </div>\n"""

            html_content += """
    </div>
</body>
</html>
"""

            # 저장
            report_file = output_path / f"{target}_{method}_grouping.html"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"✅ 리포트 저장: {report_file}")


def main():
    """메인 실행 함수"""
    grouper = IntelligentFeatureGrouping()

    target_vars = ['anxiety', 'depression', 'stress']
    methods = ['pca', 'correlation', 'hybrid']

    print(f"\n{'='*60}")
    print(f"🚀 지능형 특성 그룹핑 시작")
    print(f"{'='*60}\n")

    # 각 타겟과 방법에 대해 그룹핑 수행
    for target in target_vars:
        for method in methods:
            try:
                grouper.select_groups_for_target('data_splits', target, method=method)
            except Exception as e:
                print(f"❌ {target} - {method} 실패: {e}")

    # 리포트 저장
    grouper.save_grouping_report()

    print(f"\n{'='*60}")
    print(f"✅ 지능형 특성 그룹핑 완료")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
