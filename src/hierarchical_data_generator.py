"""
계층화 데이터 생성 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from data_loader import KLOSDOMDataLoader
from pca_analyzer import PCAAnalyzer
from pattern_analyzer import PatternAnalyzer


class HierarchicalDataGenerator:
    """PCA 기반 계층화 데이터 생성기"""

    def __init__(self, output_dir: str = "hierarchical_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.loader = KLOSDOMDataLoader()
        self.pca_analyzer = PCAAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()

    def generate_hierarchical_data(
        self,
        target_variable: str = 'anxiety',
        n_pca_components: int = 10,
        n_top_features: int = 10,
        n_levels: int = 3,
        clustering_method: str = 'kmeans',
        min_data_points: int = 10
    ) -> Dict:
        """
        계층화 데이터 생성 전체 파이프라인

        Args:
            target_variable: 종속변수 ('anxiety', 'depression', 'stress')
            n_pca_components: PCA 주성분 개수
            n_top_features: 선택할 주요 변수 개수
            n_levels: 계층 개수
            clustering_method: 클러스터링 방법 ('kmeans', 'quantile')
            min_data_points: 최소 데이터 포인트 수

        Returns:
            생성 결과 딕셔너리
        """
        print(f"\n{'='*60}")
        print(f"계층화 데이터 생성 시작: {target_variable.upper()}")
        print(f"{'='*60}\n")

        # 1. 데이터 로드
        print("1️⃣ 데이터 로드 중...")
        X, y, feature_names = self.loader.prepare_pca_data(
            target_variable=target_variable,
            min_data_points=min_data_points
        )
        print(f"   ✓ 샘플 수: {len(X):,}, 특징 수: {len(feature_names)}")

        # 2. PCA 수행
        print("\n2️⃣ PCA 분석 중...")
        self.pca_analyzer.fit_transform(X, feature_names)
        loadings_df = self.pca_analyzer.get_loadings()
        explained_var, _ = self.pca_analyzer.get_explained_variance()
        print(f"   ✓ PCA 완료 (상위 {n_pca_components}개 주성분)")

        # 3. 주요 변수 선택
        print("\n3️⃣ 주요 변수 선택 중...")

        # 방법 1: PCA 로딩 기반
        top_features_pca = self.pattern_analyzer.select_top_features_from_pca(
            loadings_df,
            n_components=min(3, n_pca_components),
            n_features_per_pc=n_top_features // 3
        )

        # 방법 2: 가중 중요도 기반
        feature_importance = self.pattern_analyzer.calculate_feature_importance(
            loadings_df,
            explained_var[:min(5, n_pca_components)]
        )
        top_features_importance = feature_importance.head(n_top_features)['feature'].tolist()

        # 두 방법 결합
        selected_features = list(set(top_features_pca) | set(top_features_importance))
        selected_features = selected_features[:n_top_features]  # 최대 개수 제한

        print(f"   ✓ 선택된 변수 ({len(selected_features)}개):")
        for i, feat in enumerate(selected_features, 1):
            importance = feature_importance[
                feature_importance['feature'] == feat
            ]['importance'].values[0]
            print(f"      {i}. {feat} (중요도: {importance:.3f})")

        # 4. 선택된 변수로 데이터 필터링
        print("\n4️⃣ 선택된 변수로 데이터 필터링...")
        X_selected = X[selected_features]
        print(f"   ✓ 필터링된 데이터 크기: {X_selected.shape}")

        # 5. 패턴 계층화
        print(f"\n5️⃣ 패턴 계층화 중 ({clustering_method} 방법, {n_levels}개 계층)...")
        labels = self.pattern_analyzer.create_hierarchical_labels(
            X_selected,
            method=clustering_method,
            n_levels=n_levels
        )
        print(f"   ✓ 계층 레이블 생성 완료")

        # 6. 계층별 분석
        print("\n6️⃣ 계층별 패턴 분석 중...")
        pattern_analysis = self.pattern_analyzer.analyze_pattern_by_target(
            X_selected,
            y,
            labels
        )
        print(f"   ✓ 계층별 통계 계산 완료")

        # 결과 출력
        print("\n" + "="*60)
        print("계층별 통계 요약")
        print("="*60)
        for _, row in pattern_analysis.iterrows():
            print(f"\n📊 {row['level_name']} (레벨 {row['level']})")
            print(f"   샘플 수: {row['n_samples']:,} ({row['percentage']:.1f}%)")
            print(f"   {target_variable} 평균: {row['target_mean']:.2f} ± {row['target_std']:.2f}")
            print(f"   {target_variable} 범위: [{row['target_min']:.1f}, {row['target_max']:.1f}]")

        # 7. 계층화 데이터 저장
        print("\n7️⃣ 데이터 저장 중...")
        result_files = self._save_hierarchical_data(
            X_selected,
            y,
            labels,
            pattern_analysis,
            feature_importance,
            selected_features,
            target_variable
        )
        print(f"   ✓ 저장 완료: {len(result_files)}개 파일")

        # 결과 요약
        summary = {
            'target_variable': target_variable,
            'n_samples': len(X),
            'n_features_original': len(feature_names),
            'n_features_selected': len(selected_features),
            'selected_features': selected_features,
            'n_levels': n_levels,
            'clustering_method': clustering_method,
            'pattern_analysis': pattern_analysis,
            'feature_importance': feature_importance,
            'output_files': result_files
        }

        print(f"\n{'='*60}")
        print("✅ 계층화 데이터 생성 완료!")
        print(f"{'='*60}\n")

        return summary

    def _save_hierarchical_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        labels: pd.Series,
        pattern_analysis: pd.DataFrame,
        feature_importance: pd.DataFrame,
        selected_features: List[str],
        target_variable: str
    ) -> Dict[str, str]:
        """계층화 데이터 및 분석 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{target_variable}_{timestamp}"

        output_files = {}

        # 1. 계층화된 전체 데이터
        hierarchical_df = X.copy()
        hierarchical_df[f'{target_variable}_score'] = y.values
        hierarchical_df['level'] = labels.values
        hierarchical_df['level_name'] = labels.apply(
            lambda x: self.pattern_analyzer._get_level_name(x, labels.nunique())
        )

        filepath = self.output_dir / f"{prefix}_hierarchical_data.csv"
        hierarchical_df.to_csv(filepath, index=False)
        output_files['hierarchical_data'] = str(filepath)
        print(f"      - {filepath.name}")

        # 2. 계층별 통계
        filepath = self.output_dir / f"{prefix}_level_statistics.csv"
        pattern_analysis.to_csv(filepath, index=False)
        output_files['level_statistics'] = str(filepath)
        print(f"      - {filepath.name}")

        # 3. 변수 중요도
        filepath = self.output_dir / f"{prefix}_feature_importance.csv"
        feature_importance.to_csv(filepath, index=False)
        output_files['feature_importance'] = str(filepath)
        print(f"      - {filepath.name}")

        # 4. 선택된 변수 목록
        selected_features_df = pd.DataFrame({
            'feature': selected_features,
            'importance': [
                feature_importance[feature_importance['feature'] == f]['importance'].values[0]
                for f in selected_features
            ]
        }).sort_values('importance', ascending=False)

        filepath = self.output_dir / f"{prefix}_selected_features.csv"
        selected_features_df.to_csv(filepath, index=False)
        output_files['selected_features'] = str(filepath)
        print(f"      - {filepath.name}")

        # 5. 계층별 샘플 데이터 (각 계층당 별도 파일)
        for level in sorted(labels.unique()):
            level_name = self.pattern_analyzer._get_level_name(level, labels.nunique())
            level_data = hierarchical_df[hierarchical_df['level'] == level]

            filepath = self.output_dir / f"{prefix}_level_{level}_{level_name.replace(' ', '_')}.csv"
            level_data.to_csv(filepath, index=False)
            output_files[f'level_{level}_data'] = str(filepath)
            print(f"      - {filepath.name} ({len(level_data):,} 샘플)")

        # 6. 요약 리포트 (텍스트)
        report_path = self.output_dir / f"{prefix}_summary_report.txt"
        self._generate_text_report(
            report_path,
            target_variable,
            pattern_analysis,
            feature_importance,
            selected_features
        )
        output_files['summary_report'] = str(report_path)
        print(f"      - {report_path.name}")

        return output_files

    def _generate_text_report(
        self,
        filepath: Path,
        target_variable: str,
        pattern_analysis: pd.DataFrame,
        feature_importance: pd.DataFrame,
        selected_features: List[str]
    ):
        """텍스트 리포트 생성"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"계층화 데이터 생성 리포트 - {target_variable.upper()}\n")
            f.write("="*70 + "\n\n")

            f.write(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"종속변수: {target_variable}\n\n")

            # 선택된 변수
            f.write("-"*70 + "\n")
            f.write("선택된 주요 변수\n")
            f.write("-"*70 + "\n")
            for i, feat in enumerate(selected_features, 1):
                importance = feature_importance[
                    feature_importance['feature'] == feat
                ]['importance'].values[0]
                f.write(f"{i:2d}. {feat:25s} (중요도: {importance:.4f})\n")

            # 계층별 통계
            f.write("\n" + "-"*70 + "\n")
            f.write("계층별 통계\n")
            f.write("-"*70 + "\n\n")

            for _, row in pattern_analysis.iterrows():
                f.write(f"레벨 {row['level']}: {row['level_name']}\n")
                f.write(f"  샘플 수: {row['n_samples']:,} ({row['percentage']:.1f}%)\n")
                f.write(f"  {target_variable} 평균: {row['target_mean']:.3f}\n")
                f.write(f"  {target_variable} 표준편차: {row['target_std']:.3f}\n")
                f.write(f"  {target_variable} 범위: [{row['target_min']:.2f}, {row['target_max']:.2f}]\n")

                # 주요 변수 평균값
                f.write(f"  주요 변수 평균값:\n")
                for feat in selected_features[:5]:  # 상위 5개만
                    mean_val = row[f'{feat}_mean']
                    if not pd.isna(mean_val):
                        f.write(f"    - {feat}: {mean_val:.2f}\n")
                f.write("\n")

            f.write("="*70 + "\n")
            f.write("리포트 종료\n")
            f.write("="*70 + "\n")


def main():
    """실행 예시"""
    generator = HierarchicalDataGenerator(output_dir="hierarchical_data")

    # 세 가지 종속변수 모두에 대해 생성
    targets = ['anxiety', 'depression', 'stress']

    for target in targets:
        try:
            result = generator.generate_hierarchical_data(
                target_variable=target,
                n_pca_components=10,
                n_top_features=10,
                n_levels=3,
                clustering_method='kmeans',
                min_data_points=10
            )
            print(f"\n✅ {target} 완료\n")

        except Exception as e:
            print(f"\n❌ {target} 실패: {str(e)}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
