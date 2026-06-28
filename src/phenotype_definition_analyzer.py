"""
정신건강 피노타입 정의 및 분석 모듈
표준화된 좌표변환 공간에서 피노타입을 정의하고 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class PhenotypeDefinitionAnalyzer:
    """정신건강 피노타입 정의 및 분석 클래스"""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 피노타입 정의
        self.phenotype_definitions = {
            'anxiety_acute': {
                'name': '급성 불안형',
                'description': '높은 심박수와 피부온도, 낮은 HRV를 특징으로 하는 급성 생리적 각성 상태',
                'criteria': {
                    'heart_beat': 'high',  # 절대값 중요
                    'skin_temperature': 'high',
                    'hrv': 'low',
                    'coordinate_pattern': 'polar_high_radius'
                }
            },
            'depression_low_activity': {
                'name': '저활동 우울형',
                'description': '낮은 활동량(걸음수, 센서활동)과 수면 장애를 특징으로 하는 우울 상태',
                'criteria': {
                    'walk': 'low',
                    'stick_sensor': 'low',
                    'total_sleep': 'disrupted',
                    'coordinate_pattern': 'standardized_polar_low_radius'
                }
            },
            'stress_multidimensional': {
                'name': '다차원 스트레스형',
                'description': '수면, 심혈관, 대사 지표가 복합적으로 변화하는 스트레스 상태',
                'criteria': {
                    'total_sleep': 'low',
                    'heart_beat': 'elevated',
                    'blood_sugar': 'fluctuating',
                    'coordinate_pattern': 'standardized_spherical_complex'
                }
            },
            'mixed_anxiety_depression': {
                'name': '혼합 불안-우울형',
                'description': '불안과 우울 증상이 동시에 나타나는 복합 상태',
                'criteria': {
                    'heart_beat': 'elevated',
                    'walk': 'low',
                    'rem_sleep': 'disrupted',
                    'coordinate_pattern': 'cylindrical_mixed'
                }
            },
            'resilient_normal': {
                'name': '정상 회복탄력형',
                'description': '균형 잡힌 생리적 지표를 보이는 정상 상태',
                'criteria': {
                    'all_features': 'balanced',
                    'coordinate_pattern': 'centered_moderate'
                }
            }
        }

    def load_standardized_data(self, target: str, coordinate_system: str, split: str = 'test'):
        """표준화된 좌표변환 데이터 로드"""
        # 디렉토리 이름 매핑 (spherical->sphere, cylindrical->cylinder)
        coord_dir_map = {
            'spherical': 'sphere',
            'cylindrical': 'cylinder',
            'polar': 'polar'
        }

        # 파일 이름 매핑 (디렉토리와 파일명이 다름)
        coord_file_map = {
            'spherical': 'spherical',
            'cylindrical': 'cylindrical',
            'polar': 'polar'
        }

        coord_dir = coord_dir_map.get(coordinate_system, coordinate_system)
        coord_file = coord_file_map.get(coordinate_system, coordinate_system)

        filename = f"{target}_{split}_standardized_{coord_file}.csv"
        filepath = self.data_dir / f"data_splits_standardized_{coord_dir}" / filename

        if not filepath.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

        df = pd.read_csv(filepath)

        # 이벤트 컬럼명 통일 (target_binary -> event)
        binary_col = f"{target}_binary"
        if binary_col in df.columns:
            df['event'] = df[binary_col]

        print(f"✓ {target} ({coordinate_system}, {split}) 데이터 로드 완료: {len(df)} 샘플")
        return df

    def assign_phenotype(self, row, target, coordinate_system, stats):
        """개별 샘플에 피노타입 할당 (전체 데이터 통계 기반)"""

        # 이벤트 발생 여부
        event_occurred = row['event'] == 1

        if not event_occurred:
            return 'resilient_normal'

        # 좌표값 기반 피노타입 할당
        if coordinate_system == 'polar':
            if target == 'anxiety':
                # 극좌표에서 높은 반지름 = 절대값이 큰 특징들
                r_cols = [c for c in row.index if c.startswith('r_')]
                if r_cols:
                    r_mean = np.mean([row[c] for c in r_cols])
                    if r_mean > stats['r_mean'] + stats['r_std']:
                        return 'anxiety_acute'

            elif target == 'depression':
                # 표준화된 극좌표에서 낮은 반지름 = 모든 특징이 낮음
                r_cols = [c for c in row.index if c.startswith('r_')]
                if r_cols:
                    r_mean = np.mean([row[c] for c in r_cols])
                    if r_mean < stats['r_mean'] - 0.5 * stats['r_std']:
                        return 'depression_low_activity'

        elif coordinate_system == 'spherical':
            if target == 'stress':
                # 구좌표에서 특정 각도 범위 = 복잡한 패턴
                r_cols = [c for c in row.index if c.startswith('r_')]
                if r_cols:
                    r_mean = np.mean([row[c] for c in r_cols])
                    if r_mean > stats['r_mean']:
                        return 'stress_multidimensional'

        elif coordinate_system == 'cylindrical':
            # 원통좌표에서 혼합 패턴
            rho_cols = [c for c in row.index if c.startswith('rho_')]
            z_cols = [c for c in row.index if c.startswith('z_')]
            if rho_cols and z_cols:
                rho_mean = np.mean([row[c] for c in rho_cols])
                z_mean = np.mean([row[c] for c in z_cols])
                if rho_mean > stats['rho_mean'] and z_mean < stats['z_mean']:
                    return 'mixed_anxiety_depression'

        # 기본값: 이벤트는 발생했지만 특정 패턴에 맞지 않음
        return f'{target}_unspecified'

    def analyze_phenotype_distribution(self, target: str, coordinate_system: str):
        """피노타입 분포 분석"""

        # 데이터 로드
        df = self.load_standardized_data(target, coordinate_system, 'test')

        # 통계 계산
        stats = {}
        r_cols = [c for c in df.columns if c.startswith('r_')]
        rho_cols = [c for c in df.columns if c.startswith('rho_')]
        z_cols = [c for c in df.columns if c.startswith('z_')]

        if r_cols:
            r_values = df[r_cols].values.flatten()
            stats['r_mean'] = np.nanmean(r_values)
            stats['r_std'] = np.nanstd(r_values)

        if rho_cols:
            rho_values = df[rho_cols].values.flatten()
            stats['rho_mean'] = np.nanmean(rho_values)
            stats['rho_std'] = np.nanstd(rho_values)

        if z_cols:
            z_values = df[z_cols].values.flatten()
            stats['z_mean'] = np.nanmean(z_values)
            stats['z_std'] = np.nanstd(z_values)

        # 피노타입 할당
        df['phenotype'] = df.apply(
            lambda row: self.assign_phenotype(row, target, coordinate_system, stats),
            axis=1
        )

        # 분포 계산
        phenotype_counts = df['phenotype'].value_counts()
        phenotype_pct = (phenotype_counts / len(df) * 100).round(2)

        print(f"\n{'='*60}")
        print(f"{target.upper()} - {coordinate_system.upper()} 피노타입 분포")
        print(f"{'='*60}")
        for phenotype, count in phenotype_counts.items():
            pct = phenotype_pct[phenotype]
            print(f"{phenotype:30s}: {count:6d} ({pct:5.2f}%)")

        return df

    def visualize_phenotypes_3d(self, target: str, coordinate_system: str = 'spherical'):
        """3D 좌표공간에서 피노타입 시각화"""

        # 데이터 로드 및 피노타입 할당
        df = self.load_standardized_data(target, coordinate_system, 'test')

        # 통계 계산
        stats = {}
        r_cols = [c for c in df.columns if c.startswith('r_')]
        rho_cols = [c for c in df.columns if c.startswith('rho_')]
        z_cols = [c for c in df.columns if c.startswith('z_')]

        if r_cols:
            r_values = df[r_cols].values.flatten()
            stats['r_mean'] = np.nanmean(r_values)
            stats['r_std'] = np.nanstd(r_values)

        if rho_cols:
            rho_values = df[rho_cols].values.flatten()
            stats['rho_mean'] = np.nanmean(rho_values)
            stats['rho_std'] = np.nanstd(rho_values)

        if z_cols:
            z_values = df[z_cols].values.flatten()
            stats['z_mean'] = np.nanmean(z_values)
            stats['z_std'] = np.nanstd(z_values)

        # 샘플링 (시각화를 위해)
        if len(df) > 5000:
            df = df.sample(5000, random_state=42)

        df['phenotype'] = df.apply(
            lambda row: self.assign_phenotype(row, target, coordinate_system, stats),
            axis=1
        )

        # 첫 번째 좌표 트리플 추출 (실제 컬럼 이름 사용)
        if coordinate_system == 'spherical':
            r_cols = [c for c in df.columns if c.startswith('r_')]
            theta_cols = [c for c in df.columns if c.startswith('theta_')]
            phi_cols = [c for c in df.columns if c.startswith('phi_')]

            x_col = r_cols[0] if r_cols else 'r_0'
            y_col = theta_cols[0] if theta_cols else 'theta_0'
            z_col = phi_cols[0] if phi_cols else 'phi_0'
            x_label, y_label, z_label = 'Radius (r)', 'Azimuthal (θ)', 'Polar (φ)'

        elif coordinate_system == 'cylindrical':
            rho_cols = [c for c in df.columns if c.startswith('rho_')]
            phi_cols = [c for c in df.columns if c.startswith('phi_')]
            z_cols = [c for c in df.columns if c.startswith('z_')]

            x_col = rho_cols[0] if rho_cols else 'rho_0'
            y_col = phi_cols[0] if phi_cols else 'phi_0'
            z_col = z_cols[0] if z_cols else 'z_0'
            x_label, y_label, z_label = 'Radial (ρ)', 'Angular (φ)', 'Height (z)'

        elif coordinate_system == 'polar':
            # 극좌표는 2D이므로 여러 쌍을 3D로 표현
            r_cols = [c for c in df.columns if c.startswith('r_')]
            theta_cols = [c for c in df.columns if c.startswith('theta_')]

            x_col = r_cols[0] if r_cols else 'r_0'
            y_col = theta_cols[0] if theta_cols else 'theta_0'
            z_col = r_cols[1] if len(r_cols) > 1 else r_cols[0]
            x_label, y_label, z_label = 'Radius 1 (r₀)', 'Angle 1 (θ₀)', 'Radius 2 (r₁)'

        # 3D 산점도
        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color='phenotype',
            title=f'{target.upper()} 피노타입 분포 ({coordinate_system.upper()} 좌표계)',
            labels={
                x_col: x_label,
                y_col: y_label,
                z_col: z_label,
                'phenotype': '피노타입'
            },
            opacity=0.6,
            size_max=5
        )

        fig.update_layout(
            width=1000,
            height=800,
            font=dict(size=12),
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label
            )
        )

        # 저장
        output_file = self.output_dir / f"{target}_{coordinate_system}_phenotype_3d.html"
        fig.write_html(str(output_file))
        print(f"✓ 3D 시각화 저장: {output_file}")

        return fig

    def compare_phenotypes_across_coordinates(self, phenotype_type: str):
        """여러 좌표계에서 특정 피노타입 비교"""

        coordinate_systems = ['polar', 'spherical', 'cylindrical']
        targets = ['anxiety', 'depression', 'stress']

        results = []

        for target in targets:
            for coord_sys in coordinate_systems:
                try:
                    df = self.load_standardized_data(target, coord_sys, 'test')

                    # 통계 계산
                    stats = {}
                    r_cols = [c for c in df.columns if c.startswith('r_')]
                    rho_cols = [c for c in df.columns if c.startswith('rho_')]
                    z_cols = [c for c in df.columns if c.startswith('z_')]

                    if r_cols:
                        r_values = df[r_cols].values.flatten()
                        stats['r_mean'] = np.nanmean(r_values)
                        stats['r_std'] = np.nanstd(r_values)

                    if rho_cols:
                        rho_values = df[rho_cols].values.flatten()
                        stats['rho_mean'] = np.nanmean(rho_values)
                        stats['rho_std'] = np.nanstd(rho_values)

                    if z_cols:
                        z_values = df[z_cols].values.flatten()
                        stats['z_mean'] = np.nanmean(z_values)
                        stats['z_std'] = np.nanstd(z_values)

                    df['phenotype'] = df.apply(
                        lambda row: self.assign_phenotype(row, target, coord_sys, stats),
                        axis=1
                    )

                    phenotype_count = (df['phenotype'] == phenotype_type).sum()
                    phenotype_pct = phenotype_count / len(df) * 100

                    results.append({
                        'target': target,
                        'coordinate_system': coord_sys,
                        'phenotype': phenotype_type,
                        'count': phenotype_count,
                        'percentage': phenotype_pct
                    })
                except Exception as e:
                    print(f"⚠ {target}-{coord_sys} 처리 중 오류: {e}")

        results_df = pd.DataFrame(results)

        # 히트맵 시각화
        pivot_df = results_df.pivot(
            index='coordinate_system',
            columns='target',
            values='percentage'
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': '피노타입 비율 (%)'}
        )
        plt.title(f'{phenotype_type} 피노타입 분포 비교')
        plt.xlabel('Target')
        plt.ylabel('좌표계')
        plt.tight_layout()

        output_file = self.output_dir / f"{phenotype_type}_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 비교 히트맵 저장: {output_file}")

        return results_df

    def create_phenotype_separation_visualization(self):
        """피노타입 간 분리도 시각화 (Figure for paper)"""

        # 스트레스 표준화 구좌표에서 가장 좋은 성능
        target = 'stress'
        coord_sys = 'spherical'

        df = self.load_standardized_data(target, coord_sys, 'test')

        # 통계 계산
        stats = {}
        r_cols = [c for c in df.columns if c.startswith('r_')]
        if r_cols:
            r_values = df[r_cols].values.flatten()
            stats['r_mean'] = np.nanmean(r_values)
            stats['r_std'] = np.nanstd(r_values)

        # 샘플링
        if len(df) > 2000:
            df = df.sample(2000, random_state=42)

        df['phenotype'] = df.apply(
            lambda row: self.assign_phenotype(row, target, coord_sys, stats),
            axis=1
        )

        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Radius (r) 분포',
                'Azimuthal Angle (θ) 분포',
                'Polar Angle (φ) 분포',
                '3D 좌표공간에서의 분리'
            ),
            specs=[
                [{'type': 'box'}, {'type': 'box'}],
                [{'type': 'box'}, {'type': 'scatter3d'}]
            ]
        )

        phenotypes = df['phenotype'].unique()
        colors = px.colors.qualitative.Set2[:len(phenotypes)]

        # 컬럼 이름 찾기
        r_cols = [c for c in df.columns if c.startswith('r_')]
        theta_cols = [c for c in df.columns if c.startswith('theta_')]
        phi_cols = [c for c in df.columns if c.startswith('phi_')]

        r_col = r_cols[0] if r_cols else None
        theta_col = theta_cols[0] if theta_cols else None
        phi_col = phi_cols[0] if phi_cols else None

        if not (r_col and theta_col and phi_col):
            print("⚠ 필수 좌표 컬럼을 찾을 수 없습니다")
            return None

        # Radius 분포
        for i, phenotype in enumerate(phenotypes):
            subset = df[df['phenotype'] == phenotype]
            fig.add_trace(
                go.Box(y=subset[r_col], name=phenotype, marker_color=colors[i],
                       showlegend=False),
                row=1, col=1
            )

        # Theta 분포
        for i, phenotype in enumerate(phenotypes):
            subset = df[df['phenotype'] == phenotype]
            fig.add_trace(
                go.Box(y=subset[theta_col], name=phenotype, marker_color=colors[i],
                       showlegend=False),
                row=1, col=2
            )

        # Phi 분포
        for i, phenotype in enumerate(phenotypes):
            subset = df[df['phenotype'] == phenotype]
            fig.add_trace(
                go.Box(y=subset[phi_col], name=phenotype, marker_color=colors[i],
                       showlegend=False),
                row=2, col=1
            )

        # 3D 산점도
        for i, phenotype in enumerate(phenotypes):
            subset = df[df['phenotype'] == phenotype]
            fig.add_trace(
                go.Scatter3d(
                    x=subset[r_col],
                    y=subset[theta_col],
                    z=subset[phi_col],
                    mode='markers',
                    name=phenotype,
                    marker=dict(size=3, color=colors[i], opacity=0.6)
                ),
                row=2, col=2
            )

        fig.update_layout(
            title_text=f"표준화 구좌표계에서 스트레스 피노타입 분리",
            height=1000,
            width=1400,
            showlegend=True
        )

        output_file = self.output_dir / f"phenotype_separation_visualization.html"
        fig.write_html(str(output_file))
        print(f"✓ 피노타입 분리 시각화 저장: {output_file}")

        return fig


def main():
    """메인 실행 함수"""

    # 경로 설정
    base_dir = Path("/Users/jihwanpark/claude/KLOSDOM/Lifelog_Pattern_Data_Generation")
    output_dir = base_dir / "phenotype_analysis_results"

    # 분석기 초기화
    analyzer = PhenotypeDefinitionAnalyzer(
        data_dir=str(base_dir),
        output_dir=str(output_dir)
    )

    print("=" * 70)
    print("정신건강 피노타입 정의 및 분석")
    print("=" * 70)

    # 1. 각 타겟에 대한 피노타입 분포 분석
    print("\n[1단계] 피노타입 분포 분석")
    print("-" * 70)

    for target in ['anxiety', 'depression', 'stress']:
        for coord_sys in ['polar', 'spherical', 'cylindrical']:
            try:
                analyzer.analyze_phenotype_distribution(target, coord_sys)
            except Exception as e:
                print(f"⚠ {target}-{coord_sys} 분석 중 오류: {e}")

    # 2. 3D 시각화
    print("\n[2단계] 3D 피노타입 시각화")
    print("-" * 70)

    for target in ['anxiety', 'depression', 'stress']:
        for coord_sys in ['spherical', 'cylindrical']:
            try:
                analyzer.visualize_phenotypes_3d(target, coord_sys)
            except Exception as e:
                print(f"⚠ {target}-{coord_sys} 시각화 중 오류: {e}")

    # 3. 피노타입 분리도 시각화 (논문용)
    print("\n[3단계] 피노타입 분리도 시각화 (논문용)")
    print("-" * 70)
    analyzer.create_phenotype_separation_visualization()

    print("\n" + "=" * 70)
    print("분석 완료!")
    print(f"결과 저장 위치: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
