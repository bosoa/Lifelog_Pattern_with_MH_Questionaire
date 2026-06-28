"""
생존 분석 리포트에서 C-index 추출
"""
import re
from pathlib import Path
import pandas as pd


def extract_cindex_from_html(html_file: Path) -> float:
    """HTML 리포트에서 C-index 추출"""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # C-index 패턴 찾기
        patterns = [
            r'<h4>Concordance Index</h4>\s*<p>([0-9.]+)</p>',
            r'Concordance Index:\s*([0-9.]+)',
            r'C-index:\s*([0-9.]+)',
            r'✓ Concordance Index:\s*([0-9.]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return float(match.group(1))

        return None
    except Exception as e:
        print(f"오류: {html_file.name} - {e}")
        return None


def main():
    """메인 실행 함수"""
    results = []

    # 원본 데이터
    original_dir = Path("model_results")
    # 극좌표 데이터
    polar_dir = Path("model_results_polar")

    for target in ['anxiety', 'depression', 'stress']:
        # 원본
        original_file = original_dir / f"{target}_survival_analysis_report.html"
        original_cindex = None
        if original_file.exists():
            original_cindex = extract_cindex_from_html(original_file)

        # 극좌표
        polar_file = polar_dir / f"{target}_survival_analysis_polar_report.html"
        polar_cindex = None
        if polar_file.exists():
            polar_cindex = extract_cindex_from_html(polar_file)

        results.append({
            'target': target,
            'original_cindex': original_cindex,
            'polar_cindex': polar_cindex,
            'difference': polar_cindex - original_cindex if (original_cindex and polar_cindex) else None
        })

    # 결과 출력
    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("생존 분석 C-index 비교 (원본 vs 극좌표)")
    print("="*70 + "\n")
    print(df.to_string(index=False))

    # 저장
    df.to_csv("model_results/cindex_comparison.csv", index=False)
    print(f"\n✅ 결과 저장: model_results/cindex_comparison.csv")


if __name__ == "__main__":
    main()
