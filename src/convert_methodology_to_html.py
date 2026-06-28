"""
방법론 마크다운 문서를 HTML로 변환
"""
import markdown
from pathlib import Path


def convert_md_to_html(md_file: str, output_dir: str = "reports_for_paper"):
    """마크다운 파일을 HTML로 변환"""

    # 마크다운 파일 읽기
    md_path = Path(md_file)
    if not md_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {md_file}")
        return None

    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 마크다운을 HTML로 변환
    html_body = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite']
    )

    # 전체 HTML 문서 생성
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{md_path.stem}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.8;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f5f5f5;
            color: #333;
        }}

        .container {{
            background: white;
            padding: 60px;
            border-radius: 10px;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}

        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
            font-size: 1.8em;
        }}

        h3 {{
            color: #3498db;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.4em;
        }}

        h4 {{
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.2em;
        }}

        p {{
            margin: 15px 0;
            line-height: 1.8;
        }}

        ul, ol {{
            margin: 15px 0;
            padding-left: 30px;
        }}

        li {{
            margin: 8px 0;
            line-height: 1.7;
        }}

        code {{
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', Courier, monospace;
            color: #e74c3c;
            font-size: 0.9em;
        }}

        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 20px 0;
            line-height: 1.5;
        }}

        pre code {{
            background: none;
            color: #ecf0f1;
            padding: 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        th, td {{
            padding: 15px;
            text-align: left;
            border: 1px solid #ddd;
        }}

        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
        }}

        tr:nth-child(even) {{
            background: #f8f9fa;
        }}

        tr:hover {{
            background: #e8f4f8;
        }}

        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin: 20px 0;
            color: #555;
            font-style: italic;
            background: #f8f9fa;
            padding: 15px 20px;
            border-radius: 4px;
        }}

        strong {{
            color: #2c3e50;
            font-weight: 600;
        }}

        a {{
            color: #3498db;
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: border-bottom 0.3s;
        }}

        a:hover {{
            border-bottom: 1px solid #3498db;
        }}

        .back-btn {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border-radius: 8px;
            text-decoration: none;
            margin-top: 40px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            border-bottom: none !important;
        }}

        .back-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }}

        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 40px 0;
        }}

        .meta-info {{
            background: #ecf0f1;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            font-size: 0.9em;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_body}

        <div style="text-align: center; margin-top: 60px;">
            <a href="../model_results/index.html" class="back-btn">← 대시보드로 돌아가기</a>
        </div>
    </div>
</body>
</html>
"""

    # HTML 파일 저장
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    output_file = output_path / f"{md_path.stem}.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ 변환 완료: {output_file}")
    return str(output_file)


def main():
    """메인 실행 함수"""
    print(f"\n{'='*60}")
    print(f"📄 방법론 문서 HTML 변환")
    print(f"{'='*60}\n")

    # FEATURE_GROUPING_METHODOLOGY.md 변환
    convert_md_to_html("FEATURE_GROUPING_METHODOLOGY.md")

    print(f"\n{'='*60}")
    print(f"✅ 변환 완료")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
