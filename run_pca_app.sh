#!/bin/bash
# PCA 분석 웹 앱 실행 스크립트

echo "🚀 KLOSDOM PCA 분석 웹 앱 시작..."
echo ""
echo "필수 패키지를 설치합니다..."
pip install -r requirements.txt

echo ""
echo "Streamlit 앱을 시작합니다..."
echo "브라우저에서 http://localhost:8501 으로 접속하세요."
echo ""

streamlit run src/app.py
