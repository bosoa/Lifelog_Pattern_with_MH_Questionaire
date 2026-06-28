"""
Hugging Face Med-Gemma를 활용한 정신건강 분석

수치형 라이프로그 데이터를 의료 컨텍스트로 변환하고
Med-Gemma 모델로 정신건강 상태를 추론
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("analysis_results/06_medgemma_huggingface")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class LifelogToMedicalText:
    """라이프로그 수치 데이터를 의료 텍스트로 변환"""

    @staticmethod
    def convert_to_medical_context(row, feature_cols):
        """
        개인의 라이프로그 데이터를 의료 컨텍스트 텍스트로 변환

        예시:
        "환자는 평균 심박변이도(HRV) 25ms, 일일 걸음 수 8,500보,
        평균 수면시간 6.5시간을 기록했습니다..."
        """
        # 특징별 의미 매핑
        feature_meanings = {
            'a01_mean': ('심박변이도(HRV)', 'ms'),
            'a02_mean': ('일일 걸음 수', '보'),
            'a03_mean': ('활동량', '단위'),
            'a04_mean': ('깊은 수면', '시간'),
            'a05_mean': ('REM 수면', '시간'),
            'a06_mean': ('산소포화도', '%'),
            'a07_mean': ('화면 사용시간', '시간'),
            'a08_mean': ('평균 심박수', 'bpm'),
            'a09_mean': ('체온', '°C'),
            'a10_mean': ('얕은 수면', '시간'),
            'a11_mean': ('이동 거리', 'km'),
            'a12_mean': ('기상 시간', '시'),
            'a13_mean': ('취침 시간', '시'),
            'a14_mean': ('조도', 'lux'),
            'a15_mean': ('총 수면시간', '시간'),
            'a16_mean': ('피부 온도', '°C'),
            'a17_mean': ('혈당', 'mg/dL'),
            'a18_mean': ('혈압', 'mmHg'),
        }

        text_parts = []
        for col in feature_cols:
            if col in row.index and pd.notna(row[col]):
                if col in feature_meanings:
                    name, unit = feature_meanings[col]
                    value = row[col]
                    text_parts.append(f"{name} {value:.1f}{unit}")

        if not text_parts:
            return "데이터 없음"

        # 의료 컨텍스트 구성
        medical_text = f"""
환자의 생체 신호 데이터:
{', '.join(text_parts[:8])}

활동 패턴:
{', '.join(text_parts[8:])}
"""
        return medical_text.strip()

    @staticmethod
    def create_prompt_for_mental_health(medical_context, target_type='depression'):
        """정신건강 평가를 위한 프롬프트 생성"""

        target_descriptions = {
            'PHQ9_Score': ('우울증(PHQ-9)', '0-27점 범위, 높을수록 우울 증상 심각'),
            'GAD7_Score': ('불안(GAD-7)', '0-21점 범위, 높을수록 불안 증상 심각'),
            'ISI_Score': ('불면증(ISI)', '0-28점 범위, 높을수록 불면증 심각')
        }

        target_name, description = target_descriptions.get(
            target_type, ('정신건강', '점수가 높을수록 증상 심각')
        )

        prompt = f"""당신은 정신건강 전문의입니다. 다음 환자의 생체 신호 데이터를 바탕으로 {target_name} 상태를 평가해주세요.

{medical_context}

질문: 위 생체 신호 데이터를 바탕으로, 이 환자의 {target_name} 점수는 얼마일 것으로 추정됩니까?
({description})

평가 근거:
1. 수면 패턴의 질과 양
2. 일주기 리듬의 규칙성
3. 신체 활동 수준
4. 심혈관 지표
5. 전반적인 생활 패턴

답변 형식:
점수: [예상 점수]
근거: [간단한 설명]
"""
        return prompt


def load_medgemma_model():
    """Hugging Face에서 Med-Gemma 또는 의료 특화 모델 로드"""

    print("모델 로드 중...")

    # Med-Gemma 또는 대체 모델 시도
    models_to_try = [
        "google/gemma-2b",  # Gemma 2B (경량)
        "google/gemma-7b",  # Gemma 7B
        "microsoft/BioGPT",  # 생물의학 GPT
        # "google/med-gemma",  # Med-Gemma (공개 시)
    ]

    model = None
    tokenizer = None
    model_name = None

    for model_id in models_to_try:
        try:
            print(f"\n시도: {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            model_name = model_id
            print(f"✅ 모델 로드 성공: {model_id}")
            break
        except Exception as e:
            print(f"❌ {model_id} 로드 실패: {str(e)[:100]}")
            continue

    if model is None:
        print("\n⚠️  Hugging Face 모델 로드 실패")
        print("대안: 로컬 딥러닝 모델 사용")
        return None, None, None

    return model, tokenizer, model_name


def analyze_with_medgemma(model, tokenizer, prompt, max_length=512):
    """Med-Gemma로 분석"""
    if model is None or tokenizer is None:
        return None

    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 토크나이즈
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 프롬프트 제거하고 답변만 추출
    if prompt in response:
        response = response.replace(prompt, "").strip()

    return response


def extract_score_from_response(response):
    """모델 응답에서 점수 추출"""
    if response is None:
        return None

    # "점수: X" 형태 추출
    import re
    score_patterns = [
        r'점수[:\s]*(\d+\.?\d*)',
        r'score[:\s]*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*점',
        r'예상.*?(\d+\.?\d*)',
    ]

    for pattern in score_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except:
                continue

    return None


def main():
    """메인 실행"""
    print("="*80)
    print("Hugging Face Med-Gemma 정신건강 분석")
    print("="*80)

    # 데이터 로드
    print("\n데이터 로드...")
    df = pd.read_csv("analysis_results/04_lifelog_integration/integrated_data.csv")

    feature_cols = [col for col in df.columns if '_mean' in col and col.startswith('a')]
    target_vars = ['PHQ9_Score', 'GAD7_Score', 'ISI_Score']

    print(f"총 데이터: {len(df)}명")
    print(f"특징: {len(feature_cols)}개")

    # 모델 로드
    model, tokenizer, model_name = load_medgemma_model()

    if model is None:
        print("\n❌ Hugging Face 모델을 사용할 수 없습니다.")
        print("대안: 로컬 딥러닝 모델 사용 (medgemma_inspired_model.py)")
        return

    # 샘플 분석 (시간 절약을 위해 일부만)
    n_samples = min(50, len(df))
    print(f"\n샘플 분석: {n_samples}명")

    converter = LifelogToMedicalText()
    results = {}

    for target in target_vars:
        print(f"\n{'='*80}")
        print(f"{target} 분석")
        print(f"{'='*80}")

        # 유효 데이터
        df_valid = df[feature_cols + [target]].dropna().head(n_samples)

        if len(df_valid) == 0:
            print("유효 데이터 없음")
            continue

        predictions = []
        actual = []

        for idx, row in df_valid.iterrows():
            # 의료 컨텍스트 생성
            medical_context = converter.convert_to_medical_context(row, feature_cols)

            # 프롬프트 생성
            prompt = converter.create_prompt_for_mental_health(medical_context, target)

            # 모델 추론
            response = analyze_with_medgemma(model, tokenizer, prompt)

            # 점수 추출
            predicted_score = extract_score_from_response(response)

            if predicted_score is not None:
                predictions.append(predicted_score)
                actual.append(row[target])

                if len(predictions) <= 3:  # 처음 3개만 출력
                    print(f"\n샘플 {len(predictions)}:")
                    print(f"  실제: {row[target]:.1f}")
                    print(f"  예측: {predicted_score:.1f}")
                    print(f"  응답: {response[:200]}...")

        if len(predictions) > 0:
            predictions = np.array(predictions)
            actual = np.array(actual)

            mae = np.mean(np.abs(predictions - actual))
            rmse = np.sqrt(np.mean((predictions - actual) ** 2))
            r2 = 1 - (np.sum((actual - predictions) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

            print(f"\n{target} 성능:")
            print(f"  MAE: {mae:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  R²: {r2:.3f}")

            results[target] = {
                'predictions': predictions,
                'actual': actual,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }

    # 결과 저장
    if results:
        summary = pd.DataFrame([
            {
                '척도': target,
                'N': len(res['actual']),
                'MAE': f"{res['mae']:.3f}",
                'RMSE': f"{res['rmse']:.3f}",
                'R²': f"{res['r2']:.3f}"
            }
            for target, res in results.items()
        ])

        print(f"\n\n{'='*80}")
        print("전체 결과 요약")
        print(f"{'='*80}\n")
        print(summary.to_string(index=False))

        summary.to_csv(OUTPUT_DIR / "medgemma_results.csv", index=False)
        print(f"\n✅ 결과 저장: {OUTPUT_DIR / 'medgemma_results.csv'}")

        # 모델 정보 저장
        with open(OUTPUT_DIR / "model_info.txt", 'w', encoding='utf-8') as f:
            f.write(f"사용 모델: {model_name}\n")
            f.write(f"분석 샘플: {n_samples}명\n")
            f.write(f"특징 수: {len(feature_cols)}개\n")

    print(f"\n{'='*80}")
    print("Hugging Face Med-Gemma 분석 완료!")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    results = main()
