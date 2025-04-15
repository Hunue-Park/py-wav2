# evaluation.py (CER 전용 버전 - 수정)
import os
import torch
import numpy as np
from tqdm import tqdm
import Levenshtein  # pip install python-Levenshtein
from model import Wav2VecCTC
from data_processing import load_audio_file, load_pcm_file
import Levenshtein
import re

def preprocess_ksponspeech_text(text):
    """KsponSpeech 텍스트 전처리 (수정)"""
    # 원본 텍스트 출력
    # print(f"원본 텍스트: '{text}'")
    
    # 1. 단독 잡음 기호 (단어 시작에 나오는 'o', 'b' 등)
    text = re.sub(r'^\s*[onblu+*]\s+', '', text)  # 문장 시작
    text = re.sub(r'\s+[onblu+*]\s+', ' ', text)  # 문장 중간
    
    # 2. 슬래시 포함 잡음 기호 패턴
    text = re.sub(r'[onblu+*]/|\s/[onbluitmaksg+*]\s', '', text)
    
    # 3. 괄호 표현 제거
    text = re.sub(r'\([^)]*\)/\([^)]*\)', '', text)  # (A)/(B) 패턴
    text = re.sub(r'\([^)]*\)', '', text)  # 일반 괄호
    
    # 4. 구두점 제거
    text = re.sub(r'[,.?!;:\'"]', '', text)
    
    # 5. 공백 정규화
    text = ' '.join(text.split())
    
    # print(f"전처리 후: '{text}'")
    return text

def calculate_cer(predictions, references):
    """
    문자 오류율(CER) 계산 - Levenshtein 거리 사용
    
    Args:
        predictions: 예측 텍스트 리스트
        references: 정답 텍스트 리스트
    
    Returns:
        float: 전체 CER
    """
    # 예측 텍스트와 정답 텍스트 리스트 길이 검증
    if len(predictions) != len(references):
        print("오류: 예측 텍스트와 정답 텍스트 리스트의 길이가 일치하지 않습니다.")
        raise ValueError("리스트의 길이가 일치하지 않습니다.")
    
    total_edits = 0
    total_chars = 0
    
    for idx, (ref, hyp) in enumerate(zip(references, predictions)):
        # 공백 제거 (문자 단위 평가를 위해)
        ref_no_space = ref.replace(" ", "")
        hyp_no_space = hyp.replace(" ", "")
        
        # Levenshtein 거리 계산
        edits = Levenshtein.distance(hyp_no_space, ref_no_space)
        
        # 누적
        total_edits += edits
        total_chars += len(ref_no_space)
    
    # 전체 CER 계산
    if total_chars == 0:
        print("경고: 참조 텍스트에 문자가 없습니다. 최대 오류율(1.0)을 반환합니다.")
        return 1.0  # 빈 참조 텍스트의 경우 최대 오류율 반환
    
    cer = total_edits / total_chars
    # print(f"총 편집 횟수: {total_edits}, 총 문자 수: {total_chars}, CER: {cer:.4f}")
    return cer



def evaluate_model(model_name, test_data_path, result_file=None):
    """모델 성능 평가 (CER만 계산)"""
    # 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Wav2VecCTC.get_instance(model_name=model_name, device=device)
    
    # 결과 저장용
    all_predictions = []
    all_references = []
    individual_results = []
    
    # 테스트 데이터 디렉토리 내 모든 오디오 파일 처리
    audio_files = [f for f in os.listdir(test_data_path) if f.endswith('.wav') or f.endswith('.pcm')]
    
    for audio_file in tqdm(audio_files, desc="평가 중"):
        # PCM 및 WAV 모두 처리 가능하도록 설정
        audio_path = os.path.join(test_data_path, audio_file)
        text_path = os.path.join(test_data_path, audio_file.replace('.pcm', '.txt').replace('.wav', '.txt'))
        
        if not os.path.exists(text_path):
            print(f"경고: {text_path} 파일이 없습니다. 건너뜁니다.")
            continue
            
        try:
            # PCM 또는 WAV 파일 로드
            if audio_file.endswith('.pcm'):
                audio = load_pcm_file(audio_path)
            else:
                audio = load_audio_file(audio_path)
                
            with open(text_path, 'r', encoding='utf-8') as f:
                reference = f.read().strip()
                
            # 예측 수행
            logits, _ = model.predict(audio)
            prediction = model.processor.batch_decode(logits.argmax(dim=-1))[0]
            
            # 텍스트 전처리 (공백 정규화)
            prediction = prediction.strip()
            reference = preprocess_ksponspeech_text(reference)

            # [unk] 토큰 제거 (결과 표시용)
            cleaned_prediction = prediction.replace("[unk]", "")
            cleaned_prediction = " ".join(cleaned_prediction.split())
            
            # 개별 파일 CER 계산
            file_cer = calculate_cer([cleaned_prediction], [reference])
            
            individual_results.append({
                'audio_file': audio_file,
                'reference': reference,
                'prediction': cleaned_prediction,
                'cer': file_cer
            })
            
            all_predictions.append(cleaned_prediction)
            all_references.append(reference)
            
        except Exception as e:
            print(f"오류: {audio_file} 평가 중 문제 발생 - {e}")
    
    # 전체 CER 계산
    overall_cer = calculate_cer(all_predictions, all_references)
    
    # 결과 출력
    print(f"\n===== 평가 결과: {model_name} =====")
    print(f"테스트 샘플 수: {len(all_references)}")
    print(f"전체 CER: {overall_cer:.4f} (낮을수록 우수)")
    
    # 개별 결과 분석
    cer_values = [result['cer'] for result in individual_results]
    if cer_values:
        print(f"최소 CER: {min(cer_values):.4f}")
        print(f"최대 CER: {max(cer_values):.4f}")
        print(f"평균 CER: {np.mean(cer_values):.4f}")
    
    # 결과 저장
    if result_file:
        import json
        result = {
            'model_name': model_name,
            'overall_cer': overall_cer,
            'sample_count': len(all_references),
            'individual_results': individual_results
        }
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"평가 결과가 {result_file}에 저장되었습니다.")
    
    return overall_cer, individual_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="한국어 음성 인식 모델 CER 평가")
    parser.add_argument('--model_name', type=str, default='kresnik/wav2vec2-large-xlsr-korean', 
                        help='평가할 모델 이름')
    parser.add_argument('--test_data', type=str, required=True, 
                        help='테스트 데이터 디렉토리 경로 (wav/pcm 및 txt 파일)')
    parser.add_argument('--result_file', type=str, 
                        help='평가 결과 저장 파일 경로 (JSON)')
    
    args = parser.parse_args()
    evaluate_model(args.model_name, args.test_data, args.result_file)