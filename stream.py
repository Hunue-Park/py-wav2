# file_asr.py
import torch
import numpy as np
import os
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from data_processing import load_pcm_file, load_audio_file

def recognize_file(model_path, audio_file, device=None):
    """오디오 파일 인식 함수"""
    # 장치 설정
    if device is None:
        device = torch.device('cpu')  # macOS에서는 CPU 사용 권장
    else:
        device = torch.device(device)
            
    print(f"사용 장치: {device}")
    
    # 모델 및 프로세서 로드
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    model.eval()
    
    # 오디오 파일 로드
    if audio_file.endswith('.pcm'):
        audio = load_pcm_file(audio_file)
    else:
        audio = load_audio_file(audio_file)
    
    # 모델 입력 형식으로 변환
    inputs = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values.to(device)
    
    # 인식 수행
    with torch.no_grad():
        logits = model(inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        # [unk] 토큰 제거 및 공백 정규화
        transcription = transcription.replace("[unk]", "")
        transcription = " ".join(transcription.split())
    
    return transcription

def main():
    parser = argparse.ArgumentParser(description='오디오 파일 한국어 음성 인식')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='파인튜닝된 모델 경로')
    parser.add_argument('--audio_file', type=str, required=True,
                        help='인식할 오디오 파일 경로 (.wav 또는 .pcm)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='사용할 장치 (cpu 또는 cuda, 기본값은 cpu)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"오류: 파일 '{args.audio_file}'을 찾을 수 없습니다.")
        return
    
    # 음성 인식 실행
    transcription = recognize_file(
        model_path=args.model_path, 
        audio_file=args.audio_file, 
        device=args.device
    )
    
    print(f"\n인식 결과: {transcription}")

if __name__ == "__main__":
    main()