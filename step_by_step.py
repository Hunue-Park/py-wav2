# main.py - 모델 의존성 최소화 버전
import argparse
import json
import numpy as np
import torch
import os
import logging
from g2pk import G2p  # 한국어 g2p 라이브러리
from dtw import dtw

from data_processing import load_audio_file, load_transcript
from w2v_engine import Wav2VecCTCEngine
from transformers import Wav2Vec2CTCTokenizer
import pprint

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    print("\n===== 한국어 발음 평가 시스템 (기본형) =====\n")

    tokenizer = Wav2Vec2CTCTokenizer(
                    "./env/fine-tuned-wav2vec2-kspon/vocab.json",
                    unk_token="[UNK]",
                    pad_token="[PAD]",
                    word_delimiter_token=" "
                )  # :contentReference[oaicite:4]{index=4}
    
    # 1. 오디오 파일과 정답 텍스트 로드
    try:
        print("1. 오디오 및 텍스트 파일 로드 중...")
        audio = load_audio_file(args.audio_file)
        transcript = load_transcript(args.transcript_file)
        print(f"   - 오디오 파일: {args.audio_file} ({len(audio)/16000:.2f}초)")
        print(f"   - 정답 텍스트: '{transcript}'")
    except Exception as e:
        logger.error(f"파일 로드 실패: {e}")
        return
    
    # 2. 한국어 발음 변환 (G2P) - 음절 단위 유지
    try:
        print("\n2. 정답 텍스트 발음 변환 중...")
        syllable_tokens = tokens = tokenizer.tokenize(transcript)
        
        print(f"   - 음절 시퀀스: {syllable_tokens} ({len(syllable_tokens)}개 음절)")
    except Exception as e:
        logger.error(f"G2P 변환 실패: {e}")
        return
    
    # 3. 모델 로드 및 확률 분포 획득
    try:
        print("\n3. 음성 인식 모델 로드 및 확률 분포 계산 중...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   - 사용 장치: {device}")
        
        w2v_engine = Wav2VecCTCEngine.get_instance(model_name=args.model_name, device=device)
        
    except Exception as e:
        logger.error(f"모델 예측 실패: {e}")
        print(f"오류 세부 정보: {str(e)}")
        return
    
    # 4. 음절별 프레임 정렬 (진짜 DTW 방식)
    try:
        print("\n4. 음절별 프레임 정렬 중...")
        
        
        
    except Exception as e:
        logger.error(f"DTW 음절 정렬 실패: {e}")
        print(f"오류 세부 정보: {str(e)}")
        return
    
    # 5. GOP 계산 (음절 단위)
    print("\n5. 음절별 발음 점수 계산 중...")
    gop_scores = w2v_engine.calculate_gop(args.audio_file, transcript)
    
    
    
    
    # 6. 최종 결과 계산 및 출력
    print("\n===== 발음 평가 최종 결과 =====")
    result = w2v_engine.format_gop_as_response(gop_scores)
    pprint.pprint(result)
    
    # 전체 평균 점수
    
    
    # 결과 저장
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="한국어 발음 평가 시스템 (기본형)")
    parser.add_argument('--audio_file', type=str, required=True, help='입력 오디오 파일 경로 (.wav)')
    parser.add_argument('--transcript_file', type=str, required=True, help='정답 텍스트 파일 경로 (.txt)')
    parser.add_argument('--model_name', type=str, default='kresnik/wav2vec2-large-xlsr-korean', 
                        help='사용할 wav2vec2 모델 이름')
    parser.add_argument('--output_file', type=str, help='결과 저장 파일 경로 (.json)')
    args = parser.parse_args()
    
    main(args)
