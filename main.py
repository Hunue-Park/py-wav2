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
from model import Wav2VecCTC
from gop_calculation import normalize_gop_score, get_pronunciation_grade
from transformers import Wav2Vec2CTCTokenizer

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    print("\n===== 한국어 발음 평가 시스템 (기본형) =====\n")
    
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
        # g2p = G2p()  # 한국어 G2P 초기화
        # standard_pronunciation = g2p(transcript)
        
        # 음절 단위로 분리 (공백 제거)
        syllables = transcript.replace(" ", "")
        syllable_tokens = list(syllables)  # '안녕하세요' → ['안', '녕', '하', '세', '요']
        
        # print(f"   - 표준 발음: {standard_pronunciation}")
        print(f"   - 음절 시퀀스: {syllable_tokens} ({len(syllable_tokens)}개 음절)")
    except Exception as e:
        logger.error(f"G2P 변환 실패: {e}")
        return
    
    # 3. 모델 로드 및 확률 분포 획득
    try:
        print("\n3. 음성 인식 모델 로드 및 확률 분포 계산 중...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   - 사용 장치: {device}")
        
        model = Wav2VecCTC.get_instance(model_name=args.model_name, device=device)
        model_vocab = model.get_vocab()

        tokenizer = Wav2Vec2CTCTokenizer(
            "./env/fine-tuned-wav2vec2-kspon/vocab.json",
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token=" "
        )  # :contentReference[oaicite:4]{index=4}

        tokens = tokenizer.tokenize(transcript)
        token_index = tokenizer.convert_tokens_to_ids(tokens)
        print(token_index, 'token_index !!!!!!')
        
        # 인덱스→토큰 매핑 생성
        idx_to_token = {}
        for token, idx in model_vocab.items():
            idx_to_token[idx] = token
        
        # 토큰→인덱스 매핑 생성
        token_to_idx = {v: k for k, v in idx_to_token.items()}
        
        # 예측 수행 - 로짓과 확률 분포만 필요
        logits, probs = model.predict(audio)
        probs_matrix = probs.squeeze(0).cpu()
        
        print(f"   - 모델 예측 완료: 프레임 수 {probs_matrix.shape[0]}, 토큰 수 {probs_matrix.shape[1]}")
        
        # 예측 텍스트 확인 (참조용)
        predicted_text = model.processor.batch_decode(logits.argmax(dim=-1))[0]
        print(f"   - 모델이 인식한 텍스트: {predicted_text}")
        
        # 한글 음절 토큰 수 확인
        hangul_tokens = [t for t in token_to_idx.keys() if any('\uAC00' <= c <= '\uD7A3' for c in t)]
        print(f"   - 한글 음절 토큰 수: {len(hangul_tokens)}")
        
    except Exception as e:
        logger.error(f"모델 예측 실패: {e}")
        print(f"오류 세부 정보: {str(e)}")
        return
    
    # 4. 음절별 프레임 정렬 (진짜 DTW 방식)
    try:
        print("\n4. 음절별 프레임 정렬 중...")
        
        # 정답 음절 중 모델 vocab에 있는 것만 필터링
        valid_syllables = [s for s in syllable_tokens if s in token_to_idx]
        print(f"   - 유효한 음절: {valid_syllables} ({len(valid_syllables)}/{len(syllable_tokens)}개)")
        
        if not valid_syllables:
            print("경고: 모델 vocabulary에 있는 정답 음절이 없습니다.")
            return
        
        # 음절별 확률 행렬 구성 (프레임 x 음절)
        n_frames = probs_matrix.shape[0]
        
        # 음절 순서대로 해당 음절의 확률값만 뽑아서 행렬 구성
        query_seq = []
        for i, syllable in enumerate(valid_syllables):
            syllable_idx = token_to_idx.get(syllable)
            if syllable_idx is not None:
                prob_seq = probs_matrix[:, syllable_idx].cpu().numpy()
                query_seq.append(prob_seq)
        
        # 목표 행렬: 각 음절별로 1인 단위 벡터 (n_syllables x n_syllables)
        target_seq = np.eye(len(valid_syllables))
        
        # DTW 적용: 음절별 목표 벡터와 프레임별 확률 벡터 사이의 최적 정렬 찾기
        alignment = dtw(
            target_seq,
            np.column_stack(query_seq).T,  # (n_frames x n_syllables) 행렬로 변환
            keep_internals=True,
            step_pattern="symmetric2"
        )
        
        # DTW 결과에서 각 음절의 프레임 범위 추출
        syllable_segments = []
        
        # 음절별로 매핑된 프레임 범위 찾기
        for syl_idx in range(len(valid_syllables)):
            # 해당 음절 인덱스에 매핑된 프레임 인덱스 찾기 (DTW 경로 사용)
            mask = alignment.index1 == syl_idx
            frames = alignment.index2[mask]
            
            if len(frames) > 0:
                start_frame = np.min(frames)
                end_frame = np.max(frames)
                
                # 최소 길이 3 보장
                if end_frame - start_frame + 1 < 3:
                    end_frame = min(n_frames - 1, start_frame + 2)
                
                syllable = valid_syllables[syl_idx]
                syllable_segments.append((syllable, int(start_frame), int(end_frame)))
        
        # 정렬 결과 출력
        print(f"   - DTW 정렬 완료: {len(syllable_segments)}개 음절 세그먼트")
        for i, (syllable, start, end) in enumerate(syllable_segments):
            frame_duration = end - start + 1
            print(f"   - 음절 {i+1}: '{syllable}' (프레임 {start}-{end}, 길이: {frame_duration})")
        
    except Exception as e:
        logger.error(f"DTW 음절 정렬 실패: {e}")
        print(f"오류 세부 정보: {str(e)}")
        return
    
    # 5. GOP 계산 (음절 단위)
    print("\n5. 음절별 발음 점수 계산 중...")
    syllable_scores = []
    syllable_details = []
    
    for syllable, start, end in syllable_segments:
        # 범위 검증
        if start < 0 or end > probs_matrix.shape[0] or start >= end:
            print(f"   - 음절 '{syllable}': 잘못된 프레임 범위 {start}-{end}, 평가 불가")
            continue
        
        # 해당 음절의 인덱스 조회
        syllable_idx = token_to_idx.get(syllable)
        if syllable_idx is None:
            print(f"   - 음절 '{syllable}': 인덱스를 찾을 수 없음, 평가 불가")
            continue
        
        # 해당 구간의 모든 프레임에서 목표 음절 확률 추출
        syllable_probs = []
        for i in range(start, end):
            syllable_probs.append(float(probs_matrix[i, syllable_idx]))
        
        # 상위 30%만 사용 (품질 좋은 부분)
        if len(syllable_probs) >= 3:
            top_n = max(1, len(syllable_probs) // 3)
            top_probs = sorted(syllable_probs, reverse=True)[:top_n]
        else:
            top_probs = syllable_probs
        
        # 평균 확률 계산
        avg_prob = sum(top_probs) / len(top_probs) if top_probs else 0
        
        # GOP 점수 계산 (로그 취함)
        raw_score = float(np.log(avg_prob + 1e-6))
        
        # 정규화된 점수로 변환
        normalized_score = normalize_gop_score(raw_score)
        grade = get_pronunciation_grade(normalized_score)
        
        # 첫 프레임 확률 (참조용)
        first_frame_prob = probs_matrix[start, syllable_idx].item()
        
        print(f"   - 음절 '{syllable}': {normalized_score:.1f}점 ({grade}) [첫 프레임 확률: {first_frame_prob:.4f}]")
        
        syllable_scores.append(normalized_score)
        syllable_details.append({
            'syllable': syllable,
            'raw_score': raw_score,
            'normalized_score': float(normalized_score),
            'grade': grade,
            'avg_probability': float(avg_prob),
            'frames': (start, end)
        })
    
    # 6. 최종 결과 계산 및 출력
    print("\n===== 발음 평가 최종 결과 =====")
    
    # 전체 평균 점수
    if syllable_scores:
        overall_score = np.mean(syllable_scores)
        overall_grade = get_pronunciation_grade(overall_score)
        print(f"전체 평균 점수: {overall_score:.1f}점 ({overall_grade})")
    else:
        print("평가 가능한 음절이 없습니다.")
    
    # 결과 저장
    if args.output_file:
        try:
            result = {
                "standard_pronunciation": standard_pronunciation,
                "syllable_details": syllable_details,
                "overall_score": float(overall_score) if syllable_scores else None,
                "overall_grade": overall_grade if syllable_scores else "평가 불가",
                "transcript": transcript
            }
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n결과가 {args.output_file}에 저장되었습니다.")
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="한국어 발음 평가 시스템 (기본형)")
    parser.add_argument('--audio_file', type=str, required=True, help='입력 오디오 파일 경로 (.wav)')
    parser.add_argument('--transcript_file', type=str, required=True, help='정답 텍스트 파일 경로 (.txt)')
    parser.add_argument('--model_name', type=str, default='kresnik/wav2vec2-large-xlsr-korean', 
                        help='사용할 wav2vec2 모델 이름')
    parser.add_argument('--output_file', type=str, help='결과 저장 파일 경로 (.json)')
    args = parser.parse_args()
    
    main(args)
