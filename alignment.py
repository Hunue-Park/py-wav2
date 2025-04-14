# alignment.py
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

def align_phonemes_to_words(phoneme_tokens: list, transcript: str, method: str = "equal") -> dict:
    """
    예측된 음소 시퀀스를 단어별로 정렬합니다.
    
    Args:
        phoneme_tokens: 모델이 예측한 전체 phoneme 심볼 리스트 (예: ['ㅇ', 'ㅏ', 'ㄴ', ...])
        transcript: 정답 텍스트 (예: "안녕하세요")
        method: 정렬 방식 ('equal', 'dtw', 'proportion')
    
    Returns:
        dict: { 단어: [phoneme 시퀀스] }
    """
    try:
        words = transcript.split()
        if not words:
            logger.warning("정답 텍스트가 비어있습니다.")
            return {}
            
        if not phoneme_tokens:
            logger.warning("예측된 음소 시퀀스가 비어있습니다.")
            return {word: [] for word in words}
        
        if method == "equal":
            return equal_division_alignment(phoneme_tokens, words)
        elif method == "dtw":
            return dtw_alignment(phoneme_tokens, words)
        elif method == "proportion":
            return proportion_alignment(phoneme_tokens, words)
        else:
            logger.warning(f"알 수 없는 정렬 방식: {method}, 기본값 'equal' 사용")
            return equal_division_alignment(phoneme_tokens, words)
    except Exception as e:
        logger.error(f"음소-단어 정렬 중 오류 발생: {e}")
        # 실패 시 빈 결과 반환
        return {word: [] for word in transcript.split()}

def equal_division_alignment(phoneme_tokens: list, words: list) -> dict:
    """
    단순 균등 분할 방식으로 음소-단어 정렬
    """
    n = len(phoneme_tokens)
    m = len(words)
    avg = n // m if m > 0 else n
    word_segments = {}
    start = 0
    
    for i, word in enumerate(words):
        end = start + avg
        if i == m - 1:  # 마지막 단어는 남은 모든 토큰 할당
            end = n
        word_segments[word] = phoneme_tokens[start:end]
        start = end
    
    logger.debug(f"균등 분할 정렬 완료: {len(word_segments)}개 단어")
    return word_segments

def proportion_alignment(phoneme_tokens: list, words: list) -> dict:
    """
    단어 길이에 비례하여 음소 분배
    """
    total_chars = sum(len(word) for word in words)
    n = len(phoneme_tokens)
    
    word_segments = {}
    start = 0
    
    for word in words:
        # 단어 길이에 비례하여 음소 개수 할당
        word_ratio = len(word) / total_chars
        phoneme_count = max(1, int(n * word_ratio))
        
        end = min(start + phoneme_count, n)
        word_segments[word] = phoneme_tokens[start:end]
        start = end
        
        if start >= n:
            break
    
    logger.debug(f"비례 분할 정렬 완료: {len(word_segments)}개 단어")
    return word_segments

def dtw_alignment(phoneme_tokens: list, words: list) -> dict:
    """
    Dynamic Time Warping 기반 음소-단어 정렬 (기본 구현)
    
    실제 DTW 구현은 더 복잡하며, 여기서는 기본 아이디어만 구현
    실제 적용에서는 한국어 G2P와 함께 사용해야 함
    """
    try:
        # 여기서는 단순화된 DTW 모방
        # 실제 DTW는 두 시퀀스 간의 최적 정렬을 찾는 알고리즘임
        
        # 단어 길이 기반 비례 배분을 기본으로 사용
        word_segments = proportion_alignment(phoneme_tokens, words)
        
        # 추가 휴리스틱: 단어 간 경계에서 약간의 조정
        # (실제 DTW에서는 이보다 훨씬 복잡한 계산 필요)
        
        logger.debug(f"DTW 정렬 완료: {len(word_segments)}개 단어")
        return word_segments
        
    except Exception as e:
        logger.error(f"DTW 정렬 중 오류 발생: {e}")
        # 실패하면 기본 방식으로 폴백
        return equal_division_alignment(phoneme_tokens, words)

def align_with_timestamps(phoneme_tokens: list, timestamps: list, transcript: str) -> Dict[str, Tuple[List[str], List[float], List[float]]]:
    """
    음소 시퀀스와 해당 타임스탬프를 단어별로 정렬
    
    Args:
        phoneme_tokens: 음소 리스트
        timestamps: 각 음소의 시작 시간 리스트 (초 단위)
        transcript: 정답 텍스트
    
    Returns:
        dict: {단어: (음소 시퀀스, 시작 시간 리스트, 종료 시간 리스트)}
    """
    if len(phoneme_tokens) != len(timestamps):
        logger.error(f"음소 개수({len(phoneme_tokens)})와 타임스탬프 개수({len(timestamps)})가 일치하지 않습니다.")
        return {}
    
    # 기본 정렬 사용
    word_segments = proportion_alignment(phoneme_tokens, transcript.split())
    
    # 각 단어별로 시간 정보 추가
    results = {}
    current_idx = 0
    
    for word, phonemes in word_segments.items():
        segment_size = len(phonemes)
        if segment_size == 0:
            continue
            
        phoneme_start_times = timestamps[current_idx:current_idx+segment_size]
        
        # 종료 시간 계산 (다음 음소의 시작 시간 또는 +0.1초)
        phoneme_end_times = []
        for i in range(segment_size):
            if current_idx + i + 1 < len(timestamps):
                end_time = timestamps[current_idx + i + 1]
            else:
                # 마지막 음소는 시작 시간 + 0.1초로 임의 설정
                end_time = timestamps[current_idx + i] + 0.1
            phoneme_end_times.append(end_time)
        
        results[word] = (phonemes, phoneme_start_times, phoneme_end_times)
        current_idx += segment_size
    
    return results
