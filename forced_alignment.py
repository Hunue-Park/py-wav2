# forced_alignment.py
import numpy as np
import torch
from dtw import dtw  # dtw-python 패키지 필요

def align_frames_to_phonemes(probs_matrix, phonemes, reverse_map, method='dtw'):
    """
    음소 시퀀스와 프레임 간의 강제 정렬을 수행합니다.
    
    Args:
        probs_matrix: 모델이 예측한 프레임별 확률 분포 (프레임 수 x 어휘 크기)
        phonemes: 정답 음소 시퀀스 (문자열 리스트)
        reverse_map: 음소→인덱스 매핑 (딕셔너리)
        method: 정렬 방식 ('linear', 'dtw', 'hmm')
        
    Returns:
        List of tuples: [(phoneme, start_frame, end_frame), ...]
    """
    # 알 수 없는 음소 필터링
    phonemes = [p for p in phonemes if p in reverse_map]
    if not phonemes:
        raise ValueError("유효한 음소가 없습니다. reverse_map을 확인하세요.")
    
    n_frames = probs_matrix.shape[0]
    n_phonemes = len(phonemes)
    
    if method == 'linear':
        # 선형 정렬 (균등 분할)
        frames_per_phoneme = n_frames / n_phonemes
        segments = []
        
        for i, phoneme in enumerate(phonemes):
            start = int(i * frames_per_phoneme)
            end = int((i + 1) * frames_per_phoneme) if i < n_phonemes - 1 else n_frames
            segments.append((phoneme, start, end))
            
        return segments
    
    elif method == 'dtw':
        # DTW 기반 정렬
        # 각 음소의 타겟 인덱스 시퀀스 생성
        phoneme_indices = [reverse_map[p] for p in phonemes]
        
        # 각 프레임에서 각 타겟 음소에 대한 확률 추출
        phoneme_probs = np.array([[probs_matrix[f, idx].item() for idx in phoneme_indices] 
                                for f in range(n_frames)])
        
        # DTW로 최적 경로 찾기 (비용 = 1 - 확률)
        alignment = dtw(1 - phoneme_probs, keep_internals=True)
        path = alignment.index2
        
        # 세그먼트 생성
        segments = []
        current_phoneme_idx = 0
        segment_start = 0
        
        for i in range(1, len(path)):
            if path[i] > path[i-1]:  # 다음 음소로 전환
                segments.append((phonemes[current_phoneme_idx], segment_start, i-1))
                current_phoneme_idx = path[i]
                segment_start = i
                
        # 마지막 세그먼트 추가
        segments.append((phonemes[current_phoneme_idx], segment_start, n_frames-1))
        
        return segments
    
    elif method == 'hmm':
        # HMM 기반 정렬 (간소화된 구현)
        # 실제로는 더 복잡한 HMM 정렬 알고리즘 사용 필요
        # 예시로 간단한 비터비 알고리즘 기반 구현
        # ...
        
        # 임시 대체: 선형 정렬 사용
        print("HMM 정렬은 현재 구현되지 않았습니다. 선형 정렬로 대체합니다.")
        return align_frames_to_phonemes(probs_matrix, phonemes, reverse_map, method='linear')
    
    else:
        raise ValueError(f"지원하지 않는 정렬 방식: {method}")