# gop_calculation.py
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_gop(prob_dist, target_index: int, method: str = 'posterior') -> float:
    """
    한 프레임의 확률 분포와 정답 음소 인덱스로 GOP 점수 계산
    
    Args:
        prob_dist: 확률 분포 (크기: vocab_size)
        target_index: 정답 음소 인덱스
        method: GOP 계산 방식 ('ratio', 'log_ratio', 'posterior')
    
    Returns:
        float: 계산된 GOP 점수
    """
    try:
        # 입력 검증 및 변환
        if isinstance(target_index, str):
            logger.warning(f"target_index가 문자열입니다: {target_index}, 오류 반환")
            return -10.0
            
        target_index = int(target_index)  # 정수로 변환
        
        # 텐서 차원 확인
        if not isinstance(prob_dist, torch.Tensor):
            logger.warning(f"prob_dist가 텐서가 아닙니다: {type(prob_dist)}, 변환 시도")
            if isinstance(prob_dist, (list, np.ndarray)):
                prob_dist = torch.tensor(prob_dist, dtype=torch.float32)
            else:
                return -10.0
        
        # 1차원 텐서 확인
        if len(prob_dist.shape) != 1:
            logger.warning(f"prob_dist가 1차원 텐서가 아닙니다: {prob_dist.shape}, 첫 번째 차원 사용")
            if len(prob_dist.shape) > 1:
                prob_dist = prob_dist[0]  # 첫 번째 차원만 사용
            else:
                return -10.0
        
        # 범위 확인
        if target_index < 0 or target_index >= len(prob_dist):
            logger.warning(f"target_index가 범위를 벗어남: {target_index}, 범위: 0~{len(prob_dist)-1}")
            return -10.0
        
        # 확률 분포에서 target 확률 추출
        target_prob = prob_dist[target_index]
        
        # 방법에 따라 다른 GOP 계산
        if method == 'ratio':
            # 기존 비율 방식: target_prob / max(other_probs)
            other_probs = torch.cat([prob_dist[:target_index], prob_dist[target_index+1:]])
            max_other_prob = torch.max(other_probs)
            # 수치적 안정성을 위한 epsilon
            epsilon = 1e-10
            return float(torch.log(target_prob / (max_other_prob + epsilon) + epsilon))
            
        elif method == 'log_ratio':
            # 로그 영역에서 계산 (수치적 안정성 향상)
            log_target_prob = torch.log(target_prob + 1e-10)
            other_probs = torch.cat([prob_dist[:target_index], prob_dist[target_index+1:]])
            log_max_other = torch.log(torch.max(other_probs) + 1e-10)
            return float(log_target_prob - log_max_other)
            
        elif method == 'posterior':
            # 직접 확률 값 사용 (로그만 취함)
            return float(torch.log(target_prob + 1e-6))
            
        else:
            logger.warning(f"알 수 없는 GOP 계산 방식: {method}, 기본값 'posterior' 사용")
            return float(torch.log(target_prob + 1e-6))
            
    except Exception as e:
        logger.error(f"GOP 계산 중 오류 발생: {e}")
        return -10.0  # 오류 시 매우 낮은 점수 반환

def compute_segment_gop_improved(prob_matrix, start, end, target_idx):
    # 범위 보정
    if start >= end or start < 0 or end >= len(prob_matrix):
        return -5.0
        
    # 각 프레임의 목표 음절 확률 추출
    target_probs = []
    for i in range(start, end):
        target_probs.append(float(prob_matrix[i, target_idx]))
    
    # 상위 30% 프레임만 사용 (오디오 품질이 좋은 부분)
    if len(target_probs) >= 3:
        target_probs = sorted(target_probs, reverse=True)[:max(1, len(target_probs)//3)]
    
    # 로그 변환 후 평균
    return np.log(np.mean(target_probs) + 1e-6)

def normalize_gop_score(raw_score: float, min_score: float = -10.0, 
                        max_score: float = 0.0, target_min: float = 0.0, 
                        target_max: float = 100.0) -> float:
    """
    GOP 원시 점수를 정규화된 점수로 변환
    
    Args:
        raw_score: 원시 GOP 점수
        min_score: 예상되는 최소 GOP 점수 (-10.0으로 변경)
        max_score: 예상되는 최대 GOP 점수 (0.0으로 변경)
        target_min: 정규화 결과 최소값
        target_max: 정규화 결과 최대값
    
    Returns:
        float: 정규화된 점수 (target_min ~ target_max 범위)
    """
    # 범위 제한
    raw_score = max(min_score, min(max_score, raw_score))
    
    # 선형 스케일링
    if max_score == min_score:  # 분모가 0이 되는 것 방지
        return target_min
    normalized = ((raw_score - min_score) / (max_score - min_score)) * (target_max - target_min) + target_min
    return float(normalized)

def get_pronunciation_grade(normalized_score: float) -> str:
    """
    정규화된 점수를 등급으로 변환
    
    Args:
        normalized_score: 0-100 사이의 정규화된 점수
    
    Returns:
        str: 발음 등급 ("우수", "양호", "보통", "미흡", "부족" 중 하나)
    """
    if normalized_score >= 90:
        return "우수"
    elif normalized_score >= 75:
        return "양호"
    elif normalized_score >= 60:
        return "보통"
    elif normalized_score >= 40:
        return "미흡"
    else:
        return "부족"
