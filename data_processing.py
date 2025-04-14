# data_processing.py
import librosa
import numpy as np
import os

def load_audio_file(file_path: str, sr: int = 16000):
    """
    주어진 경로의 오디오 파일을 librosa로 불러오며,
    sr에 맞게 리샘플링합니다.
    """
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def load_transcript(file_path: str):
    """
    UTF-8로 인코딩된 텍스트 파일을 읽어 정답 문장을 반환합니다.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        transcript = f.read().strip()
    return transcript

def load_pcm_file(file_path, sample_rate=16000):
    """KsponSpeech PCM 파일 로드 - 특수 포맷 지원"""
    try:
        # 파일 크기 확인
        file_size = os.path.getsize(file_path)
        
        # 2의 배수로 정렬
        aligned_size = file_size - (file_size % 2)
        
        with open(file_path, 'rb') as f:
            # 정렬된 크기만큼만 읽기
            pcm_data = f.read(aligned_size)
        
        # 16비트 PCM 데이터를 NumPy 배열로 변환
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
        
        # 정규화 (-1 ~ 1 범위)
        audio = audio / 32768.0
        
        return audio
    
    except Exception as e:
        raise IOError(f"PCM 파일 로드 실패: {str(e)}")
