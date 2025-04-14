# data_processing.py
import librosa

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
