import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def recognize_speech(audio_file_path, model_name="kresnik/wav2vec2-large-xlsr-korean"):
    """
    wav2vec2 모델을 사용하여 오디오 파일의 음성을 인식합니다.
    
    Args:
        audio_file_path (str): 인식할 오디오 파일 경로
        model_name (str): 사용할 wav2vec2 모델 이름
    
    Returns:
        dict: 인식 결과 (텍스트, 신뢰도 점수 등)
    """
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델과 프로세서 로드
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # 오디오 로드 및 전처리
    audio, sr = librosa.load(audio_file_path, sr=16000)
    
    # pre-emphasis 적용 (선택 사항)
    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # 입력 형식에 맞게 변환
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # 추론 수행
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    
    # 가장 확률이 높은 토큰 시퀀스 찾기
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # 토큰을 텍스트로 변환
    transcription = processor.batch_decode(predicted_ids)
    
    # 각 토큰에 대한 확률값 계산 (선택 사항)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence = torch.mean(torch.max(probs, dim=-1)[0]).item()
    
    return {
        "text": transcription[0],
        "confidence": confidence
    }

def main():
    # 오디오 파일 경로 지정
    audio_file = "./recordings/recorded_audio_1.wav"  # 인식할 오디오 파일 경로
    
    # 모델 이름 지정 (필요시 변경)
    model_name = "./env/fine-tuned-wav2vec2-kspon"  # 한국어 모델 예시
    
    print(f"오디오 파일 '{audio_file}'을 인식하는 중...")
    result = recognize_speech(audio_file, model_name)
    
    print("\n--- 인식 결과 ---")
    print(f"텍스트: {result['text']}")
    print(f"신뢰도: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()