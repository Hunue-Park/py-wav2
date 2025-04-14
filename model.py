# model.py
import torch
import logging
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pathlib import Path

logger = logging.getLogger(__name__)

class Wav2VecCTC:
    _instances = {}  # 모델 인스턴스 캐싱을 위한 클래스 변수
    
    @classmethod
    def get_instance(cls, model_name, device=None):
        """싱글톤 패턴: 같은 모델은 한 번만 로드하여 메모리 절약"""
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name, device)
        return cls._instances[model_name]
    
    def __init__(self, model_name: str = 'kresnik/wav2vec2-large-xlsr-korean', device=None):
        """
        HuggingFace의 pretrained 한국어 wav2vec2 모델과 processor를 불러옵니다.
        
        Args:
            model_name: 사용할 사전학습된 모델 이름
            device: 모델을 로드할 장치 (None이면 자동 감지)
        """
        self.model_name = model_name
        
        # 장치 설정 (None인 경우 자동 감지)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"모델 '{model_name}'을 '{self.device}' 장치에 로드합니다.")
        
        try:
            # 캐시 저장 경로 설정 (사용자 홈 디렉토리의 .cache/py-wav2)
            cache_dir = Path.home() / ".cache" / "py-wav2"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # 모델과 프로세서 로드
            self.processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache_dir)
            
            # 모델을 지정된 장치로 이동
            self.model.to(self.device)
            self.model.eval()  # 평가 모드로 설정
            logger.info(f"모델 '{model_name}' 로드 완료")
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            raise RuntimeError(f"모델 '{model_name}' 로드 중 오류 발생: {str(e)}")

    def predict(self, audio, sampling_rate: int = 16000):
        """
        오디오에서 phoneme 예측을 수행합니다.
        
        Args:
            audio: numpy array 또는 list 형태의 원시 오디오 데이터  
            sampling_rate: 오디오 샘플링 주파수 (기본값 16kHz)
        
        Returns:
            logits: CTC 로짓 (모델 최종 층 출력)
            probs: 각 프레임별 softmax 확률분포
            
        Raises:
            RuntimeError: 예측 중 오류 발생 시
        """
        try:
            # 오디오 전처리
            input_values = self.processor(
                audio, 
                sampling_rate=sampling_rate, 
                return_tensors='pt'
            ).input_values.to(self.device)

            # 모델 추론
            with torch.no_grad():
                logits = self.model(input_values).logits  # (batch, time, vocab_size)
            
            # softmax 적용하여 확률 분포 계산
            probs = torch.softmax(logits, dim=-1)
            
            return logits, probs
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            raise RuntimeError(f"예측 중 오류 발생: {str(e)}")
    
    def get_vocab_size(self):
        """모델의 vocabulary 크기를 반환합니다."""
        return self.model.config.vocab_size
    
    def get_vocab(self):
        """모델의 vocabulary 사전을 반환합니다."""
        return self.processor.tokenizer.get_vocab()
    
    def get_sampling_rate(self):
        """모델이 사용하는 샘플링 레이트를 반환합니다."""
        return self.processor.feature_extractor.sampling_rate
