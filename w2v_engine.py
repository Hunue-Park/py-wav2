import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from dtw import dtw, rabinerJuangStepPattern
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
import librosa
import pprint
import matplotlib.pyplot as plt
class Wav2VecCTCEngine:
    _instances = {}  # 모델 인스턴스 캐싱을 위한 클래스 변수
    
    @classmethod
    def get_instance(cls, model_name, device=None):
        """싱글톤 패턴: 같은 모델은 한 번만 로드하여 메모리 절약"""
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name, device)
        return cls._instances[model_name]
    
    def __init__(self,
                 model_name: str = 'kresnik/wav2vec2-large-xlsr-korean',
                 device: torch.device = None):
        """
        HuggingFace pretrained wav2vec2 모델과 processor를 불러옵니다.
        Args:
            model_name: 사용할 사전학습된 모델 이름
            device: 모델을 로드할 장치 (None이면 자동 감지)
        """
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 캐시 저장 경로 설정
        cache_dir = Path.home() / ".cache" / "py-wav2"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # processor, embedding 모델, CTC head 모델 로드
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=cache_dir)
        self.feature_extractor = Wav2Vec2Model.from_pretrained(model_name, cache_dir=cache_dir)
        self.ctc_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=cache_dir)

        # 장치 이동 및 평가 모드
        self.feature_extractor.to(self.device).eval()
        self.ctc_model.to(self.device).eval()

    def load_audio(self, path: str, target_sr: int = 16000):
        audio, sr = librosa.load(path, sr=target_sr)
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        return audio, sr

    def extract_embedding(self, audio: np.ndarray, sr: int):
        """
        오디오 배열로부터 Wav2Vec2 임베딩 시퀀스 (T, D) 반환
        """
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            hidden = self.feature_extractor(
                inputs.input_values.to(self.device)
            ).last_hidden_state
        return hidden.squeeze(0).cpu().numpy()

    def extract_logits(self, audio: np.ndarray, sr: int):
        """
        오디오 배열로부터 Wav2Vec2 CTC logits 시퀀스 (T, V) 반환
        """
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.ctc_model(
                inputs.input_values.to(self.device)
            ).logits
        return logits.squeeze(0).cpu().numpy()

    def get_prototypes(self):
        """
        CTC 헤드의 lm_head 가중치로부터 token prototypes (V, D) 반환
        """
        return self.ctc_model.lm_head.weight.detach().cpu().numpy()

    def dtw_align(self, X: np.ndarray, Y: np.ndarray):
        """
        DTW를 수행하여 최적 경로 인덱스 리스트 (path_X, path_Y) 반환
        global_constraint: Sakoe-Chiba 등의 윈도우 제약
        """

        alignment = dtw(X, Y, keep_internals=True, step_pattern="asymmetricP1",)
        return alignment.index1, alignment.index2

    def align_audio_to_text(self, audio_path: str, text: str, global_constraint=None):
        """
        전체 파이프라인: 오디오 로드 → 피처 추출 → text→ids → prototype 추출 → DTW 정렬 → segments 반환
        segments: dict[token_index] = (start_frame, end_frame)
        """
        # 1) audio → embedding X
        audio, sr = self.load_audio(audio_path)
        X = self.extract_embedding(audio, sr)
        print(X.shape, 'X.shape')
        T = X.shape[0]

        # 2) text → input_ids (음절 단위)
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.squeeze(0).cpu().tolist()

        # 3) token prototypes
        proto = self.get_prototypes()
        Y = proto[input_ids]  # shape (M, D)
        M = Y.shape[0]

        # 4) 평균 토큰당 프레임 수로 Y 확장
        avg_len = max(1, T // M)
        Y_expanded = np.repeat(Y, avg_len, axis=0)  # shape (M*avg_len, D)

        # 4) DTW alignment
        path_X, path_Y = self.dtw_align(X, Y_expanded)
        print(path_X, 'path_X')
        print(path_Y, 'path_Y')

        # 5) segment 계산
        from collections import defaultdict
        segments = defaultdict(list)
        for i_frame, j_tok in zip(path_X, path_Y):
            segments[j_tok].append(i_frame)

        boundaries = {j: (min(frames), max(frames)) for j, frames in segments.items()}
        return {self.processor.tokenizer.convert_ids_to_tokens([j])[0]: boundaries[j] for j in boundaries}
    
    def calculate_gop(self, audio_path: str, text: str, eps: float = 1e-8):
        """음절 단위 GOP 점수 계산 (텍스트 순서 유지)"""
        # 1) 오디오 로드
        audio, sr = self.load_audio(audio_path)
        # 2) logits → probabilities (softmax 안정화)
        logits = self.extract_logits(audio, sr)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 3) 피처 및 DTW 정렬
        X = self.extract_embedding(audio, sr)
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.squeeze(0).cpu().tolist()
        T = X.shape[0]

        proto = self.get_prototypes()
        Y = proto[input_ids]
        M = Y.shape[0]
        avg_len = max(1, T // M)
        Y_expanded = np.repeat(Y, avg_len, axis=0)

        path_X, path_Y_exp = self.dtw_align(X, Y_expanded)
        path_Y = [int(idx // avg_len) for idx in path_Y_exp]

        # 4) 토큰별 프레임 그룹화
        from collections import defaultdict
        frames_per_token = defaultdict(list)
        for i_frame, j_tok in zip(path_X, path_Y):
            frames_per_token[j_tok].append(i_frame)

        # 5) GOP 점수 리스트 생성
        gop_list = []  # (token_str, gop_score) 순차 리스트
        for j_tok, token_id in enumerate(input_ids):
            token_str = self.processor.tokenizer.convert_ids_to_tokens([token_id])[0]
            frames = frames_per_token.get(j_tok, [])
            if frames:
                token_probs = probs[frames, token_id]
                score = float(np.mean(np.log(token_probs + eps)))
            else:
                score = float('-inf')
            gop_list.append((token_str, score))

        # 6) 정규화 (min-max 스케일)
        raw_scores = np.array([score for _, score in gop_list])
        valid = np.isfinite(raw_scores)
        normalized = []
        if valid.any():
            min_s = raw_scores[valid].min()
            max_s = raw_scores[valid].max()
            span = max_s - min_s if max_s > min_s else eps
            for token, score in gop_list:
                if np.isfinite(score):
                    norm_score = (score - min_s) / span * 100.0
                else:
                    norm_score = 0.0
                normalized.append((token, float(norm_score)))
        else:
            normalized = [(token, 0.0) for token, _ in gop_list]

        return normalized
    
    def format_gop_as_response(self, gop_list):
        """
        gop_list: [(token_str, score), ...] 
            - token_str에는 한글 음절 혹은 '|'(스페이스) 포함
            - score는 0~100 사이 float
        """
        words = []
        curr_tokens = []
        curr_scores = []
        for token, score in gop_list:
            if token == "[UNK]":
                continue
            if token == "|":
                # 공백이니까 지금까지 모인 음절을 하나의 단어로
                if curr_tokens:
                    word = "".join(curr_tokens)
                    avg_score = sum(curr_scores) / len(curr_scores)
                    words.append({
                        "word": word,
                        "scores": {"pronunciation": round(avg_score, 0)}
                    })
                    curr_tokens = []
                    curr_scores = []
            else:
                curr_tokens.append(token)
                curr_scores.append(score)
        # 마지막 단어 flush
        if curr_tokens:
            word = "".join(curr_tokens)
            avg_score = sum(curr_scores) / len(curr_scores)
            words.append({
                "word": word,
                "scores": {"pronunciation": round(avg_score, 0)}
            })

        # overall 점수는 단어별 점수 평균
        if words:
            overall = sum(w["scores"]["pronunciation"] for w in words) / len(words)
        else:
            overall = 0.0

        result = {
            "overall": round(overall, 1),
            "pronunciation": round(overall, 1),
            "words": words
        }
        return result


# 사용 예시:
# w2v = Wav2VecCTC.get_instance('kresnik/wav2vec2-large-xlsr-korean')\#
# segments = w2v.align_audio_to_text('user.wav', '안녕하세요. 오늘 날씨가 좋네요.')
# print(segments)  # {'안': (0,10), '녕': (11,20), ...}