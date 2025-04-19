import numpy as np
import soundfile as sf
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from tokenizers import Tokenizer
from collections import defaultdict
import logging
# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Wav2VecCTCOnnxEngine:
    """
    ONNX Runtime based Wav2Vec2 CTC inference engine using a quantized ONNX model.
    Includes fallback for prototype matrix in either orientation.
    """

    def __init__(
        self,
        onnx_model_path: str,
        tokenizer_path: str,
        device: str = "CPU"
    ):
        providers = ["CPUExecutionProvider"] if device.upper() == "CPU" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.onnx_model = onnx.load(onnx_model_path)

        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # IO names
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_name = inputs[0].name
        self.hidden_name = outputs[0].name
        self.logits_name = outputs[1].name
        logger.debug("Model IO names: input=%s, hidden=%s, logits=%s",
                     self.input_name, self.hidden_name, self.logits_name)

        # Determine hidden dimension
        hidden_dim = None
        for output in self.onnx_model.graph.output:
            if output.name == self.hidden_name:
                dims = output.type.tensor_type.shape.dim
                hidden_dim = dims[2].dim_value
                break
        if hidden_dim is None:
            raise RuntimeError("Hidden output dimension not found.")

        # Determine vocab size V from logits output
        vocab_size = None
        for output in self.onnx_model.graph.output:
            if output.name == self.logits_name:
                dims = output.type.tensor_type.shape.dim
                vocab_size = dims[2].dim_value
                break
        if vocab_size is None:
            raise RuntimeError("Logits output dimension not found.")

        # Load prototype matrix (lm_head weight) from initializers
        self.prototype_matrix = None
        # Collect all initializer shapes for debugging
        for init in self.onnx_model.graph.initializer:
            arr = numpy_helper.to_array(init)
            # Match (V, hidden_dim)
            if arr.ndim == 2 and arr.shape == (vocab_size, hidden_dim):
                self.prototype_matrix = arr
                break
        # Fallback: check transposed orientation
        if self.prototype_matrix is None:
            for init in self.onnx_model.graph.initializer:
                arr = numpy_helper.to_array(init)
                if arr.ndim == 2 and arr.shape == (hidden_dim, vocab_size):
                    self.prototype_matrix = arr.T
                    break
        if self.prototype_matrix is None:
            raise RuntimeError("Prototype matrix not found in any initializer orientation.")

    def load_audio(self, file_path: str, target_sr: int = 16000) -> np.ndarray:
        logger.debug("Loading audio: %s", file_path)
        audio, sr = sf.read(file_path)
        logger.debug("Original SR=%d, samples=%d", sr, len(audio))
        emphasis_value = 0.97
        if sr != target_sr:
            duration = len(audio) / sr
            new_length = int(duration * target_sr)
            audio = np.interp(
                np.linspace(0.0, duration, new_length, endpoint=False),
                np.linspace(0.0, duration, len(audio), endpoint=False),
                audio
            )
            logger.debug("Resampled to %d samples", new_length)
        emphasized = np.append(audio[0], audio[1:] - emphasis_value * audio[:-1])
        logger.debug("Applied pre-emphasis, output len=%d", len(emphasized))
        return emphasized.astype(np.float32)

    def _infer(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logger.debug("Running inference on audio len=%d", len(audio))
        batch = audio[np.newaxis, :]
        hidden, logits = self.session.run([self.hidden_name, self.logits_name], {self.input_name: batch})
        logger.debug("Inference outputs: hidden=%s, logits=%s", hidden.shape, logits.shape)
        return hidden[0], logits[0]

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        hidden, _ = self._infer(audio)
        logger.debug("Extracted embedding shape=%s", hidden.shape)
        return hidden

    def extract_logits(self, audio: np.ndarray) -> np.ndarray:
        _, logits = self._infer(audio)
        logger.debug("Extracted logits shape=%s", logits.shape)
        return logits

    def dtw_align(self, X, Y):
        N, M = X.shape[0], Y.shape[0]
        cost = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)
        D = np.full((N+1, M+1), np.inf, dtype=np.float32)
        D[0,0] = 0.0

        for i in range(1, N+1):
            for j in range(1, M+1):
                c = cost[i-1, j-1]
                # 대각선 (1,1)
                diag = D[i-1, j-1] + c
                # 수직 (1,0)
                vert = D[i-1, j]   + c
                # 확장 대각 (2,1) – asymmetricP1 특유의 이동
                if i-2 >= 0:
                    ext = D[i-2, j-1] + 2*c  # 가중치는 c 또는 2*c 등으로 조정
                else:
                    ext = np.inf
                D[i, j] = min(diag, vert, ext)

        # 경로 역추적은 기존과 동일
        i, j = N, M
        path_X, path_Y = [], []
        while i>0 and j>0:
            path_X.insert(0, i-1)
            path_Y.insert(0, j-1)
            choices = [
                (D[i-1, j-1], i-1, j-1),
                (D[i-1, j],   i-1, j),
            ]
            if i-2 >= 0:
                choices.append((D[i-2, j-1], i-2, j-1))
            _, i, j = min(choices, key=lambda x: x[0])
        return path_X, path_Y

    def calculate_gop(self, audio_path: str, text: str, eps: float = 1e-8) -> dict:
        logger.debug("Calculating GOP for text: %s", text)
        temperature = 1
        audio = self.load_audio(audio_path)
        # 1) logits 추출
        logits = self.extract_logits(audio)

        # 2) Temperature scaling
        scaled = logits / temperature

        # 3) 안정화된 softmax
        exp_logits = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        logger.debug("Computed probabilities shape=%s", probs.shape)

        X = self.extract_embedding(audio)
        token_ids = self.tokenizer.encode(text).ids
        logger.debug("Token IDs: %s", token_ids)

        V = self.prototype_matrix.shape[0]
        blank_id = self.tokenizer.token_to_id("|")
        safe_ids = [tid if 0 <= tid < V else blank_id for tid in token_ids]
        logger.debug("Safe token IDs: %s", safe_ids)

        proto = self.prototype_matrix[safe_ids]
        T, M = X.shape[0], len(safe_ids)
        avg = max(1, T // M)
        Yexp = np.repeat(proto, avg, axis=0)
        pX, pYexp = self.dtw_align(X, Yexp)
        pY = [y // avg for y in pYexp]

        frames = defaultdict(list)
        for f, t in zip(pX, pY):
            frames[t].append(f)
        logger.debug("Frames per token: %s", {k: len(v) for k, v in frames.items()})

        tok_scores = []
        for idx, tid in enumerate(safe_ids):
            tok = self.tokenizer.id_to_token(tid)
            frs = frames.get(idx, [])
            if frs:
                ps = probs[frs, tid]
                score = float(np.mean(np.log(ps + eps)))
            else:
                score = float(-np.inf)
            tok_scores.append((tok, score))
        logger.debug("Token scores: %s", tok_scores)
        # pprint.pprint(tok_scores)

        # Corrected normalization block
        raw = np.array([s for _, s in tok_scores], dtype=np.float32)
        mask = np.isfinite(raw)
        if mask.any():
            mn = raw[mask].min()
            mx = raw[mask].max()
            span = mx - mn if mx > mn else eps
            norm = [(t, (s - mn) / span * 100.0 if np.isfinite(s) else 0.0) for t, s in tok_scores]
        else:
            norm = [(t, 0.0) for t, _ in tok_scores]
        logger.debug("Normalized scores: %s", norm)

        words, ct, cs = [], [], []
        for t, sc in norm:
            if t == "|":
                continue
            if t == "[UNK]":
                # '[UNK]' 를 단어 경계로 사용
                if ct:
                    words.append({
                        "word": "".join(ct),
                        "scores": {"pronunciation": round(sum(cs)/len(cs))}
                    })
                ct, cs = [], []
                continue
            else:
                ct.append(t)
                cs.append(sc)
        if ct:
            words.append({"word": "".join(ct), "scores": {"pronunciation": round(sum(cs) / len(cs))}})
        overall = round(sum(w["scores"]["pronunciation"] for w in words) / len(words), 1) if words else 0.0
        logger.debug("Final GOP overall=%.1f", overall)
        return {"overall": overall, "pronunciation": overall, "words": words}

    def transcribe(self, audio_path: str) -> str:
        logger.debug("Transcribing %s", audio_path)
        audio = self.load_audio(audio_path)
        logits = self.extract_logits(audio)
        ids = logits.argmax(axis=1).tolist()
        logger.debug("Greedy IDs: %s", ids)
        text = self.tokenizer.decode(ids)
        logger.debug("Decoded text: %s", text)
        return text
