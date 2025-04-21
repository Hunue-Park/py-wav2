import numpy as np
import soundfile as sf
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from tokenizers import Tokenizer
from collections import defaultdict
import logging
import torch
from dtw import dtw

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
        # 1) session & model load
        providers = ["CPUExecutionProvider"] if device.upper() == "CPU" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.onnx_model = onnx.load(onnx_model_path)

        # 2) tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # 3) I/O names
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_name = inputs[0].name
        self.hidden_name = outputs[0].name
        self.logits_name = outputs[1].name
        logger.debug("Model IO names: input=%s, hidden=%s, logits=%s",
                     self.input_name, self.hidden_name, self.logits_name)

        # 4) infer hidden_dim & vocab_size from output shapes
        hidden_dim = None
        vocab_size = None
        for output in self.onnx_model.graph.output:
            dims = output.type.tensor_type.shape.dim
            if output.name == self.hidden_name:
                hidden_dim = dims[2].dim_value
            elif output.name == self.logits_name:
                vocab_size = dims[2].dim_value
        if hidden_dim is None or vocab_size is None:
            raise RuntimeError("Could not determine hidden_dim or vocab_size from model outputs.")

        # 5) load & dequantize lm_head weight (prototype matrix)
        proto = None
        # try to find quantized weight + scale + zero_point
        quant_init = None
        for init in self.onnx_model.graph.initializer:
            arr = numpy_helper.to_array(init)
            # look for (hidden_dim, vocab_size) quantized weight
            if arr.ndim == 2 and arr.shape == (hidden_dim, vocab_size) and init.name.endswith("_quantized"):
                quant_init = init
                quant_arr = arr.astype(np.float32)
                break

        if quant_init is not None:
            base = quant_init.name[:-len("_quantized")]
            # find scale and zero_point of shape (vocab_size,)
            scale_arr = numpy_helper.to_array(
                next(i for i in self.onnx_model.graph.initializer if i.name == base + "_scale")
            ).astype(np.float32)
            zp_arr = numpy_helper.to_array(
                next(i for i in self.onnx_model.graph.initializer if i.name == base + "_zero_point")
            ).astype(np.float32)
            # dequantize: (Q - zp) * scale
            dequant = (quant_arr - zp_arr) * scale_arr
            # transpose => (vocab_size, hidden_dim)
            proto = dequant.T
        else:
            # fallback: scan initializers for exact or transposed orientation
            for init in self.onnx_model.graph.initializer:
                arr = numpy_helper.to_array(init)
                if arr.ndim == 2 and arr.shape == (vocab_size, hidden_dim):
                    proto = arr
                    break
            if proto is None:
                for init in self.onnx_model.graph.initializer:
                    arr = numpy_helper.to_array(init)
                    if arr.ndim == 2 and arr.shape == (hidden_dim, vocab_size):
                        proto = arr.T
                        break

        if proto is None:
            raise RuntimeError("Prototype matrix (lm_head weight) not found in any initializer.")
        self.prototype_matrix = proto

        logger.debug("Loaded prototype_matrix of shape %s", self.prototype_matrix.shape)
        
    def load_and_preprocess(
        self,
        audio_path: str,
        target_sr: int = 16000,
        do_normalize: bool = True
    ) -> torch.Tensor:
        # 1) load
        audio, sr = sf.read(audio_path, dtype="float32")   # 이때 float32

        # 2) monoize
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # 3) resample → target_sr (np.interp는 float64 반환)
        if sr != target_sr:
            duration = len(audio) / sr
            t_orig = np.linspace(0.0, duration, num=len(audio), endpoint=False)
            t_new  = np.linspace(0.0, duration, num=int(duration * target_sr), endpoint=False)
            audio = np.interp(t_new, t_orig, audio)  # float64

        # 4) normalize (여기도 float64)
        if do_normalize:
            m = audio.mean()
            s = audio.std()
            audio = (audio - m) / (s + 1e-8)

        # --- 여기를 추가하여 float32로 캐스팅 ---
        audio = audio.astype(np.float32)

        # 5) batch 차원 추가하여 torch.Tensor 로
        #    (libtorch/C++에서도 torch::from_blob → unsqueeze(0) 으로 동일 처리 가능)
        tensor = torch.from_numpy(audio).unsqueeze(0)  # shape: [1, T], dtype=torch.float32
        return tensor


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
        alignment = dtw(X, Y, keep_internals=True, step_pattern="asymmetricP1")
        return alignment.index1, alignment.index2

    def calculate_gop(self, audio_path: str, text: str, eps: float = 1e-8) -> dict:
        logger.debug("Calculating GOP for text: %s", text)
        temperature = 1

        # 1) load & preprocess → [1, T], float32 torch.Tensor
        values = self.load_and_preprocess(audio_path)      # torch.Tensor [1,T]
        input_np = values.numpy()                          # np.ndarray [1,T], float32

        # 2) run ONNX to get hidden & logits
        hidden_np, logits_np = self.session.run(
            [self.hidden_name, self.logits_name],
            {self.input_name: input_np}
        )
        # remove batch dim
        X      = hidden_np[0]   # shape (T, D)
        logits = logits_np[0]    # shape (T, V)

        # 3) temperature‐scaled softmax → probs
        scaled     = logits
        exp_logits = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs      = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        logger.debug("Computed probabilities shape=%s", probs.shape)

        # 4) tokenize
        text = text.replace(" ", "|")
        token_ids = self.tokenizer.encode(text).ids
        logger.debug("Token IDs: %s", token_ids)

        V = self.prototype_matrix.shape[0]
        blank_id = self.tokenizer.token_to_id("|")
        safe_ids = [tid if 0 <= tid < V else blank_id for tid in token_ids]
        logger.debug("Safe token IDs: %s", safe_ids)

        # 5) expand prototypes & DTW
        proto = self.prototype_matrix[safe_ids]  # (M, D)
        T, M   = X.shape[0], len(safe_ids)
        avg    = max(1, T // M)
        Yexp   = np.repeat(proto, avg, axis=0)  # (M*avg, D)
        pX, pYexp = self.dtw_align(X, Yexp)
        pY    = [y // avg for y in pYexp]

        # 6) collect frames per token
        frames = defaultdict(list)
        for f, t in zip(pX, pY):
            frames[t].append(f)
        logger.debug("Frames per token: %s", {k: len(v) for k, v in frames.items()})

        # 7) per‐token log‐prob scores
        tok_scores = []
        for idx, tid in enumerate(safe_ids):
            tok = self.tokenizer.id_to_token(tid)
            frs = frames.get(idx, [])
            if frs:
                ps    = probs[frs, tid]
                score = float(np.mean(np.log(ps + eps)))
            else:
                score = float(-np.inf)
            tok_scores.append((tok, score))
        logger.debug("Token scores: %s", tok_scores)

        # 8) normalize to [0,100]
        raw  = np.array([s for _, s in tok_scores], dtype=np.float32)
        mask = np.isfinite(raw)
        if mask.any():
            mn   = raw[mask].min()
            mx   = raw[mask].max()
            span = mx - mn if mx > mn else eps
            norm = [(t, (s - mn) / span * 100.0 if np.isfinite(s) else 0.0)
                    for t, s in tok_scores]
        else:
            norm = [(t, 0.0) for t, _ in tok_scores]
        logger.debug("Normalized scores: %s", norm)

        # 9) group into words
        words, ct, cs = [], [], []
        for t, sc in norm:
            if t == "[UNK]":
                continue
            if t == "|":
                if ct:
                    words.append({
                        "word": "".join(ct),
                        "scores": {"pronunciation": round(sum(cs) / len(cs))}
                    })
                ct, cs = [], []
            else:
                ct.append(t)
                cs.append(sc)
        if ct:
            words.append({
                "word": "".join(ct),
                "scores": {"pronunciation": round(sum(cs) / len(cs))}
            })

        overall = (
            round(sum(w["scores"]["pronunciation"] for w in words) / len(words), 1)
            if words else 0.0
        )
        logger.debug("Final GOP overall=%.1f", overall)
        return {"overall": overall, "pronunciation": overall, "words": words}

    def transcribe(self, audio_path: str, raw_ids: list) -> str:
        # logger.debug("Transcribing %s", audio_path)
        # audio = self.load_audio(audio_path)
        # logits = self.extract_logits(audio)
        # logits.shape == (T, V) 이어야 함

        # raw_ids = logits.argmax(axis=1).tolist()
        logger.debug("Raw Greedy IDs: %s", raw_ids)

        # special tokens 문자열로 직접 지정
        blank_token = "|"      # CTC blank
        pad_token   = "[PAD]"
        unk_token   = "[UNK]"

        # ID 조회
        blank_id = self.tokenizer.token_to_id(blank_token)
        pad_id   = self.tokenizer.token_to_id(pad_token)
        unk_id   = self.tokenizer.token_to_id(unk_token)

        dedup_ids = []
        prev = None
        for idx in raw_ids:
            # special token이면 prev도 초기화해서 다음 non‑special이 반드시 append 되도록
            if idx in (blank_id, pad_id, unk_id):
                prev = None
                continue

            # non‑special인데 연속 중복이면 skip
            if idx == prev:
                continue

            dedup_ids.append(idx)
            prev = idx

        logger.debug("CTC‑decoded IDs: %s", dedup_ids)

        # 최종 디코딩
        text = self.tokenizer.decode(dedup_ids)
        logger.debug("Decoded text: %s", text)
        return text

