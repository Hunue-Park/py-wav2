# main.py

import logging
import numpy as np
import onnxruntime as ort
import torch
from collections import Counter

from realtime_engine.w2v_onnx_core import Wav2VecCTCOnnxCore
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_logits(ids_pt, ids_onx, special, slice_start=100, slice_end=150):
    print("\nPT unique ids & counts:", Counter(ids_pt).most_common())
    print("ONNX unique ids & counts:", Counter(ids_onx).most_common())

    non_special_pt  = np.where(~np.isin(ids_pt, special))[0]
    non_special_onx = np.where(~np.isin(ids_onx, special))[0]
    print("\nPT first non‑special frames:",  non_special_pt[:10].tolist())
    print("ONNX first non‑special frames:", non_special_onx[:10].tolist())

    print(f"\nPT argmax[{slice_start}:{slice_end}] =",  ids_pt[slice_start:slice_end].tolist())
    print(f"ONNX argmax[{slice_start}:{slice_end}] =", ids_onx[slice_start:slice_end].tolist())

def main():
    audio_path = "./recordings/recorded_audio_1.wav"
    processor = Wav2Vec2Processor.from_pretrained("./env/fine-tuned-wav2vec2-kspon")
    model     = Wav2Vec2ForCTC.from_pretrained("./env/fine-tuned-wav2vec2-kspon")
    model.eval().cpu()

    audio, _     = librosa.load(audio_path, sr=16000)
    inputs       = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    dummy_values = inputs.input_values  # [1, T]

    sess = ort.InferenceSession("./env/wav2vec2_ctc_dynamic.onnx", providers=["CPUExecutionProvider"])

    # 1) PT / ONNX logits
    with torch.no_grad():
        logits_pt = model(dummy_values).logits.cpu().numpy()
    [logits_onx] = sess.run(["logits"], {"input_values": dummy_values.numpy()})

    print("MAE FP32 ONNX↔PT:", np.mean(np.abs(logits_pt - logits_onx)))

    # 2) argmax → id 시퀀스 ([T])
    ids_pt  = np.argmax(logits_pt,  axis=-1)[0]
    ids_onx = np.argmax(logits_onx, axis=-1)[0]

    # 3) special token set
    pad_id   = processor.tokenizer.pad_token_id
    unk_id   = processor.tokenizer.unk_token_id
    blank_id = pad_id
    special  = {pad_id, unk_id, blank_id}

    # 4) 상세 비교
    compare_logits(ids_pt, ids_onx, special)

    # 5) CTC 디코딩 비교
    onnx_engine = Wav2VecCTCOnnxCore(
        onnx_model_path='./env/wav2vec2_ctc_dynamic.onnx',
        tokenizer_path='./env/fine-tuned-wav2vec2-kspon/tokenizer.json'
    )
    values = onnx_engine.load_and_preprocess(audio_path)
        # torch.Tensor 인 경우
    if hasattr(values, "dtype") and values.dtype == torch.float64:
        values = values.float()
    [engine_logits] = sess.run(["logits"], {"input_values": values.numpy()})

    print('\nfinal compare: ',np.mean(np.abs(logits_pt - engine_logits)))
    print("\nPyTorch   :", processor.decode(ids_pt, skip_special_tokens=True))
    print("ONNX (CTC):", onnx_engine.transcribe(audio_path, ids_onx))

if __name__ == "__main__":
    main()
