import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import onnxruntime as ort
from realtime_engine.w2v_onnx_core import Wav2VecCTCOnnxCore
# 1) 준비
audio_path = "./hello_in_dinner.wav"
processor = Wav2Vec2Processor.from_pretrained("./env/fine-tuned-wav2vec2-kspon")
model     = Wav2Vec2ForCTC.from_pretrained("./env/fine-tuned-wav2vec2-kspon")
model.eval().cpu()

sess = ort.InferenceSession("./env/wav2vec2_ctc_dynamic.onnx", providers=["CPUExecutionProvider"])
onnx_engine = Wav2VecCTCOnnxCore(
    onnx_model_path='./env/wav2vec2_ctc_dynamic.onnx',
    tokenizer_path='./env/fine-tuned-wav2vec2-kspon/tokenizer.json'
)

# 2) HF processor 로 만든 입력 (길이 150869)
#    manual 과 동일한 리샘플된 음성을 processor 에 넣습니다.
# 1) manual 로 전처리
manual_values = onnx_engine.load_and_preprocess(audio_path)  # torch.Tensor [1, T], dtype=torch.float32
input_np = manual_values.numpy()                            # numpy array [1, T], float32

# 2) PT 로 logits
with torch.no_grad():
    pt_logits = model(torch.from_numpy(input_np)).logits.cpu().numpy()  # shape: (1, T, V)

# 3) ONNX 로 logits
onnx_logits = sess.run(
    ["logits"],
    {"input_values": input_np}  # 여기서도 반드시 `manual` 을 사용!
)[0]  # shape: (1, T, V)

# 4) MAE 계산
mae = np.mean(np.abs(pt_logits - onnx_logits))
print("▶ Manual vs ONNX logits MAE:", mae)

with torch.no_grad():
    hf_out = model(torch.from_numpy(input_np), output_hidden_states=True)
hidden_pt = hf_out.hidden_states[-1][0].cpu().numpy()  # (T, D)

# (b) ONNX 에서 hidden 뽑기
hidden_onx = onnx_engine.session.run(
    [onnx_engine.hidden_name],
    {onnx_engine.input_name: input_np}
)[0][0]  # (T, D)

print("Hidden MAE:", np.mean(np.abs(hidden_pt - hidden_onx)))
print("Hidden shape PT vs ONNX:", hidden_pt.shape, hidden_onx.shape)

probs_pt  = torch.softmax(torch.from_numpy(pt_logits), dim=-1).numpy()
probs_onx = torch.softmax(torch.from_numpy(onnx_logits), dim=-1).numpy()
print("Prob MAE:", np.mean(np.abs(probs_pt - probs_onx)))

diff = pt_logits - onnx_logits
flat = diff.reshape(-1, diff.shape[-1])
# 1) 프레임별 최대 절대 오차
frame_err = np.max(np.abs(flat), axis=1)
print("Top-5 worst frames:", np.argsort(frame_err)[-5:], frame_err[np.argsort(frame_err)[-5:]])
# 2) 어휘별 최대 절대 오차
token_err = np.max(np.abs(flat), axis=0)
print("Top-5 worst tokens:", np.argsort(token_err)[-5:], token_err[np.argsort(token_err)[-5:]])

hf_proto   = model.lm_head.weight.detach().cpu().numpy()      # (V, D)
onx_proto  = onnx_engine.prototype_matrix                       # (V, D)
print("Prototype MAE:", np.mean(np.abs(hf_proto - onx_proto)))