import onnx
from onnx import numpy_helper
import numpy as np
# 1) Load your ONNX
model = onnx.load("./env/wav2vec2_ctc_dynamic.onnx")

# 2) pick off the three initializers that QLinearMatMul will have produced
#    (you saw these in your printout):
q_name     = "onnx::MatMul_3745_quantized"
scale_name = "onnx::MatMul_3745_scale"
zp_name    = "onnx::MatMul_3745_zero_point"

# helper to pull out a single initializer tensor by name
def get_init(name):
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    raise KeyError(f"initializer '{name}' not found")

quant   = get_init(q_name)        # shape: (hidden_dim, vocab_size) == (1024, 1205)
scale   = get_init(scale_name)    # shape: (vocab_size,)
zero_pt = get_init(zp_name)       # shape: (vocab_size,)

# 3) dequantize: W_fp = (q - zero_point) * scale
#    note: broadcast zero_pt & scale across rows
W_fp = (quant.astype(np.float32) - zero_pt.astype(np.float32)) * scale.astype(np.float32)

# 4) transpose so you get (vocab_size, hidden_dim)
prototype_matrix = W_fp.T       # now (1205, 1024)

print("Recovered prototype_matrix.shape =", prototype_matrix.shape)
