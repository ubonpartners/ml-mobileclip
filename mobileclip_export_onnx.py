import onnxruntime as rt, numpy as np
import torch
from mobileclip import create_model_and_transforms
from torch.onnx import OperatorExportTypes

# use mobileclip_b for the big one
model="mobileclip_s0"
output_onnx="/mldata/models/clip/onnx/"+model+"_vision_batch.onnx"
mobileclip_model="/mldata/models/clip/pt/"+model+".pt"

# 1. Load & trace
device = "cuda"
model,_, _ = create_model_and_transforms(
    model,
    pretrained=mobileclip_model
)

model = model.to(device).eval()
vision = model.image_encoder

# Trace with a fixed batch=1, H=W=224
dummy = torch.randn(1, 3, 224, 224, device=device)
vision_ts = torch.jit.trace(vision, dummy)

# 2. Export WITHOUT dynamic axes, and with ATen fallback
torch.onnx.export(
    vision_ts,
    dummy,
    output_onnx,
    export_params=True,
    opset_version=20,                            # bumping opset sometimes helps
    input_names=["input"],
    output_names=["features"],
    # No dynamic_axes: batch/shape is fixed at export time
    #dynamic_axes=None,
    dynamic_axes={                 # you can still allow batch-size to vary
        "input": {0: "batch_size"},
        "features": {0: "batch_size"},
    },
    # Let unknown ops fall back to ATen instead of failing
    #operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
    #do_constant_folding=False
)

print("✅ ONNX export done (static shapes).")

sess = rt.InferenceSession(output_onnx, providers=["CUDAExecutionProvider"])
x = np.random.randn(1,3,224,224).astype(np.float32)
print(sess.run(None, {"input": x})[0].shape)

print("✅ ONNX tested ok")


# convert to TRT
# /usr/src/tensorrt/bin/trtexec   --onnx=mobileclip_vision.onnx   --saveEngine=mobileclip_fp16.engine --fp16
