# step 1: create dummy model with pytorch
import torch

model = torch.nn.Transformer(d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, batch_first=True)
model.eval()
src = torch.rand((1, 10, 64))
tgt = torch.rand((1, 20, 64))
out = model(src, tgt)
assert out.shape == (1, 20, 64)

filename = 'test_transformer.onnx'
torch.onnx.export(
        model,
        (src, tgt),
        filename,
        input_names=['src', 'tgt'],
        dynamo=True,  # see pytorch github issue #110255
        verbose=True,
        )
print('Exported a dummy transformer at', filename)

# step 2: load with onnxruntime and do the inference
import numpy as np
import onnxruntime as rt

print('ORT: available providers:', rt.get_available_providers())
session = rt.InferenceSession('test_transformer.onnx', providers=rt.get_available_providers())
print('ORT: loaded the dummy transformer')
inputs_onnx = {'src': np.random.randn(1, 10, 64).astype(np.float32),
               'tgt': np.random.randn(1, 20, 64).astype(np.float32)}
pred_onnx = session.run(None, inputs_onnx)[0]
print('pred_onnx:', pred_onnx, 'shape:', pred_onnx.shape)

print('Test OK')
