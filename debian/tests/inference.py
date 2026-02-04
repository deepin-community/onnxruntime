# step 1: create a dummy neural network with pytorch, and export
import torch

model = torch.nn.Sequential(
        torch.nn.Linear(128, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
        )

input_tensor = torch.rand((1, 128), dtype=torch.float32)

filename = 'test_model.onnx'
torch.onnx.export(
        model,
        (input_tensor,),
        filename,
        input_names=['feature'],
        dynamo=False,
        verbose=True,
        )
print('Exported a dummy neural network at', filename)

# step 2: load with onnxruntime and do the inference
import numpy as np
import onnxruntime as rt

print('ORT: available providers:', rt.get_available_providers())
session = rt.InferenceSession('test_model.onnx', providers=rt.get_available_providers())
print('ORT: loaded the dummy network')
input_name = session.get_inputs()[0].name
inputs_onnx = {'feature': np.random.randn(1, 128).astype(np.float32)}
pred_onnx = session.run(None, inputs_onnx)[0]
print('pred_onnx:', pred_onnx, 'shape:', pred_onnx.shape)
assert pred_onnx.shape == (1, 1)

print('Test OK')
