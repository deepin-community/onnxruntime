# step 1: create a dummy neural network with pytorch, and export
import sys
import os
import torch

B = 128
N = 4096
model = torch.nn.Sequential(
        torch.nn.Linear(N, N),
        torch.nn.ReLU(),
        torch.nn.Linear(N, N),
        )

input_tensor = torch.rand((B, N), dtype=torch.float32)

print('[..] Exporting dummy neural network for mainly GEMM test')
filename = 'test_provider.onnx'
torch.onnx.export(
        model,
        (input_tensor,),
        filename,
        input_names=['feature'],
        dynamo=False,
        verbose=True,
        )
print('[OK] Exported dummy neural network for mainly GEMM test:', filename)

# step 2: load with onnxruntime and do the inference, try different providers.
import numpy as np
import onnxruntime as rt
import time
import math
elapsed_time = {}
repeat = 10

print('ORT: available providers:', rt.get_available_providers())
session = rt.InferenceSession('test_provider.onnx', providers=rt.get_available_providers())

for provider in rt.get_available_providers():
    print('Testing provider:', provider)
    session.set_providers([provider])
    input_name = session.get_inputs()[0].name
    inputs_onnx = {input_name: np.random.randn(B, N).astype(np.float32)}
    tmp = []
    for _ in range(repeat):
        time_start = time.time()
        pred_onnx = session.run(None, inputs_onnx)[0]
        time_end = time.time()
        elapsed = time_end - time_start
        print('         elapsed:', elapsed)
        tmp.append(elapsed)
    assert pred_onnx.shape == (B, N)
    elapsed_time[provider] = sum(tmp) / float(repeat)

print(elapsed_time)
