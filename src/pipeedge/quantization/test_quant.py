from basic_op import _quant_op, _adaptive_quant_op
import numpy as np
import ipdb
import torch


# input_data = np.asarray([0.52,0.8,0.3]*3)
# print(input_data)
b,h,w = 2,3,4
input_data = torch.rand(b,h,w, dtype=torch.float32) #8,768,3072
# input_data = input_tensor.numpy()
ipdb.set_trace()
# res, int_map = _quant_op(input_data, bit = 4)
_, int_map = _adaptive_quant_op(input_data, n_bits=4, n_exp=2, bias = None)


