import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import sys

print("当前Python版本为：", sys.version)
print("当前Python路径为：", sys.executable)

import torch
print("当前PyTorch版本为：", torch.__version__)
print('hello world')
tensor = torch.rand(3, 4)
print(tensor)