# -*- coding = utf-8 -*-
# @Time     : 2025/7/1 00:29
# @Author   : Yao Jiamin
# @File     : test.py
# @Software : PyCharm
import torch
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.is_available())
