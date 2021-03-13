import torch
import torch.nn as nn

in_ch, out_ch = 1, 3
kh, kw = 3, 3

weights=torch.Tensor(list(range(in_ch*out_ch*kh*kw))).reshape(out_ch, in_ch, kh, kw)
bias = torch.Tensor(list(range(out_ch)))


str_h, str_w = 2, 1
pad_h, pad_w = 2, 1
dil_h, dil_w = 2, 1
cnv = nn.Conv2d(in_ch, out_ch, (kh, kw), (str_h, str_w), (pad_h, pad_w), (dil_h, dil_w))
cnv.weight = nn.Parameter(weights)
cnv.bias = nn.Parameter(bias)

h, w = 6, 6;
t = torch.Tensor(list(range(h*w))).reshape(1, in_ch, h, w)+1
out = cnv(t)

print(out)
