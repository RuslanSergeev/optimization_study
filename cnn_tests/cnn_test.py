import torch
import torch.nn as nn
import sys

sys.path.append('/home/ruslan/speechbox/scripts/train')
from save_numpy import save_numpy

#initialize parameters like sin(1), sin(2) ... sin(n) for any parameter in net.
def init_net_harm(net):
    for k, v in net.state_dict().items():
        w = torch.sin(torch.linspace(1, v.numel(), v.numel())).reshape_as(v)
        setattr(net, k, nn.Parameter(w))


#initialize input
in_ch, out_ch, h, w = 1, 3, 10, 10
x = torch.Tensor(list(range(h*w))).reshape(1, in_ch, h, w)+1

class mnet(nn.Module):
    def __init__(self):
        super().__init__()
        #create convolutional net
        kh, kw = 3, 3
        str_h, str_w = 2, 1
        pad_h, pad_w = 2, 1
        dil_h, dil_w = 2, 1
        self.cnv = nn.Conv2d(in_ch, out_ch, (kh, kw), (str_h, str_w), (pad_h, pad_w), (dil_h, dil_w))
        init_net_harm(self.cnv)

        #create recurrent net
        self.lstm = nn.LSTM(10, 2, 2, bidirectional=True)
        init_net_harm(self.lstm)

    def forward(self, x):
        y = self.cnv(x) #1, 3, 5, 10
        y, _ = self.lstm(y.reshape(15, 1, 10))
        return y.reshape(15, 4)

net = mnet()

for v in net.state_dict().values():
    print(v.shape)

y = net(x)
print(y)

save_numpy(net, 'nettest')






# avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0);
# avg_res = avg(out)
# print(avg_res)
#
# mx = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
# max_res = mx(out)
# print(max_res)
