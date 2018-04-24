import torch
from torch.autograd import Variable
src = torch.ones((8,192,6,7))
tgt = torch.ones((8,192,6,7))
S = 4
B, C, H, W = src.size()

output = Variable(torch.zeros((B, (S*2+1)**2, H, W)))
output[:,0] = (tgt*src).sum(1).unsqueeze(1)

I = 1
for i in range(1, S + 1):
    # tgt下移i像素并补0, src与之对应的部分为i之后的像素, output的上i个像素为0
    if i < H:
        output[:,I,i:,:] = (tgt[:,:,:-i,:] * src[:,:,i:,:]).sum(1).unsqueeze(1); I += 1
    output[:,I,:-i,:] = (tgt[:,:,i:,:] * src[:,:,:-i,:]).sum(1).unsqueeze(1); I += 1
    output[:,I,:,i:] = (tgt[:,:,:,:-i] * src[:,:,:,i:]).sum(1).unsqueeze(1); I += 1
    output[:,I,:,:-i] = (tgt[:,:,:,i:] * src[:,:,:,:-i]).sum(1).unsqueeze(1); I += 1

    for j in range(1, S + 1):
        output[:,I,i:,j:] = (tgt[:,:,:-i,:-j] * src[:,:,i:,j:]).sum(1).unsqueeze(1); I += 1
        output[:,I,:-i,:-j] = (tgt[:,:,i:,j:] * src[:,:,:-i,:-j]).sum(1).unsqueeze(1); I += 1
        output[:,I,i:,:-j] = (tgt[:,:,:-i,j:] * src[:,:,i:,:-j]).sum(1).unsqueeze(1); I += 1
        output[:,I,:-i,j:] = (tgt[:,:,i:,:-j] * src[:,:,:-i,j:]).sum(1).unsqueeze(1); I += 1

print(output)