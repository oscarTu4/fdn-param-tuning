# Differentiable FDN for Colorless Reverberation 
# custom loss functions 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from nnAudio import features

class mse_loss(nn.Module):
    '''Means squared error between abs(x1) and x2'''
    def forward(self, y_pred, y_true):
        loss = 0.0
        N = y_pred.size(dim=-1)
        # loss on channels' output
        for i in range(N):
            loss = loss + torch.mean(torch.pow(torch.abs(y_pred[:,i])-torch.abs(y_true), 2*torch.ones(y_pred.size(0))))  

        # loss on system's output
        y_pred_sum = torch.sum(y_pred, dim=-1)
        loss = loss/N + torch.mean(torch.pow(torch.abs(y_pred_sum)-torch.abs(y_true), 2*torch.ones(y_pred.size(0)))) 

        return loss

class sparsity_loss(nn.Module):
    ''''''
    def forward(self, A):
        N = A.shape[-1]
        return -(torch.sum(torch.abs(A)) - N*np.sqrt(N))/(N*(np.sqrt(N)-1))

class MSSpectralLoss(nn.Module):
    def __init__(self, sr=48000):
        super().__init__()
        self.n_fft = [256, 512, 1024, 2048, 4096]
        self.overlap = 0.875
        self.sr = sr
        self.l1loss = nn.L1Loss()

        self.stfts = nn.ModuleList()
        for n_fft in self.n_fft:
            hop = int(n_fft*(1-self.overlap))
            self.stfts.append(
                features.stft.STFT(
                    n_fft=n_fft,
                    hop_length=hop,
                    window='hann',
                    freq_scale='log',
                    sr=sr,
                    fmin=20,
                    fmax=sr//2,
                    output_format='Magnitude',
                    verbose=False,
                )
            )

    def forward(self, y_pred, y_true):
        loss = 0
        for stft in self.stfts:
            Y_pred = stft(y_pred)
            Y_true = stft(y_true)
            loss += self.l1loss(Y_pred, Y_true)
        return loss
