import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from nnAudio import features

class mse_loss(nn.Module):
    def forward(self, y_pred, y_true):
        #print(f"y_pred: {y_pred} | y_true: {y_true}")
        #print(f"y_pred shape: {y_pred.shape} | y_true shape: {y_true.shape}")

        B, C, T = y_pred.shape
        
        loss = 0.0
        
        # per-channel loss
        for c in range(C):
            loss += torch.mean((torch.abs(y_pred[:, c, :]) - torch.abs(y_true)) ** 2)

        # system output
        y_pred_sum = torch.sum(y_pred, dim=1)   # [B, T]
        loss += torch.mean((torch.abs(y_pred_sum) - torch.abs(y_true)) ** 2)

        loss /= (C + 1)

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
        self.eps = 1e-6

        self.l1loss = nn.L1Loss()

        self.stfts = nn.ModuleList()
        for n_fft in self.n_fft:
            hop = int(n_fft * (1 - self.overlap))
            self.stfts.append(
                features.stft.STFT(
                    n_fft=n_fft,
                    hop_length=hop,
                    window='hann',
                    freq_scale='log',
                    sr=sr,
                    fmin=20,
                    fmax=sr // 2,
                    output_format='Magnitude',
                    verbose=False,
                )
            )

    def forward(self, y_pred, y_true):
        # y_pred, y_true: [B, 1, T] or [B, T]
        y_pred = y_pred.float()
        y_true = y_true.float()

        loss = 0.0

        for stft in self.stfts:
            Yp = stft(y_pred)
            Yt = stft(y_true)
            #print(f"Yp: {Yp}")
            #print(f"Yt: {Yt}")

            loss = loss + self.l1loss(Yp, Yt)
            #print(f"loss: {loss}")

        # optional: average over scales
        return loss / len(self.stfts)

