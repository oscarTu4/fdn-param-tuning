import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import audio_utility as util

class mse_loss(nn.Module):
    def forward(self, y_pred, y_true):
        #print(f"y_pred: {y_pred} | y_true: {y_true}")
        print(f"y_pred shape: {y_pred.shape} | y_true shape: {y_true.shape}")

        B, C, T = y_pred.shape
        
        loss = 0.0
        
        # per-channel loss
        #for c in range(C):
        #    loss += torch.mean((torch.abs(y_pred[:, c, :]) - torch.abs(y_true)) ** 2)
        #for t in range(T):
        #    loss += torch.mean((torch.abs(y_pred[:, :, t]) - torch.abs(y_true)) ** 2)

        # system output
        y_pred_sum = torch.sum(y_pred, dim=1)   # [B, T]
        loss += torch.mean((torch.abs(y_pred_sum) - torch.abs(y_true)) ** 2)

        #loss /= (C + 1)
        
        print(f"loss: {loss}")

        return loss

class sparsity_loss(nn.Module):
    ''''''
    def forward(self, A):
        N = A.shape[-1]
        return -(torch.sum(torch.abs(A)) - N*np.sqrt(N))/(N*(np.sqrt(N)-1))

class STFTLoss(nn.Module):
    def __init__(self, sr=48000):
        super().__init__()
        n_ffts = [256, 512, 1024, 2048, 4096]
        overlap = 0.875
        self.sr = sr
        self.eps = 1e-6

        self.loss = nn.L1Loss()
        #self.loss = nn.MSELoss()
        #self.loss = nn.CrossEntropyLoss()

        self.stfts = nn.ModuleList()
        for n_fft in n_ffts:
            hop = int(n_fft * (1 - overlap))
            self.stfts.append(
                util.STFT(
                    num_fft=n_fft,
                    hop_length=hop
                )
                
            )

    def forward(self, y_pred, y_true):
        # y_pred, y_true: [B, 1, T] or [B, T]
        #print(f"y_pred shape: {y_pred.shape} | y_true shape: {y_true.shape}")
        y_pred = y_pred.float()
        y_true = y_true.float()

        loss = 0.0

        for stft in self.stfts:
            Yp, _ = stft.encode(y_pred) # nur magnitude wird aktuell trainiert
            Yt, _ = stft.encode(y_true)

            loss += self.loss(Yp, Yt)
            #loss += self.loss(y_pred, y_true)

        #print(f"loss: {loss}")
        # optional: average over scales
        return loss / len(self.stfts)

