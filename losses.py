import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import audio_utility as util

class spectral_loss(nn.Module):
    def forward(self, y_pred, y_true):
        raise NotImplementedError

class sparsity_loss(nn.Module):
    def forward(self, y_pred, y_true):
        raise NotImplementedError

class mse_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    def forward(self, y_pred, y_true):
        y_pred_freq = torch.fft.rfft(y_pred)
        y_true_freq = torch.fft.rfft(y_true)

        time_loss = self.loss(y_pred, y_true)
        freq_loss = self.loss(torch.abs(y_pred_freq), torch.abs(y_true_freq))

        return time_loss + freq_loss

class STFTLoss(nn.Module):
    def __init__(self, sr=48000):
        super().__init__()
        n_ffts = [256, 512, 1024, 2048, 4096]
        overlap = 0.875
        self.sr = sr
        self.eps = 1e-6

        self.l1loss = nn.L1Loss()
        self.mseloss = nn.MSELoss()

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

            loss += self.mseloss(Yp, Yt)
            #loss += self.l1loss(Yp, Yt)
            #loss += self.loss(y_pred, y_true)

        #print(f"loss: {loss}")
        # optional: average over scales
        return loss / len(self.stfts)

