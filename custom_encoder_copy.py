# mit tweaks kopiert von https://github.com/gdalsanto/diff-delay-net.git
import torch
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange
from nnAudio import features
import audio_utility as util

class Encoder(nn.Module):
    def __init__(self, n_fft=1024, sr=48000, overlap=0.875):
        super().__init__()

        self.hop_length = int(n_fft*(1-overlap))
        """self.stft = features.stft.STFT(
            n_fft = n_fft,
            hop_length = self.hop_length,
            window = 'hann',
            freq_scale = 'log',
            sr = sr,
            fmin = 20,
            fmax = sr // 2,
            output_format = 'Magnitude',
            verbose=False
        )"""
        self.n_fft = n_fft
        self.sr = sr
        #self.overlap = overlap
        
        self.stft = util.STFT(
            num_fft=n_fft,
            hop_length=self.hop_length
        )
        
        conv_depth = 5
        chn_out = [64, 128, 128, 128, 128]  
        kernel = [(7,5), (5,5), (5,5), (5,5), (5,5)]
        strides = [(1,2), (2,1), (2,2), (2,2), (1,1)]
        
        self.conv_list = nn.ModuleList([])
        for i in range(conv_depth):
            if i == 0:
                chn_in = 1
            else:
                chn_in = chn_out[i-1]
            
            self.conv_list.append(
                conv2d_block(chn_in, chn_out[i], kernel[i], strides[i])
            )
        
        self.gru1 = nn.GRU(
            input_size=128,
            num_layers=2,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )
        self.gru2 = nn.GRU(
            input_size=7168,
            num_layers=1,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.lin_depth = 2
        in_feat, out_feat = 256, 256
        self.lin_list = nn.ModuleList([])
        for i in range(self.lin_depth):
            self.lin_list.append(
                linear_block(in_feat, out_feat)
            )
    
    def forward(self, x):
        printshapes = False
        b = x.shape[0]
        if printshapes:
            print(f"x.shape before stft: {x.shape}")
        x, _ = self.stft.encode(x)  # x is magnitude, _ would be phase
        #x = torch.log(x+1e-7)
        
        if printshapes:
            print(f"x.shape pre downsample: {x.shape}")

        for i, module in enumerate(self.conv_list):
            x = module(x)
            if printshapes:
                print(f"x.shape after downsample {i+1}: {x.shape}")
        
        x = rearrange(x, 'b c f t -> (b t) f c')
        x, _ = self.gru1(x)
        if printshapes:
            print(f"x.shape after gru1: {x.shape}")

        x = rearrange(x, '(b t) f c -> b t (f c)', b=b)
        x, _ = self.gru2(x)
        if printshapes:
            print(f"x.shape after gru2: {x.shape}")

        # 3. stack of 2 linear layer + layernorm + relu
        for i, module in enumerate(self.lin_list):
            x = module(x)
            if printshapes:
                print(f"x.shape after ll {i}: {x.shape}")
        
        return x



class conv2d_block(nn.Module):
    def __init__(self, chn_in, chn_out, kernel, stride):
        super().__init__()
        self.conv2d = nn.utils.weight_norm(
            nn.Conv2d(chn_in, chn_out, kernel, stride, bias=False)
        )
        
        self.norm = nn.InstanceNorm2d(chn_out, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.conv2d(x)
        #x = self.norm(x) # optional, weiss noch nicht was das bringt
        #x = self.act(x)
        x = F.relu(x) # relu oder act, nicht beide
        return x 

class linear_block(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()

        self.linear = nn.Linear(in_feat, out_feat)
        self.layer_norm = nn.LayerNorm(out_feat)

    def forward(self, x):
        x = self.linear(x)
        x = self.layer_norm(x)
        y = F.relu(x)
        return y 