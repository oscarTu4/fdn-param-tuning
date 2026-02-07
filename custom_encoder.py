# mit tweaks kopiert von https://github.com/gdalsanto/diff-delay-net.git
import torch
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange
from nnAudio import features
from utils.utility import *
from torchaudio.models import Conformer

class CustomEncoder(nn.Module):
    def __init__(self, n_fft=1024, sr=48000, overlap=0.875):
        super().__init__()
        self.sr = sr
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hop_length = int(n_fft*(1-overlap))
        self.stft = features.stft.STFT(
            n_fft = n_fft,
            hop_length = hop_length,
            window = 'hann',
            freq_scale = 'log',
            sr = sr,
            fmin = 20,
            fmax = sr // 2,
            output_format = 'Magnitude',
            verbose=False
        ) #[b c f t]
        
        base_chn = 32
        chn_multiplier = [1, 2, 4] 
        kernel = [(7,7), (5,5), (5,5)]
        strides = [(2,4), (4,1), (2,2)]
        
        self.conv_list = nn.ModuleList([])
        for i in range(len(chn_multiplier)):
            if i == 0:
                chn_in = 1
            else:
                chn_in = base_chn*chn_multiplier[i-1]
            
            self.conv_list.append(
                conv2d_block(chn_in, base_chn*chn_multiplier[i], kernel[i], strides[i])
            )
        
        cf = 30*128
        dim = 256
        #self.conf_conv_in = nn.Conv1d(cf, dim, kernel_size=3, padding=1)
        self.conf_lin_in = nn.Linear(cf, dim)
        self.conformer = Conformer(
            input_dim=dim,
            num_heads=8,
            ffn_dim=4*dim,
            num_layers=4,
            depthwise_conv_kernel_size=15,
            dropout=0.1
        )
        
        self.lin_depth = 2
        self.lin_list = nn.ModuleList([])
        for i in range(self.lin_depth):
            self.lin_list.append(nn.ModuleList(
                [linear_block(256, 256)]
            ))
    
    def forward(self, x):
        printshapes = False
        x = x.to(self.device)
        b = x.shape[0]
        # convert to log-freq log-mag stft 
        x = torch.log(self.stft(x) + 1e-7)
        # add channel dimension 
        x = torch.unsqueeze(x, 1)
        
        if printshapes:
            print(f"x.shape pre downsample: {x.shape}")

        for i, module in enumerate(self.conv_list):
            x = module(x)
            if printshapes:
                print(f"x.shape after downsample {i+1}: {x.shape}")
        
        x = rearrange(x, 'b c f t -> b t (c f)')
        #x = rearrange(x, 'b c f t -> b (c f) t') # für conv in layer
        
        if printshapes:
            print(f"x.shape after rearrange: {x.shape}")
        #x = self.conf_conv_in(x)
        x = self.conf_lin_in(x)

        if printshapes:
            print(f"x.shape after conf_conv_in: {x.shape}")
        B, T, _ = x.shape
        lengths = torch.full((B,), T, device=x.device, dtype=torch.long)
        x, _ = self.conformer(x, lengths)
        if printshapes:
            print(f"x.shape after conformer: {x.shape}")

        # 3. stack of 2 linear layer + layernorm + relu
        for i, module in enumerate(self.lin_list):
            x = module[0](x)
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