# mit tweaks kopiert von https://github.com/gdalsanto/diff-delay-net.git
import torch
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange
from nnAudio import features
from utils.utility import *
#from torchaudio.models import Conformer
from ConformerBlock import Conformer

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
        
        #base_chn = 32
        #chn_multiplier = [1, 2, 4] 
        #kernel = [(7,7), (5,5), (5,5)]
        #strides = [(2,4), (4,1), (2,2)]
        #channels_in = [513, 384, 256]
        channels_in = [1, 513, 384, 256, 128]
        channels_out = [513, 384, 256, 128, 64]
        kernel = [(7,5), (5,5), (5,5), (5,5), (5,5)]
        strides = [(2,2), (2,1), (2,2), (2,2), (2,1)]
        
        self.conv_list = nn.ModuleList([])
        for i in range(len(channels_in)):
            self.conv_list.append(
                #conv1d_block(channels_in[i], channels_out[i], kernel[i], strides[i])
                conv2d_block(channels_in[i], channels_out[i], kernel[i], strides[i])
            )
        
        c_o = 832
        dim = 256
        self.conf_lin_in = nn.Sequential(
            nn.Linear(c_o, dim),
            nn.LayerNorm(dim)
        )
        self.conformer = Conformer(
            input_dim=dim,
            num_heads=8,
            ffn_dim=4*dim,
            num_layers=4,
            depthwise_conv_kernel_size=15,
            dropout=0.0
        )
        
        self.lin_depth = 2
        self.lin_list = nn.ModuleList([])
        for i in range(self.lin_depth): 
            self.lin_list.append(nn.ModuleList(
                [linear_block(256, 256)]
            ))
    
    def forward(self, x):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"moving x to {self.device}")
        x = x.to(self.device)
        x = torch.log(self.stft(x) + 1e-7)
        # add channel dimension 
        x = torch.unsqueeze(x, 1)

        for i, module in enumerate(self.conv_list):
            x = module(x)
        
        x = rearrange(x, 'b c f t -> b t (c f)')
        #x = x.permute(0, 2, 1)
        x = self.conf_lin_in(x)

        B, T, _ = x.shape
        lengths = torch.full((B,), T, device=x.device, dtype=torch.long)
        x, _ = self.conformer(x, lengths)

        # 3. stack of 2 linear layer + layernorm + relu
        for i, module in enumerate(self.lin_list):
            x = module[0](x)
        
        return x



class conv2d_block(nn.Module):
    def __init__(self, chn_in, chn_out, kernel, stride):
        super().__init__()
        self.conv2d = nn.utils.weight_norm(
            nn.Conv2d(chn_in, chn_out, kernel, stride, bias=False)
        )

    def forward(self, x):
        x = self.conv2d(x)
        x = F.relu(x)
        return x 

class conv1d_block(nn.Module):
    def __init__(self, chn_in, chn_out, kernel, stride):
        super().__init__()
        self.conv1d = nn.utils.weight_norm(
            nn.Conv1d(chn_in, chn_out, kernel, stride, bias=False)
        )
    
    def forward(self, x):
        x = self.conv1d(x)
        x = F.relu(x)
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