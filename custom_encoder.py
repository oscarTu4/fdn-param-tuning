# mit tweaks kopiert von https://github.com/gdalsanto/diff-delay-net.git
import torch
import torch.nn as nn 
import torch.nn.functional as F
from einops import rearrange
from nnAudio import features

class Encoder(nn.Module):
    def __init__(self, n_fft=1024, sr=48000, overlap=0.875):
        super().__init__()

        self.hop_length = int(n_fft*(1-overlap))
        self.stft = features.stft.STFT(
            n_fft = n_fft,
            hop_length = self.hop_length,
            window = 'hann',
            freq_scale = 'log',
            sr = sr,
            fmin = 20,
            fmax = sr // 2,
            output_format = 'Magnitude',
            verbose=False
        )
        self.n_fft = n_fft
        self.sr = sr
        self.overlap = overlap
        
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
        
        """self.gru1 = nn.GRU(input_size=128, num_layers=2, hidden_size=64, 
            batch_first = True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=7168, num_layers=1, hidden_size=128,
            batch_first = True, bidirectional=True)"""
        
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

        self.gru1_norm = nn.LayerNorm(64*2)   # bi-directional
        self.gru2_norm = nn.LayerNorm(128*2)

        self.lin_depth = 2
        in_feat, out_feat = 256, 256
        self.lin_list = nn.ModuleList([])
        for i in range(self.lin_depth):
            self.lin_list.append(
                linear_block(in_feat, out_feat)
            )
    
    def forward(self, x):
        #print(f"x shape at start of encoder forward: {x.shape}")
        b = x.shape[0]
        # convert to log-freq log-mag stft 
        x = torch.log(self.stft(x) + 1e-7)

        #x = torch.clamp(x, min=-12.0, max=2.0)
        
        # add channel dimension 
        x = torch.unsqueeze(x, 1)

        for i, module in enumerate(self.conv_list):
            x = module(x)
        
        x = rearrange(x, 'b c f t -> (b t) f c')
        x, _ = self.gru1(x)
        #x = self.gru1_norm(x)

        x = rearrange(x, '(b t) f c -> b t (f c)', b=b)
        x, _ = self.gru2(x)
        #x = self.gru2_norm(x)

        # 3. stack of 2 linear layer + layernorm + relu
        for i, module in enumerate(self.lin_list):
            x = module(x)
        
        return x



class conv2d_block(nn.Module):
    def __init__(self, chn_in, chn_out, kernel, stride):
        super().__init__()
        self.conv2d = nn.utils.weight_norm(
            nn.Conv2d(chn_in, chn_out, kernel, stride, bias=False)
        )
        
        self.norm = nn.InstanceNorm2d(chn_out, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        
        #nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity="leaky_relu")
        #self.conv2d.weight.data *= 0.1

    def forward(self, x):
        x = self.conv2d(x)
        #x = self.norm(x)
        #x = self.act(x)
        x = F.relu(x) # relu or act, nicht beide
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