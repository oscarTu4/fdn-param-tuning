import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from custom_encoder import Encoder
import audio_utility as util

# inspiriert von https://github.com/gdalsanto/diff-delay-net.git
class MultiLinearProjectionLayer(nn.Module):
    def __init__(self, features1, features2, num_out_params, chn_out, activation = None):
        super().__init__()
        self.linear1 = nn.Linear(features1, num_out_params)
        self.linear2 = nn.Linear(features2, chn_out)
        self.activation = activation

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.linear1(x)

        x = torch.transpose(x, 2, 1)
        x = self.linear2(x)

        # nonlinear activation
        if self.activation is not None:
            x = self.activation(x)
        return x

class SingleProjectionLayer(nn.Module):
    def __init__(self, in_feat, out_feat, activation = None):
        super().__init__()
        self.linear = nn.Linear(in_feat, out_feat)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)

        if self.activation is not None:
            x = self.activation(x)
        return x


# Herz vom Modell
class DiffFDN(nn.Module):
    def __init__(self, delay_lens, z, sr: int = 48000, ir_length: float = 1.,):
        super().__init__()

        self.delay_lens = delay_lens
        self.N = len(delay_lens)
        self.z = z
        
        self.ir_length = ir_length
        self.sr = sr
        
        self.num_U = self.N*self.N
        self.num_BC = self.N
        self.num_Gamma = self.N
        
        self.encoder = Encoder()#n_fft=512, overlap=0.5)
        
        # shape die aus dem encoder rauskommt. kann nach länge der ir dann noch optimiert werden. 
        # batch hier nicht wichtig, muss aber beachtet werden
        # wenn ein fehler wie "shapes cannot be multiplied" kommt, dann muss hier höchstwahrscheinlich die shape geändert werden
        shape = [84, 256] # [T, F]
        
        self.single_proj_U = MultiLinearProjectionLayer(shape[0], shape[1], self.N, self.N)
        
        
        self.proj_U = nn.ModuleList()
        self.ortho_force = nn.Sequential(Skew(), MatrixExponential())
        self.paraFiR = paraunitaryFIR()
        K = 4
        for _ in range(K):
            self.proj_U.append(
                nn.Sequential(
                    MultiLinearProjectionLayer(shape[0], shape[1], self.N, self.N),
                    Skew(),
                    MatrixExponential()
                )
            )
        
        self.proj_Gamma = MultiLinearProjectionLayer(
            shape[0],
            shape[1],
            num_out_params=1,
            chn_out=1,
        )
        
        #self.freqProject = nn.Linear(256, self.z.shape[-1])
        
        self.proj_BC = MultiLinearProjectionLayer(
            shape[0], 
            shape[1], 
            num_out_params=2, 
            chn_out=self.num_BC
        )
        
        
        
    ### implementierung von RIR2FDN bis 3.2
    ### gamma müsste gelernt werden damit Decay was interessanteres macht
    def forward(self, x, gamma):
        batch_size = x.size()[0]
        
        z_N = self.z.numel()
        
        x = self.encoder(x)

        BC = self.proj_BC(x)
        B, C = BC[:, 0, :], BC[:, 1, :]
        B = torch.complex(B, torch.zeros_like(B))
        C = torch.complex(C, torch.zeros_like(C))
        
        U_list = []
        for proj in self.proj_U:
            u = proj(x)
            u = torch.complex(u, torch.zeros_like(u))
            U_list.append(u)
        #print(f"U shape: {U.shape}")
        
        
        ### fixed gamma
        #gamma = 0.9999
        ### gamma via conditioning
        
        ### für inference/evaluation, dass gamma auch als skalar übergeben werden kann
        """if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma)

        if gamma.dim() == 0:
            gamma = gamma.unsqueeze(0)
        
        gamma = gamma.view(-1, 1)
        m = self.delay_lens.view(1, -1)
        Gamma = torch.diag_embed(gamma.view(-1,1) ** m)
        Gamma = Gamma.unsqueeze(1).expand(batch_size, z_N, self.N, self.N)"""
        
        ### gamma via condition + randomizing um filter decay zu simulieren
        """eps = 1e-3 * torch.randn(batch_size, self.N)
        eps = F.avg_pool1d(eps.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
        gamma = gamma.view(-1,1) * (1 + eps)
        m = self.delay_lens.view(1, -1)
        Gamma = torch.diag_embed(gamma ** m)
        Gamma = Gamma[:, None].expand(batch_size, z_N, self.N, self.N)"""
        
        ### gamma via projection, könnte ansatz sein um gamma learnable zu machen
        gamma = self.proj_Gamma(x)
        print(f"gamma: {gamma}")

        Gamma = torch.diag_embed(gamma**self.delay_lens)
        
        #Gamma = torch.diag_embed(gamma ** self.delay_lens)

        #######
        Gamma = torch.complex(Gamma, torch.zeros_like(Gamma))
        
        # H(z) = T (z)c⊤ I − U (z)Γ(z)Dm(z)]−1U (z)Γ(z)b + D(z)
        B = B.unsqueeze(-1).unsqueeze(1).expand(-1, z_N, -1, -1)
        C = C.unsqueeze(1).unsqueeze(2).expand(-1, z_N, -1, -1)
        
        #print(B)
        #print(C)
        
        I = torch.eye(self.N, dtype=Gamma.dtype).unsqueeze(0).unsqueeze(0)
        I = I.expand(batch_size, z_N, self.N, self.N)
        
        #D_m = torch.diag_embed(z.view(1, F, 1) ** (-self.delay_lens.view(1, 1, self.N)))   ### chatgpt sag -m, checke nicht ganz warum
        #D_m = D_m.expand(batch_size, F, self.N, self.N)
        
        U_z = self.paraFiR(U_list, self.z, self.delay_lens)
        
        dtype = torch.complex64
        U_z = U_z.to(dtype)
        Gamma = Gamma.to(dtype)
        I = I.to(dtype)
        
        Klammer = torch.linalg.inv(I - U_z @ Gamma)# @ D_m)
        cKlammer = C @ Klammer
        H1 = cKlammer @ U_z
        H2 = Gamma @ B
        H = H1 @ H2 # Übertragungsfunktion in Frequenzdarstellung
        H  = H.squeeze(-1).squeeze(-1)
        ir = torch.fft.irfft(H, n=int(self.ir_length*self.sr))

        return ir.unsqueeze(1), U_z#, H
        
        # ARPnet Formel
        # H(z)=T(z)*b@(D-A)@c  #A=U@gamma
        """gamma = 0.9999
        gamma = torch.diag_embed(gamma**self.delay_lens)
        gamma = torch.complex(gamma, torch.zeros_like(gamma))
        U = self.single_proj_U(x)
        U = torch.complex(U, torch.zeros_like(U))
        A = torch.matmul(U, gamma).unsqueeze(1)
        
        D = torch.diag_embed(z.unsqueeze(-1)  ** (self.delay_lens))
        D = D.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        Klammer = torch.linalg.inv(D - A)
        B = B.unsqueeze(1).unsqueeze(-1)
        B = B.expand(-1, Klammer.shape[1], -1, -1)
        
        C = C.unsqueeze(1).unsqueeze(2)
        C = C.expand(-1, Klammer.shape[1], -1, -1)
        
        Db = Klammer @ B
        H = (C @ Db).squeeze(-1).squeeze(-1)
        
        ir = torch.fft.irfft(H, n=int(self.ir_length*self.sr))
        return ir.unsqueeze(1), U"""

class paraunitaryFIR(nn.Module):
    ### nach RIR2FDN
    def forward(self, U_list, z, m):
        batch_size, N, _ = U_list[0].shape
        F = z.numel()
        U_z = torch.eye(N, dtype=z.dtype)
        U_z = U_z.unsqueeze(0).unsqueeze(0)
        U_z = U_z.expand(batch_size, F, N, N)
        
        D_m = torch.diag_embed(z.view(1, F, 1) ** (-m.view(1, 1, N)))   ### chatgpt sag -m, checke nicht ganz warum aber funktioniert wohl
        D_m = D_m.expand(batch_size, F, N, N)
        
        for U in U_list:
            U = U.unsqueeze(1).expand(batch_size, F, N, N)
            U_z = U_z @ D_m @ U
        
        return U_z

class Skew(nn.Module):
    def forward(self, X):
        X = X.triu(1)
        return X - X.transpose(-1, -2)

class MatrixExponential(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)