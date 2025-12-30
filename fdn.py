import pyfar as pf
import soundfile as sf
import pandas as pd

import torch 
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from custom_encoder import Encoder

# inspiriert von https://github.com/gdalsanto/diff-delay-net.git
class DoubleProjectionLayer(nn.Module):
    def __init__(self, in_features1, in_features2, activation = None):
        super().__init__()
        self.linear1 = nn.Linear(in_features1[0], in_features1[1])
        self.linear2 = nn.Linear(in_features2[0], in_features2[1])
        self.activation = activation

    def forward(self, x):
        #x = torch.transpose(x, 2, 1)
        x = self.linear1(x)
        x = torch.transpose(x, 2, 1)
        y = self.linear2(x)
        # nonlinear activation
        if self.activation is not None:
            y = self.activation(y)
        return y

class SingleProjectionLayer(nn.Module):
    def __init__(self, in_feat, out_feat, activation = None):
        super().__init__()
        self.linear = nn.Linear(in_feat, out_feat)
        self.activation = activation
        
        #nn.init.zeros_(self.linear.weight)
        #nn.init.zeros_(self.linear.bias)

    def forward(self, x):

        x = self.linear(x)

        if self.activation is not None:
            y = self.activation(x)
        return y

# Herz vom Modell
# TODO eventuell Encoder auch hier mit reinschreiben, bzw. in diese Datei
class DiffFDN(nn.Module):
    def __init__(self, delay_lens, sr: int = 48000, ir_length: float = 1.):
        super().__init__()

        self.delay_lens = delay_lens
        self.ir_length = ir_length
        self.N = len(delay_lens)
        
        num_A = self.N*self.N
        num_B = self.N
        num_C = self.N
        
        self.fdn = FDN(sr=sr, ir_length=ir_length)
        self.encoder = Encoder()
        
        features = 256 # kommt so aus encoder raus. kann angepasst werden, muss aber analog zu encoder passieren
        self.proj_A = SingleProjectionLayer(features, num_A, nn.Tanh())
        self.proj_B = SingleProjectionLayer(features, num_B, nn.Tanh())
        self.proj_C = SingleProjectionLayer(features, num_C, nn.Tanh())
        
    def forward(self, x):
        
        x = self.encoder(x)
        
        x = x.mean(dim=1)
        A = self.proj_A(x)
        A = A.view(-1, self.N, self.N)
        #A = A*0.000001
        
        B = self.proj_B(x)
        
        C = self.proj_C(x)
        
        ### das hier ist hässlich und langsam 
        ### fdn muss noch mit batched tensoren funktionieren (oder ist das das block fdn?)
        ### das wichtig
        outputs = []
        for i in range(A.shape[0]):
            y_i = self.fdn(A[i], B[i], C[i], self.delay_lens, self.N)
            y_i = y_i.transpose(0, 1)
            outputs.append(y_i)
        
        y = torch.stack(outputs, dim=0)
        return y

class Skew(nn.Module):
    def forward(self, X):
        X = X.triu(1)
        return X - X.transpose(-1, -2)

### Simons Code als nn Module, numpy musste mit torch ersetzt werden
class FDN(nn.Module):
    def __init__(self, sr=48000, ir_length=1.):
        super().__init__()
        self.skew = Skew()
        self.fs = sr
        self.t60 = ir_length
    
    def forward(self, A, B, C, delay_lens, N):
        assert A.device == B.device and B.device == C.device, "wie zum fick sind die devices unterschiedlich"
        device = A.device
        # Force correct shapes
        B = B.view(N, 1)      # column vector
        C = C.view(1, N)      # row vector
        
        # Delayline-Puffer (Liste von N Arrays)
        delay_lines = [
            torch.zeros(delay_lens[i], device=device, dtype=A.dtype)
            for i in range(N)
        ]
        
        ir_len = int(self.t60*self.fs)
        g = 10**(-3/(self.fs*self.t60))
        
        delay_tensor = torch.tensor(delay_lens, device=device, dtype=A.dtype)
        G = torch.diag(g ** delay_tensor)
        #G = torch.diag(g**delay_lens).to(A.dtype).to(device)

        A_g = torch.linalg.matrix_exp(self.skew(A)) @ G  # Feedback Matrix mit Dämpfung 
        #print(f"A_g: {A_g}")
        A_g = A_g.to(device)

        impulse = torch.zeros((ir_len, C.shape[0]), device=device, dtype=A.dtype)
        impulse[0, :] = 1
        #print(f"impulse: {impulse}")

        output  = torch.zeros((ir_len, 1), device=device, dtype=A.dtype)
        #print(f"output: {output}")

        # Pointer zum Lesen und Schreiben
        write_ptr = torch.zeros(N, dtype=torch.long, device=device)
        read_ptr = torch.zeros(N, dtype=torch.long, device=device)

        for n in range(ir_len):
            d = torch.zeros((N,1), dtype=C.dtype, device=device)
            
            for i in range(N):
                d[i,0] = delay_lines[i][read_ptr[i]]
            output[n,0] = C @ d
            next_input = A_g @ d + B * impulse[n,0]

            for i in range(N):
                delay_lines[i][write_ptr[i]] = next_input[i,0]

            for i in range(N):
                write_ptr[i] = (write_ptr[i] + 1) % delay_lens[i]
                read_ptr[i]  = (read_ptr[i] + 1) % delay_lens[i]

        return output

### das hier checke ich nicht wirklich, darum hab ich erstmal das alte fdn genommen um die Pipeline aufzubauen
class FeedbackDelay:

    def __init__(self,max_block_size,delays):
        self.delays = delays 
        self.N = len(delays)
        self.values = np.zeros((np.max(delays)+max_block_size,self.N))
        self.pointers = np.zeros(self.N,dtype=int)
    
    def getIndex(self,blk_size):

        row_idx = self.pointers + np.arange(blk_size)[:, None]
        # wrap around when  row_idx > delay line length 
        row_idx = row_idx % self.delays 

        col_idx = np.arange(self.N)[None, :] 
        
        return row_idx,col_idx
    
    def getValues(self,blockSize):
        return self.values[self.getIndex(blockSize)]
        
    # def mod_delay(self,idxs):
    #     idxs = idxs - (idxs > self.delays) * self.delays
    #     return idxs
    
    def setValues(self,val):
        blk_size = val.shape[0]
        self.values[self.getIndex(blk_size)] = val
    
    def next(self,blockSize): # Pointer weiterschieben Ring Buffer 
        self.pointers = (self.pointers + blockSize) % self.delays


def compute_FDN_blk(input,delays,feedbackMatrix,inputGains,outputGains):

    maxBlockSize = 2**12
    blk_size = min([min(delays), maxBlockSize])

    DelayFilters = FeedbackDelay(maxBlockSize,delays)

    input_len = input.shape[0]
    output = np.zeros((input_len,outputGains.shape[0]))

    blk_start = 0

    while blk_start < input_len:
        if(blk_start + blk_size < input_len):
            blk_pos = np.arange(blk_start,blk_start+blk_size)
        else: # last block 
            blk_pos = np.arange(blk_start,input_len-blk_size)

        blk = input[blk_pos,:]

        if blk.shape[0] == 0:
            break

        # ... process block ...
        delay_output = DelayFilters.getValues(blk_size)

        feedback = delay_output @ feedbackMatrix.T

        delay_line_input = blk @ inputGains.T + feedback

        DelayFilters.setValues(delay_line_input)

        output[blk_pos, :] = delay_output @ outputGains.T

        DelayFilters.next(blk_size)

        blk_start += blk_size

    return output

if __name__ == "__main__":
    import numpy as np
    
    filepath = 'Params/'

    init = False
    N = 8 # number of delay lines [4, 8, 16, 32]
    delay_set = 1

    filename = 'param'

    if init == True:
        filename+='_init'

    filename+= '_N'+str(N)+'_d'+str(delay_set)

    df = pd.read_csv(filepath+filename+'.csv',
                    delimiter=';',
                    nrows=N*N,
                    dtype={'A':np.float32,'m':'Int32'})

    A = df['A'].to_numpy().reshape(N, N)
    B = df['B'][:N].to_numpy()
    C = df['C'][:N].to_numpy()
    delay_lens = df['m'][:N].to_numpy()

    B = B[:,np.newaxis]
    C = C[np.newaxis,:]
    
    output = fdn(delay_lens, N, A, B, C)
    sf.write("test.wav", output, 48000)
    
    fs = 48000
    times = np.zeros(len(output))

    for i in range(len(output)):
        times[i] = i/fs

    ir_gen = pf.Signal([output.flatten(),times.flatten()],sampling_rate=fs)
    print(ir_gen.time.shape)
