import torch
from torchaudio import transforms
import numpy as np

# adaptiert von https://github.com/gdalsanto/diff-delay-net.git
def normalize(x):
    ''' normalize energy of x to 1 '''
    energy = (x**2).sum(dim=-1, keepdim=True)
    return x / torch.sqrt(energy + 1e-8)

def pad_crop(ir, sr, target_length):
    target_samples = int(target_length*sr)
    if ir.shape[-1] < target_samples:
        # pad
        #print("padding")
        missing_zeros = torch.zeros((ir.shape[0], target_samples - ir.shape[-1]))
        return torch.cat((ir, missing_zeros), dim=-1)
    else:# ir.shape[-1] >= target_samples:
        # crop
        #print("cropping")
        return ir[..., :target_samples]
    
# übernommen von https://github.com/gdalsanto/diff-delay-net.git
def get_frequency_samples(num):
    device = device = "cuda" if torch.cuda.is_available() else "cpu"

    angle = torch.linspace(0, 1-1/num, num, device=device)
    abs = torch.ones(num, device=device)
    return torch.polar(abs, angle * np.pi)


### folgender Code von Stable Audio übernommen
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many
import torch.nn as nn
from typing import Optional, Union, Callable, TypeVar, Tuple
from torch import Tensor
import math
from inspect import isfunction

T = TypeVar("T")

def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val: Optional[T]) -> T:
    return val is not None

def closest_power_2(x: float) -> int:
    exponent = math.log2(x)
    distance_fn = lambda z: abs(x - 2 ** z)  # noqa
    exponent_closest = min((math.floor(exponent), math.ceil(exponent)), key=distance_fn)
    return 2 ** int(exponent_closest)

class STFT(nn.Module):
    """Helper for torch stft and istft"""

    def __init__(
        self,
        num_fft: int = 1023,
        hop_length: int = 256,
        window_length: Optional[int] = None,
        length: Optional[int] = None,
        use_complex: bool = False,
    ):
        super().__init__()
        self.num_fft = num_fft
        self.hop_length = default(hop_length, math.floor(num_fft // 4))
        self.window_length = default(window_length, num_fft)
        self.length = length
        self.register_buffer("window", torch.hann_window(self.window_length))
        self.use_complex = use_complex

    def encode(self, wave: Tensor) -> Tuple[Tensor, Tensor]:
        b = wave.shape[0]
        wave = rearrange(wave, "b c t -> (b c) t")

        stft = torch.stft(
            wave,
            n_fft=self.num_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=self.window,  # type: ignore
            return_complex=True,
            normalized=True,
        )

        if self.use_complex:
            # Returns real and imaginary
            stft_a, stft_b = stft.real, stft.imag
        else:
            # Returns magnitude and phase matrices
            magnitude, phase = torch.abs(stft), torch.angle(stft)
            stft_a, stft_b = magnitude, phase

        return rearrange_many((stft_a, stft_b), "(b c) f l -> b c f l", b=b)