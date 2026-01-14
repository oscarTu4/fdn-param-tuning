import torch
from torchaudio import transforms

# adaptiert von https://github.com/gdalsanto/diff-delay-net.git
def normalize(x):
    ''' normalize energy of x to 1 '''
    energy = (x**2).sum(dim=-1, keepdim=True)
    return x / torch.sqrt(energy + 1e-8)

def pad_crop(ir, sr, target_length):
    target_samples = int(target_length*sr)
    if ir.shape[-1] < target_samples:
        # pad
        missing_zeros = torch.zeros((ir.shape[0], target_samples - ir.shape[-1]))
        return torch.cat((ir, missing_zeros), dim=-1)
    elif ir.shape[-1] > target_samples:
        # crop
        return ir[..., :target_samples]