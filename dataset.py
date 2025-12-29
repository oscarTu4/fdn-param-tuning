import torch
import torchaudio
from torchaudio import transforms
import numpy as np 
import torch.utils.data as data
import os
import tqdm

def pad_crop(ir, sr, target_length):
    target_samples = int(target_length*sr)
    if ir.shape[-1] < target_samples:
        # pad
        missing_zeros = torch.zeros((ir.shape[0], target_samples - ir.shape[-1]))
        return torch.cat((ir, missing_zeros), dim=-1)
    elif ir.shape[-1] > target_samples:
        # crop
        return ir[..., :target_samples]

# kopiert von https://github.com/gdalsanto/diff-delay-net.git
def normalize(x):
    ''' normalize energy of x to 1 '''
    energy = (x**2).sum(dim=-1, keepdim=True)
    return x / torch.sqrt(energy + 1e-8)

def loadAllIRFromFolder(dir: str=None, targetSR: int = 48000):
    
    IRs = {}
    
    for item in os.listdir(dir):
        abs_path = os.path.join(dir, item)
        
        if os.path.isdir(abs_path):
            print(f"found dir, entering dir {abs_path}")
            sub_irs = loadAllIRFromFolder(dir=abs_path, targetSR=targetSR)
            IRs.update(sub_irs)
        elif item.endswith(".wav"):
            ir, sr = torchaudio.load(abs_path)
            
            # sample rate bei bedarf anpassen
            if sr != targetSR:
                sr = transforms.Resample(sr, targetSR)
            
            # anzahl kanäle muss homogenisiert werden damit torch damit arbeiten kann. hier erstmal mono, 
            # kann stereo o.ä. werden, modell müsste dafür aber angepasst werden
            if ir.shape[0] != 1:
                ir = ir.mean(dim=0, keepdim=True)
            
            # alle files auf die gleiche Länge gebracht werden, damit torch damit arbeiten kann. müsste(?) gleich t60*fs sein
            # target_length erstmal auf 0.2 (sekunden) zum testen, kann nach bedarf angepasst werden
            target_length = 0.2
            ir = pad_crop(ir, sr, target_length)
            
            ir = normalize(ir)
            
            label = item.split(".wav")[0]
            
            IRs[label] = ir
    
    return IRs
        

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_IRs: str):
        assert path_to_IRs != None, "path_to_IRs must not be emtpy"
        
        self.IRs = loadAllIRFromFolder(path_to_IRs)
        #print(len(self.IRs))
        #print(self.IRs.keys())
        
    def __len__(self):
        return len(self.IRs)
    
    def __getitem__(self, index):
        k = list(self.IRs.keys())[index]
        v = self.IRs[k]
        return v

def split_dataset(dataset, split):
    
    train_set_size = int(len(dataset) * split)
    valid_set_size = len(dataset) - train_set_size
    generator = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=generator)

    return train_set, valid_set

def load_dataset(args):
    dataset = Dataset(path_to_IRs=args.path_to_IRs)
    # split data into training and validation set 
    train_set, valid_set = split_dataset(dataset, args.split)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        generator=torch.Generator(device=device),
        drop_last = True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_set, 
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        generator=torch.Generator(device=device),
        drop_last = True
    )

    
    return train_loader, valid_loader 