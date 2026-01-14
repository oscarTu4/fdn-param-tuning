import torch
import torchaudio
from torchaudio import transforms
import numpy as np 
import torch.utils.data as data
import os
import tqdm
import audio_utility as util

def loadAllIRFromFolder(dir: str=None, targetSR: int = 48000, ir_length: float = 1.):
    IRs = {}
    
    for item in os.listdir(dir):
        abs_path = os.path.join(dir, item)
        
        if os.path.isdir(abs_path):
            print(f"loading dir {abs_path}")
            sub_irs = loadAllIRFromFolder(dir=abs_path, targetSR=targetSR, ir_length=ir_length)
            IRs.update(sub_irs)
        elif item.endswith(".wav"):
            ir, sr = torchaudio.load(abs_path)
            
            # sample rate bei bedarf anpassen
            if sr != targetSR:
                resample_tf = transforms.Resample(sr, targetSR)
                ir = resample_tf(ir)
                sr = targetSR
            
            # erstmal alles mono damit es läuft 
            # kann stereo o.ä. werden, weiss noch nicht obs flexibel geht, tensor shape sieht dementsprechend anders aus bei jedem Datenpunkt
            if ir.shape[0] != 1:
                ir = ir.mean(dim=0, keepdim=True)
            
            # fixe länge wichtig damit das Modell läuft
            ir = util.pad_crop(ir, sr, ir_length)
            
            ir = util.normalize(ir)
            
            label = item.split(".wav")[0]
            IRs[label] = ir
    
    return IRs
        

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_IRs: str, samplerate: int = 48000, ir_length: float = 1.):
        assert path_to_IRs != None, "path_to_IRs must not be emtpy"
        
        self.IRs = loadAllIRFromFolder(path_to_IRs, samplerate, ir_length)
        
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
    dataset = Dataset(path_to_IRs=args.path_to_IRs, samplerate = args.samplerate, ir_length = args.ir_length)
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