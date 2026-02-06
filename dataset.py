import torch
import torchaudio
from torchaudio import transforms
import numpy as np 
import torch.utils.data as data
import os
from glob import glob
from tqdm import tqdm
from utils.processing import *
from utils.utility import *
import random

def loadAllIRFromFolder(dir: str=None, targetSR: int = 48000, ir_length: float = 1.):
    IRs = {}
    pathlist = [y for x in os.walk(dir) for y in glob(os.path.join(x[0], '*.wav'))]
    random.shuffle(pathlist)

    for item in tqdm(pathlist):
        try:
            ir, sr = torchaudio.load(item)

            # sample rate bei bedarf anpassen
            if sr != targetSR:
                resample_tf = transforms.Resample(sr, targetSR)
                ir = resample_tf(ir)
                sr = targetSR
            
            ir = pad_crop(ir, sr, ir_length)
            
            # erstmal alles mono damit es läuft
            #if ir.shape[0] != 1:
            #    ir = ir[0, :].numpy()
            if ir.ndim == 2:
                ir = ir[0]
            ir = ir.squeeze().numpy()
            
            # --------------- PREPROCESSING --------------- #
            # remove onset 
            if ir.shape[-1] < 256:
                continue
            onset = find_onset(ir)
            #print(f"onset: {onset}")
            ir = np.pad(ir[onset:],(0, onset))
            # multply random gain to direct sound 
            ir = augment_direct_gain(ir, sr=sr)
            # nornalize 
            ir = normalize_energy(ir)
            # --------------------------------------------- #
            
            label = item.split(".wav")[0]
            #IRs[label] = ir
            IRs[label] = torch.from_numpy(ir)   # hässlich aber nan fehler beim andern dataset, onset/normalize und so nur für numpy
        except Exception as e:
            print(e)
            #raise e
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
    dataset = Dataset(
        path_to_IRs=args.path_to_IRs, 
        samplerate = args.samplerate, 
        ir_length = args.rir_length,
    )
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