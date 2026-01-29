import torch
import time
from losses import *
from tqdm import tqdm

import time
import os
import argparse
from utility import * 
from audio_utility import get_frequency_samples
from dataset import load_dataset
from fdn import DiffFDN
import pandas as pd
import numpy as np
import json

import pyfar as pf
from torchaudio import transforms
import random
import torchaudio
import soundfile as sf
from matplotlib import pyplot as plt
import shutil

class Trainer:
    def __init__(self, net, args, train_dataset, valid_dataset):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = net
        self.net = self.net.to(device)

        self.max_epochs = args.max_epochs
        self.patience = 5
        self.early_stop = 0
        self.train_dir = args.train_dir
        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        
        self.steps = 0 # falls checkpoint geladen wird muss das hier geändert werden
        self.scheduler_steps = args.scheduler_steps

        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) 
        #self.criterion = STFTLoss(sr=args.samplerate).to(device)
        self.criterion = mse_loss()
        ### TODO spectral+sparsity loss implementieren wenn wir nach RIR2FDN vorgehen
        #self.criterion = [STFTLoss(sr=args.samplerate), sparsity_loss()]
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 500, gamma = 10**(-0.2))  # step_size war 50000 das aber vlt sehr sehr hoch, müssen wir testen

        #self.normalize() # normalize sollte denke angepasst werden, erstmal rausgenommen damit es läuft. weiss nicht wie wichtig das nicht
        
        self.alpha = 2  # temporal loss scaling factor
        
        # for eval
        self.samplerate = args.samplerate
        self.ir_length = args.ir_length
    
    def train(self):
        self.train_loss, self.valid_loss = [], []

        for epoch in range(self.max_epochs):
            st_epoch = time.time()

            # training
            epoch_loss = 0
            pbar = tqdm(self.train_dataset, desc=f"Training | Epoch {epoch}/{self.max_epochs}")
            for _, ir in enumerate(pbar):
                loss = self.train_step(ir)
                epoch_loss += loss
                
                # für progressbar
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "loss": f"{loss}",
                    "lr": f"{lr}"
                })
                
                #exit()
            
            self.scheduler.step()   # lr anpassung

            self.train_loss.append(epoch_loss/len(self.train_dataset))

            # validation
            epoch_loss = 0
            pbar = tqdm(self.valid_dataset, desc="Validation")
            for _, data in enumerate(pbar):
                loss = self.valid_step(data)
                epoch_loss += loss
                
                # für progressbar
                pbar.set_postfix({
                    "loss": f"{loss}",
                })
            
            self.valid_loss.append(epoch_loss/len(self.valid_dataset))
            et_epoch = time.time()

            self.print_results(epoch, et_epoch-st_epoch)
            if epoch % 10 == 0:
                self.save_model(epoch)
            
            # loss plotten/speicher. kann auch öfter/seltener gemacht werden (mit lightning geht das auch gut)
            save_loss(self.train_loss, self.valid_loss, self.train_dir, save_plot=True)
            
            ### evaluate on epoch end
            self.evaluate(epoch=epoch)

    def train_step(self, x):
        self.optimizer.zero_grad()
        gt = x[0].clone()
        #gamma = util.gamma_batched(x)
        input = x[0]
        gamma = x[1]
        
        y, U = self.net(input, gamma)
        
        #print(f"y shape: {y.shape}")
        #print(f"gt shape: {gt.shape}")
        loss = self.criterion(y, gt)
        #loss = self.criterion[0](y, gt) + self.alpha*self.criterion[1](self.net.ortho_force(U))
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def valid_step(self, x):
        # batch processing
        self.optimizer.zero_grad()
        gt = x[0].clone()
        input = x[0]
        gamma = x[1]
        
        y, U = self.net(input, gamma) # z vielleicht auch aus ir selbst holen? so wie es jetzt ist könnte falsch sein
        #loss = self.criterion[0](y, gt) + self.alpha*self.criterion[1](self.net.ortho_force(U))
        loss = self.criterion(y, gt)
        return loss.item()

    def print_results(self, e, e_time):
        print(get_str_results(epoch=e, 
                            train_loss=self.train_loss, 
                            valid_loss=self.valid_loss, 
                            time=e_time))

    def save_model(self, e):
        dir_path = os.path.join(self.train_dir, 'checkpoints')
        # create checkpoint folder 
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)  
        # save model 
        torch.save(
            self.net.state_dict(), 
            os.path.join(dir_path, 'model_e' + str(e) + '.pt'))
    
    def evaluate(self, epoch):
        random.seed()
        eval_path = "/Users/oscar/documents/Uni/Audiokommunikation/3. Semester/DLA/Impulse Responses/eval"
        eval_paths = [f for f in os.listdir(eval_path) if f.endswith(".wav")]
        eval_files = random.sample(eval_paths, 5)
        with torch.no_grad():
            for idx, eval_file in enumerate(eval_files):
                eval_filepath = os.path.join(eval_path, eval_file)
                eval_ir, sr = torchaudio.load(eval_filepath)   
                
                if sr != self.samplerate:
                    tf = transforms.Resample(sr, self.samplerate)
                    eval_ir = tf(eval_ir)
                    sr = self.samplerate
                
                t60 = eval_ir.size()[-1]
                gamma = 10**(-3/t60)
                z = get_frequency_samples(eval_ir.size()[-1])

                if eval_ir.shape[0] != 1:
                    eval_ir = eval_ir.mean(dim=0, keepdim=True)

                eval_ir = util.pad_crop(eval_ir, sr, self.ir_length)
                if eval_ir.ndim == 2:        # [C, T] zu [B, C, T]
                    eval_ir = eval_ir.unsqueeze(0)

                pred, _ = self.net(eval_ir, gamma)#, z)

                save_path = f"{self.train_dir}/evaluation" #/epoch{epoch}"
                os.makedirs(save_path, exist_ok=True)
            
                # plot
                #pred_np = util.normalize(pred)
                pred_np = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
                #eval_np = eval_ir.squeeze(0).detach().cpu().numpy()
                
                #pred_real = np.abs(pred_np).astype(np.float32)

                #sf.write(f"{save_path}/{idx}-pred.wav", pred_real, self.samplerate)
                #sf.write(f"{save_path}/{idx}-{batch}.wav", eval_np, self.samplerate)
                #shutil.copy(eval_filepath, save_path)

                times = np.zeros(len(pred_np))
                eval_sig = pf.Signal([eval_ir.flatten(),times.flatten()],sampling_rate=self.samplerate, is_complex=True)
                pred_sig = pf.Signal([pred_np.flatten(),times.flatten()],sampling_rate=self.samplerate, is_complex=True)

                plt.figure()
                pf.plot.time_freq(eval_sig, label="eval", alpha=0.3)
                pf.plot.time_freq(pred_sig, label="pred", alpha=0.7)
                plt.legend()
                plt.savefig(os.path.join(save_path, f"e{epoch}-{idx}-plot.pdf"))
                plt.close()
            
        random.seed()


def main(args):
    train_dataset, valid_dataset = load_dataset(args)
    
    # init neural net
    filepath = 'Params/'
    N = args.N
    filename = 'param' + '_N' + str(N) + '_d' + str(args.delay_set)

    df = pd.read_csv(filepath+filename+'.csv', delimiter=';', nrows=N*N, dtype={'A':np.float32,'m':'Int32'})
    delay_lens = torch.from_numpy(df['m'][:N].to_numpy())
    
    z = get_frequency_samples(int(args.ir_length*args.samplerate))
    net = DiffFDN(delay_lens, z, args.samplerate, args.ir_length)
    #net.apply(weights_init_normal) # weiss nich ob wir das hier brauchen

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"batch size = {args.batch_size} | trainable params = {(trainable_params/1000000):.3f}M")
    print(f"trainset size: {len(train_dataset.dataset)} | valset size: {len(valid_dataset.dataset)}")
    
    trainer = Trainer(net, args, train_dataset, valid_dataset)
    trainer.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--samplerate', type=int, default=48000, help ='sample rate')
    
    # dataset 
    parser.add_argument('--path_to_IRs', type=str, default="/Users/oscar/documents/Uni/Audiokommunikation/3. Semester/DLA/Impulse Responses/train")
    parser.add_argument('--split', type=float, default=0.8, help='training / validation split')
    parser.add_argument('--shuffle', default=True, help='if true, shuffle the data in the dataset at every epoch')
    parser.add_argument('--ir_length', type=float, default=2., help="wenn != None werden alle IRs auf diese Länge gebracht. ist eig pflicht")
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--delay_set', type=int, default=1)
    
    # training
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=30,  help='maximum number of training epochs')
    parser.add_argument('--log_epochs', action='store_true', help='Store met parameters at every epoch')
    
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--scheduler_steps', default=50)
    #parser.add_argument('--clip_max_norm', default=10)
    args = parser.parse_args()

    args.train_dir = os.path.join('outputs', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.train_dir, exist_ok=True)
    
    # save arguments
    with open(os.path.join(args.train_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    
    main(args)