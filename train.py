import torch
import time
from losses import *
from model import *
from diff_dsp import *
from tqdm import tqdm
from dataset_diffnet import rirDataset, split_dataset, get_dataloader
from dataset import load_dataset
from utils.logging import *
from utils.utility import * 

import time
import os
import argparse
import pandas as pd
import numpy as np
import json

import pyfar as pf
from torchaudio import transforms
import random
import torchaudio
from matplotlib import pyplot as plt

class Trainer:
    def __init__(self, net, args, train_dataset, valid_dataset):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = net
        self.net = self.net.to(device)

        self.max_epochs = args.max_epochs
        self.train_dir = args.train_dir
        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        
        self.steps = 0 # falls checkpoint geladen wird muss das hier geändert werden
        self.scheduler_steps = args.scheduler_steps

        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) 
        self.criterion = MSSpectralLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 50000, gamma = 10**(-0.2))

        self.base_lr = args.lr
        self.warmup_steps = args.warmup_steps

        
        # for eval
        self.samplerate = args.samplerate
        self.rir_length = args.rir_length
        self.x = get_frequency_samples(args.num//2+1)
        self.test_batch = next(iter(valid_dataset))
    
    def train(self):
        write_audio(self.test_batch[0,:], 
                os.path.join(self.train_dir, 'audio_output'),
                'target_ir.wav')
        self.train_loss, self.valid_loss = [], []
        plot_out = os.path.join(self.train_dir, 'plots')
        os.makedirs(plot_out, exist_ok=True)

        for epoch in range(self.max_epochs):
            st_epoch = time.time()
            epoch_loss, grad_norm = 0, 0
            
            # ----------- TRAINING ----------- # 
            pbar = tqdm(self.train_dataset, desc=f"Training | Epoch {epoch}/{self.max_epochs}")
            for _, input in enumerate(pbar):
                target = input.clone()
                self.optimizer.zero_grad()
                estimate, _, _ = self.net(input, self.x)  # get estimate

                loss = self.criterion(estimate, target) # compute loss
                epoch_loss += loss.item()
                loss.backward()
                # clip gridients
                grad_norm += nn.utils.clip_grad_norm_(self.net.parameters(), args.clip_max_norm)

                # LR warmup
                if self.steps < self.warmup_steps:
                    lr = self.base_lr * (self.steps + 1) / self.warmup_steps
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr

                # update the weights
                self.optimizer.step()
                if self.steps >= self.warmup_steps:
                    self.scheduler.step()
                self.steps += 1
    
                # für progressbar
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "loss": f"{loss}",
                    "lr": f"{lr}"
                })

            self.train_loss.append(epoch_loss/len(self.train_dataset))


            # ----------- VALIDATION ----------- # 
            epoch_loss = 0
            pbar = tqdm(self.valid_dataset, desc="Validation")
            for _, data in enumerate(pbar):
                input = data
                target = input.clone()
                self.optimizer.zero_grad()
                estimate, _, _ = self.net(input, self.x)
                # apply loss
                loss = self.criterion(estimate, target)
                epoch_loss += loss.item() 
                
                # für progressbar
                pbar.set_postfix({
                    "loss": f"{loss}",
                })
            
            self.valid_loss.append(epoch_loss/len(self.valid_dataset))
            et_epoch = time.time()

            self.print_results(epoch, et_epoch-st_epoch)
            if epoch % 4 == 0:
                self.save_model(epoch)
            
            # loss plotten/speicher. kann auch öfter/seltener gemacht werden (mit lightning geht das auch gut)
            save_loss(self.train_loss, self.valid_loss, self.train_dir, save_plot=True)
            
            test_ir_out, _, _ = self.net(self.test_batch, self.x)
            write_audio(test_ir_out[0,:].detach(), 
                        os.path.join(self.train_dir, 'audio_output'),
                        #'e{:04d}-output-ir-loss{:.3f}.wav'.format(epoch, self.criterion(test_ir_out[0,:], self.test_batch[0,:])))
                        f"e{epoch}.wav")
        
            with open(os.path.join(self.train_dir, 'log.txt'), "a") as file:
                file.write("epoch: {:04d} train loss: {:6.4f} valid loss: {:6.4f}\n".format(
                    epoch, self.train_loss[-1], self.valid_loss[-1]))
            
            times = np.zeros(len(test_ir_out[0,:].detach()))
            eval_sig = pf.Signal([self.test_batch[0,:].flatten(),times.flatten()],sampling_rate=self.samplerate, is_complex=True)
            pred_sig = pf.Signal([test_ir_out[0,:].detach().flatten(),times.flatten()],sampling_rate=self.samplerate, is_complex=True)

            plt.figure()
            pf.plot.time_freq(eval_sig, label="eval", alpha=0.5)
            pf.plot.time_freq(pred_sig, label="pred", alpha=0.5)
            plt.legend()
            #plt.grid()
            plt.savefig(os.path.join(plot_out, f"e{epoch}.pdf"))
            plt.close()
            ### evaluate on epoch end, TODO adapt to delay-diff-net
            #self.evaluate(epoch=epoch)
    
    """def evaluate(self, epoch):
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
                #z = get_frequency_samples(eval_ir.size()[-1])

                if eval_ir.shape[0] != 1:
                    eval_ir = eval_ir.mean(dim=0, keepdim=True)

                eval_ir = pad_crop(eval_ir, sr, self.rir_length)
                if eval_ir.ndim == 2:        # [C, T] zu [B, C, T]
                    eval_ir = eval_ir.unsqueeze(0)

                estimate, _, _ = self.net(eval_ir, gamma)#, z)

                save_path = f"{self.train_dir}/evaluation" #/epoch{epoch}"
                os.makedirs(save_path, exist_ok=True)
            
                # plot
                pred_np = estimate.squeeze(0).squeeze(0).detach().cpu().numpy()
                #eval_np = eval_ir.squeeze(0).detach().cpu().numpy()
                
                #pred_real = np.abs(pred_np).astype(np.float32)

                #sf.write(f"{save_path}/{idx}-pred.wav", pred_real, self.samplerate)
                #sf.write(f"{save_path}/{idx}-{batch}.wav", eval_np, self.samplerate)
                #shutil.copy(eval_filepath, save_path)

                times = np.zeros(len(pred_np))
                eval_sig = pf.Signal([eval_ir.flatten(),times.flatten()],sampling_rate=self.samplerate, is_complex=True)
                pred_sig = pf.Signal([pred_np.flatten(),times.flatten()],sampling_rate=self.samplerate, is_complex=True)

                plt.figure()
                pf.plot.time_freq(eval_sig, label="eval", alpha=0.5)
                pf.plot.time_freq(pred_sig, label="pred", alpha=0.5)
                plt.legend()
                plt.savefig(os.path.join(save_path, f"e{epoch}-{idx}-plot.pdf"))
                plt.close()
            
        random.seed()"""

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


def main(args):
    train_dataset, valid_dataset = load_dataset(args)  #### old dataset class
    
    """dataset = rirDataset(args)
    train_dataset, valid_dataset = split_dataset(dataset, args.split)
    # dataloaders
    train_loader = get_dataloader(  train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle = args.shuffle,) 
    valid_loader = get_dataloader(  valid_dataset,
                                    batch_size=args.batch_size,
                                    shuffle = args.shuffle,)"""
    
    # init neural net
    filepath = 'Params/'
    N = args.N
    filename = 'param' + '_N' + str(N) + '_d' + str(args.delay_set)

    df = pd.read_csv(filepath+filename+'.csv', delimiter=';', nrows=N*N, dtype={'A':np.float32,'m':'Int32'})
    delay_lengths = torch.from_numpy(df['m'][:N].to_numpy())
    
    #z = get_frequency_samples(int(args.rir_length*args.samplerate))
    #net = DiffFDN(delay_lens, z, args.samplerate, args.rir_length)
    net = ASPestNet(delay_lengths=delay_lengths, ir_length=args.rir_length)

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"batch size = {args.batch_size} | trainable params = {(trainable_params/1000000):.3f}M")
    with open(os.path.join(args.train_dir, "params.txt"), 'w') as f:
        f.write(f"trainable params = {trainable_params}")
    print(f"trainset size: {len(train_dataset.dataset)} | valset size: {len(valid_dataset.dataset)}")
    
    trainer = Trainer(net, args, train_dataset, valid_dataset)
    #trainer = Trainer(net, args, train_loader, valid_loader)
    trainer.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--samplerate', type=int, default=48000, help ='sample rate')
    
    # dataset 
    parser.add_argument('--path_to_IRs', type=str, default="/Users/oscar/documents/Uni/Audiokommunikation/3. Semester/DLA/Impulse Responses/train")
    parser.add_argument('--split', type=float, default=0.8, help='training / validation split')
    parser.add_argument('--shuffle', default=True, help='if true, shuffle the data in the dataset at every epoch')
    parser.add_argument('--rir_length', type=float, default=1.8, help="wenn != None werden alle IRs auf diese Länge gebracht. ist eig pflicht")
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--delay_set', type=int, default=1)
    parser.add_argument('--num', type=int, default=120000)
    parser.add_argument('--clip_max_norm', default=10, help='gradient clipping maximum gradient norm')
    parser.add_argument('--warmup_steps', type=int, default=0, help="wichtig für conformer")
    
    # training
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=1000,  help='maximum number of training epochs')
    parser.add_argument('--log_epochs', action='store_true', help='Store met parameters at every epoch')
    
    # optimizer
    parser.add_argument('--lr', type=float, default=10e-5, help='learning rate')
    parser.add_argument('--scheduler_steps', default=250000)
    parser.add_argument('--training_name', type=str, default=None)
    args = parser.parse_args()

    if not args.training_name:
        args.training_name = time.strftime("%Y%m%d-%H%M%S")
    args.train_dir = os.path.join('outputs', args.training_name)
    os.makedirs(args.train_dir, exist_ok=True)
    
    # save arguments
    with open(os.path.join(args.train_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    
    main(args)