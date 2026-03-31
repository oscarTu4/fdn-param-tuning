import torch
import time
from losses import *
from model import *
from diff_dsp import *
from tqdm import tqdm
from dataset import load_dataset
from utils.logging import *
from utils.utility import * 

import time
import os
import argparse
import numpy as np
import json

import pyfar as pf
from matplotlib import pyplot as plt

class Trainer:
    """
        Trainer class to run/manage training

        args:
        - net: neural network to be trained
        - args: argument parser from main function
        - train_dataset: training dataset as torch.utils.data.Dataloader
        - valid_dataset: validation dataset as torch.utils.data.Dataloader
    """
    def __init__(self, net, args, train_dataset, valid_dataset):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = net
        self.net = self.net.to(self.device)

        self.max_epochs = args.max_epochs
        self.train_dir = args.train_dir
        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        
        self.steps = 0
        self.scheduler_steps = args.scheduler_steps

        print(f"args.conf_backbone: {args.conf_backbone}")
        print(f"args.lr: {args.lr}")
        if args.conf_backbone:
            print(f"init AdamW Optimizer mit WU steps")
            self.optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
            self.warmup_steps = 250*args.batch_size
        else:
            print(f"init Adam Optimizer ohne WU steps")
            self.warmup_steps = 0
            self.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        
        # init loss function
        self.criterion = MSSpectralLoss()
        # init learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 2500, gamma = 10**(-0.2))
        self.base_lr = args.lr

        self.samplerate = args.samplerate
        self.rir_length = args.rir_length
        
        # variables for in-training validation
        num = 120000
        self.x = get_frequency_samples(num//2+1).to(self.device)
        self.test_batch = next(iter(valid_dataset)).to(self.device)
    
    def train(self):
        self.net.train()
        device = get_device()

        # set aside one audio file as audible/visible validation
        write_audio(self.test_batch[0,:], 
                os.path.join(self.train_dir, 'audio_output'),
                'target_ir.wav')
        self.train_loss, self.valid_loss = [], []
        plot_out = os.path.join(self.train_dir, 'plots')
        os.makedirs(plot_out, exist_ok=True)

        # early stopping, is not used
        early_stop = EarlyStopper(patience=50000// 32, min_delta=1e-4)

        for epoch in range(self.max_epochs):
            st_epoch = time.time()
            epoch_loss, grad_norm = 0, 0
            
            # ----------- TRAINING ----------- # 
            pbar = tqdm(self.train_dataset, desc=f"Training | Epoch {epoch}/{self.max_epochs}")
            for idx, input in enumerate(pbar):
                input = input.to(device)
                target = input.clone()

                self.optimizer.zero_grad()
                estimate, H, _, _, _ = self.net(input, self.x)  # get estimate
                
                if not torch.isfinite(estimate).all():
                    print("Non-finite estimate detected")
                    print("estimate max:", estimate.abs().max().item())
                loss = self.criterion(estimate, target) # compute loss
                if torch.isnan(loss):
                    print(f"loss is nan")
                    exit()
                epoch_loss += loss.item()
                loss.backward()
                # clip gradients
                grad_norm += nn.utils.clip_grad_norm_(self.net.parameters(), args.clip_max_norm)

                # LR warmup
                if self.steps < self.warmup_steps:
                    lr = self.base_lr * (self.steps + 1) / self.warmup_steps
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr

                # update the weights and step the scheduler
                self.optimizer.step()
                if self.steps >= self.warmup_steps:
                    if self.steps >= (self.scheduler_steps + self.warmup_steps):
                        #print(f"scheduler step")
                        self.scheduler.step()
                self.steps += 1
    
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "loss": f"{loss}",
                    "lr": f"{lr}"
                })

            self.train_loss.append(epoch_loss/len(self.train_dataset))


            # ----------- VALIDATION ----------- # 
            self.net.eval()
            epoch_loss = 0
            pbar = tqdm(self.valid_dataset, desc="Validation")
            with torch.no_grad():
                for _, input in enumerate(pbar):
                    input = input.to(device)
                    target = input.clone()
                    estimate, H, _, _, _ = self.net(input, self.x)
                    # apply loss
                    loss = self.criterion(estimate, target)
                    epoch_loss += loss.item() 

                    pbar.set_postfix({
                        "loss": f"{loss}",
                    })
            self.net.train()
            
            self.valid_loss.append(epoch_loss/len(self.valid_dataset))
            et_epoch = time.time()

            self.print_results(epoch, et_epoch-st_epoch)
            if epoch % 2 == 0:
                self.save_model(epoch)
            
            # plot and save loss
            save_loss(self.train_loss, self.valid_loss, self.train_dir, save_plot=True)
            
            # make audio that was set aside audible and visible and save to file
            test_ir_out, H, _, _, _ = self.net(self.test_batch, self.x)
            write_audio(test_ir_out[0,:].detach(), 
                        os.path.join(self.train_dir, 'audio_output'),
                        f"epoch-{epoch}.wav")
        
            with open(os.path.join(self.train_dir, 'log.txt'), "a") as file:
                file.write("epoch: {:04d} train loss: {:6.4f} valid loss: {:6.4f}\n".format(
                    epoch, self.train_loss[-1], self.valid_loss[-1]))
            
            es = self.test_batch[0,:].cpu()
            ps = test_ir_out[0,:].cpu().detach()
            times = np.zeros(len(ps))
            eval_sig = pf.Signal([es.flatten(),times.flatten()],sampling_rate=self.samplerate, is_complex=True)
            pred_sig = pf.Signal([ps.flatten(),times.flatten()],sampling_rate=self.samplerate, is_complex=True)

            plt.figure()
            pf.plot.time_freq(eval_sig, label="eval", alpha=0.5)
            pf.plot.time_freq(pred_sig, label="pred", alpha=0.5)
            plt.legend()

            plt.savefig(os.path.join(plot_out, f"e{epoch}.pdf"))
            plt.close()

            # early stopping, does nothing right now, basically redundant
            if early_stop.early_stop(self.valid_loss[-1]):
                return


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
    # create empty neural network
    net = ASPestNet(rir_length=args.rir_length, conf_backbone=args.conf_backbone)

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"batch size = {args.batch_size} | trainable params = {(trainable_params/1000000):.3f}M")
    with open(os.path.join(args.train_dir, "params.txt"), 'w') as f:
        f.write(f"trainable params = {trainable_params}")
    
    # load training and validation data
    train_dataset, valid_dataset = load_dataset(args)
    
    # initialize Trainer class
    trainer = Trainer(net, args, train_dataset, valid_dataset)
    
    # train
    trainer.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--samplerate', type=int, default=48000, help ='sample rate')
    
    # dataset 
    parser.add_argument('--path_to_IRs', type=str, default="/Users/oscar/Documents/Uni/Audiokommunikation/3. Semester/DLA/Impulse Responses/train_of")
    parser.add_argument('--split', type=float, default=0.8, help='training / validation split')
    parser.add_argument('--shuffle', default=True, help='if true, shuffle the data in the dataset at every epoch')
    parser.add_argument('--rir_length', type=float, default=1.8, help="length (seconds) that IRs are padded/cropped to. this is essential")
    parser.add_argument('--clip_max_norm', type=float, default=10.0, help='gradient clipping maximum gradient norm')
    
    # training
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=30,  help='maximum number of training epochs')
    parser.add_argument('--log_epochs', action='store_true', help='Store met parameters at every epoch')
    parser.add_argument('--conf_backbone', action='store_true', help="needs to be activated if you want to use conformer architecture")
    
    # optimizer
    parser.add_argument('--lr', type=float, default=2e-6, help='learning rate')
    parser.add_argument('--scheduler_steps', type=int, default=2500)
    parser.add_argument('--training_name', type=str, default="test_training")
    args = parser.parse_args()

    if not args.training_name:
        args.training_name = time.strftime("%Y%m%d-%H%M%S")
    args.train_dir = os.path.join('outputs', args.training_name)
    os.makedirs(args.train_dir, exist_ok=True)
    
    # save arguments
    with open(os.path.join(args.train_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    
    main(args)