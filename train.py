import torch
import time
import torchaudio 
from losses import *
from utility import *
from tqdm import trange, tqdm

import time
import os
import argparse
from utility import * 
from dataset import load_dataset
from fdn import DiffFDN
import pandas as pd

class Trainer:
    # eine wichtige Überlegung wäre das Training über Pytorch Lightning zu machen, 
    # das macht das Speichern/Laden von Checkpoints wesentlich einfacher falls unser Modell gross wird
    # und macht auch das zeigen von Infos (Modelgrösse, Parameter usw) angenehmer
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
        self.clip_max_norm = args.clip_max_norm

        self.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr) 
        #self.criterion = mse_loss().to(device)
        self.criterion = MSSpectralLoss(sr=args.samplerate).to(device)  # vlt besser als MSE. sparsity macht wohl keinen sinn hier
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 50000, gamma = 10**(-0.2)) 

        #self.normalize() # normalize sollte denke angepasst werden, erstmal rausgenommen damit es läuft
    
    def train(self):
        self.train_loss, self.valid_loss = [], []
        
        st = time.time()

        for epoch in range(self.max_epochs):
            st_epoch = time.time()

            # training
            epoch_loss = 0
            pbar = tqdm(self.train_dataset, desc=f"Training | Epoch {epoch+1}/{self.max_epochs}")
            for _, data in enumerate(pbar):
                loss = self.train_step(data)
                nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_max_norm)
                self.optimizer.step()

                epoch_loss += loss
                
                if self.steps >= self.scheduler_steps:
                    self.scheduler.step()
                    print("scheduler step")
                self.steps = (self.steps + 1) % self.scheduler_steps
                
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "loss": f"{loss:.3f}",
                    "lr": f"{lr}"
                })

            self.train_loss.append(epoch_loss/len(self.train_dataset))

            # validation
            epoch_loss = 0
            pbar = tqdm(self.valid_dataset, desc="Validation")
            for _, data in enumerate(pbar):
                loss =self.valid_step(data)
                epoch_loss += loss
                
                pbar.set_postfix({
                    "loss": f"{loss:.3f}"
                })
            self.valid_loss.append(epoch_loss/len(self.valid_dataset))
            et_epoch = time.time()

            self.print_results(epoch, et_epoch-st_epoch)
            self.save_model(epoch)
            
            # loss plotten/speicher. kann auch öfter/seltener gemacht werden (mit lightning geht das auch gut)
            save_loss(self.train_loss, self.valid_loss, self.train_dir, save_plot=True)

            # early stopping, auskommentiert weil es overfitting verhindert
            """if (epoch >=1):
                if (abs(self.valid_loss[-2] - self.valid_loss[-1]) <= 0.0001):
                    self.early_stop += 1
                else: 
                    self.early_stop = 0
            if self.early_stop == self.patience:
                break"""

        et = time.time()    # end time 
        print('Training time: {:.3f}s'.format(et-st))

    def train_step(self, x):
        # batch processing
        self.optimizer.zero_grad()
        gt = x.clone() # nur zur sicherheit
        y = self.net(x)
        loss = self.criterion(y, gt)
        
        #if torch.isnan(loss):
        #    print("LOSS IS NAN")
        #    print("y", torch.isnan(y).any())
        #    print("x", torch.isnan(x).any())
        #    exit()
        
        loss.backward()
        
        return loss.item()

    def valid_step(self, x):
        # batch processing
        self.optimizer.zero_grad()
        gt = x.clone() # nur zur sicherheit
        y = self.net(x)
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


def main(args):
    train_dataset, valid_dataset = load_dataset(args)
    
    # init neural net
    filepath = 'Params/'
    N = 8
    delay_set = 1 
    filename = 'param' + '_N' + str(N) + '_d' + str(delay_set)

    #df = pd.read_csv(filepath+filename+'.csv', delimiter=';', nrows=N*N, dtype={'A':np.float32,'m':'Int32'})
    df = pd.read_csv(filepath+filename+'.csv', delimiter=';', nrows=N*N, dtype={'A':np.float16,'m':'Int16'})
    delay_lens = torch.from_numpy(df['m'][:N].to_numpy())
    
    net = DiffFDN(delay_lens)
    net.apply(weights_init_normal) # weiss nich ob wir das hier brauchen, aber denke schon
    
    trainer = Trainer(net, args, train_dataset, valid_dataset)

    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"batch size = {args.batch_size} | trainable params = {(trainable_params/1000000):.3f}M")
    trainer.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--samplerate', type=int, default=48000, help ='sample rate')
    
    # dataset 
    parser.add_argument('--path_to_IRs', type=str, default="/Users/oscar/documents/Uni/Audiokommunikation/3. Semester/DLA/Impulse Responses/ChurchIR")
    parser.add_argument('--split', type=float, default=0.8, help='training / validation split')
    parser.add_argument('--shuffle', default=True, help='if true, shuffle the data in the dataset at every epoch')
    
    # training
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=100,  help='maximum number of training epochs')
    parser.add_argument('--log_epochs', action='store_true', help='Store met parameters at every epoch')
    
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-8, help='learning rate')
    parser.add_argument('--scheduler_steps', default=1000, help='sollte viieeeeel höher sein, das hier nur test')
    parser.add_argument('--clip_max_norm', default=1)
    args = parser.parse_args()

    args.train_dir = os.path.join('outputs', time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.train_dir, exist_ok=True)

    # save arguments 
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    main(args)