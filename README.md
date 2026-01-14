# FDN

hier ein kleine Zusammenfassung was was macht in dem Projekt:

-- `train.py`
    Start und Ende der Operation. Hier wird alles initialisiert und organisiert. 
    Hyperparameter werden hier angegeben, Modell gestartet und Ergebnisse/Checkpoints gespeichert. 
    Ort für Trainings/Validierungsloops.

-- `dataset.py`
    Skript für Laden und Bearbeiten von RIRs. 
    Skript liest alle WAV Dateien aus dem angebenen `--path_to_IRS` (auch subordner) und ignoriert den Rest. 

-- `fdn.py`
    Herz des Modells. 
    Das hier wird gestartet im Trainingsloop. 

-- `custom_encoder.py`
    Encoder Logik, hier lernen die Parameter (A, b, c im Moment). 

-- `inference.py`
    hier kommt die Logik für die 'Nutzung' vom Modell, also eigene IR rein, FDN IR raus. 

-- `losses.py`
    Hier sind alle Loss Funktionen definiert. 

-- `utility.py`
    Skript mit Hilfsfunktionen. 



## Getting started 

Follow steps 1-5 to get training started

1. clone git repo

```
git clone https://github.com/oscarTu4/fdn-param-tuning.git
```

2. Create conda (oder was auch immer) environment to install required packages (python 3.11 funktioniert gut) and activate

```bash
$ conda create -n dla-fdn python=3.11
$ conda activate dla-fdn
```

3. Install required packages
````bash
$ python -m pip install -r requirements.txt
````

4. Run training script `train.py`

````bash
$ python train.py --path_to_datasets
````

oder einfach 'play' in vscode. 
die --arg parameter sind ganz unten in train.py definiert, die kann man auch einfach da in den default Wert schreiben. 
dann muss man nicht andauernd lästig den pfad im terminal eingeben

training scripts accepts the following args:

- `--path_to_IRs`
    path to IR dataset
- `--split`
    what % of dataset is trainset. rest is validation
- `--shuffle`
    wether to shuffle dataset at each epoch
- `--ir_length`
    desired length of IR samples in seconds (e.g. 1, 3, 5.5)
- `--batch_size`
- `--max_epochs`
- `--log_epochs`
- `--lr`
    learning rate
- `--scheduler_steps`
    after how many iterations should the scheduler 'step'. 
    doesn't do anything yet, scheduler steps after each epoch atm
- `--clip_max_norm`
    gradient clipping

`.  

## Inference/Evaluation

steht noch an

## Was noch gut wäre

Im Moment kann man keine Trainings von vortrainierten Checkpoints laden, das wär noch gut
Falls wir Pytorch Lightning benutzen wollen wäre das damit abgedeckt

## References

https://github.com/gdalsanto/diff-delay-net.git

https://github.com/gdalsanto/diff-fdn-colorless.git
