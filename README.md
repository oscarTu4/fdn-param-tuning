# Neural Parameter Tuning of Feedback Delay Networks

Simon Nutsch
Oscar Eckhorst
Florentin Steigerwald
Felix Flaxl

This code is an adaptation of (1) (see Code References). 

## Getting started with training

Follow steps 1-4 to get training started

1. Clone Git Repo
2. Create conda environment to install required packages and activate

```bash
$ conda create -n dla-fdn python=3.11
$ conda activate dla-fdn
```

3. Install required packages
````bash
$ python -m pip install -r requirements.txt
````

on Windows you might need to do:
````bash
$ pip install torchcodec
$ conda install "ffmpeg<8"
````

4. Run training script `train.py`

````bash
$ python train.py
````

IMPORTANT: in order to use Conformer architecture as described in the paper, activate `--conf_backbone`. The model will run with the GRU architecture by default.

The training results including loss curves, checkpoints and some validation files are saved in `outputs/your-training-name`.

`train.py` accepts the following args:

- `--path_to_IRs`
    path to IR dataset
- `--split`
    what % of dataset is used for training. rest is used in validation
- `--shuffle`
    wether to shuffle dataset at each epoch
- `--rir_length`
    desired length of IR samples in seconds (e.g. 1, 3, 5.5). 
    this is mandatory, although a default of 1.8 was used in the paper
- `--clip_max_norm`
    gradient clipping
- `--batch_size`
- `--max_epochs`
- `--log_epochs`
- `--conf_backbone`
    if activated, model will run with Conformer architecture. if this argument is ignored, the model with use the GRU architecture by default.
- `--lr`
    learning rate
- `--scheduler_steps`
    after how many iterations should the learning rate scheduler shall 'step'
- `--training_name`
    name of the training. all results will be saved in `/outputs/training_name`


## Inference

Run `inference.py`in order to create a FDN reverb from your impulse response.
As there is no argument parser, you need to enter some information at the top of the file.

## Architecture

The main FDN modelling is realised in `model.py`(taken from (1)), adapted only to replace the GRU architecture with our Conformer architecture. The GRU architecture can also be found here, untouched.
The Conformer architecture is realised in the file `custom_encoder.py` and `ConformerBlock.py`.

As already stated, run `train.py --conf_backbone` in order to activate Conformer architecture.

## Evaluation

Run evaluation.py to compute objective metrics (T30, C50, DRR) for predicted and reference room impulse responses. We follow the evaluation protocol proposed by Lee et al. (2022), but implemented our own evaluation pipeline since no code was available. 

Before running the script please adjust the following parameters:
•⁠  ⁠Path to the trained model (exp)
•⁠  ⁠Path to the evaluation dataset (eval_path)
•⁠  ⁠Dataset identifier (dataset = MIT or dataset = shoebox)
•⁠  ⁠Checkpoint epoch (epoch)

Output: The evaluation script produces: 
•⁠  ⁠.txt files: Human-readable summaries including median errors, relative T30 error and perceptual interpretation based on JND thresholds
•⁠  ⁠.csv files: Structured results for further analysis and plotting 

Visualiziation:
To visualize the results please use evaluation_plots.py. Before running, specify the paths to the .csv files and output directories at the top of the script.

## Code References

https://github.com/gdalsanto/diff-delay-net.git (1)

https://github.com/gdalsanto/diff-fdn-colorless.git (2)

https://docs.pytorch.org/audio/2.1/_modules/torchaudio/models/conformer.html (3)

## Literature References: 

Lee, S., Choi, H.-S., & Lee, K. (2022). Differentiable Artificial Reverberation (arXiv:2105.13940). arXiv. https://doi.org/10.48550/arXiv.2105.13940
