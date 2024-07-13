from google.colab import drive
drive.mount('/content/drive')

!pip install hydra-core

!pip install torchmetrics

!pip install wandb

!pip install einops

!pip install omegaconf

!pip install git+https://github.com/openai/CLIP.git

import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

# srcディレクトリの親ディレクトリをsys.pathに追加
sys.path.append('/content/drive/MyDrive/Colabdata/dl_lecture_competition_pub')
from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed

!python "/content/drive/MyDrive/Colabdata/dl_lecture_competition_pub/run.py"