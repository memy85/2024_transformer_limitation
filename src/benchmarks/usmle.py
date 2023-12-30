#%%

from pathlib import Path
projectPath = Path(__file__).parents[2]
dataPath = projectPath.joinpath("data/processed")
print(f"project path is {projectPath}")

import os
os.sys.path.append(projectPath.joinpath('src').as_posix())
from utils import *

config = load_config()

from transformers import pipeline
from datasets import load_dataset
import torch
import torch.nn as nn

#%%

print(config['template']['chainofthought']['example'])


# load dataset
#%%
data_files = {"train":dataPath.joinpath("train.jsonl").as_posix(),
              "eval": dataPath.joinpath("dev.jsonl").as_posix(), 
              "test":dataPath.joinpath("test.jsonl").as_posix()}
dataset = load_dataset('json', data_files=data_files)

#%%



#%%

dataset['train'][0]

#%%

#%%
from 






#%%
model_name = 
pipeline("text-generation", model=)