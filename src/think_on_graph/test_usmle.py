#%%
import os, sys
from pathlib import Path

srcPath = Path(__file__).absolute().parents[1]
os.sys.path.append(srcPath.as_posix())

from utils import *
import torch
#%%
config = load_config()
config.model_path("llama2")

#%% declare device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"device is {device}")

#%%
from transformers import pipeline
torch.manual_seed(0)
pipe = pipeline('text-generation', 
                model=config.model_path('llama2'),
                torch_dtype=torch.bfloat16,
                framework='pt',
                device_map='auto')

#%%
prompt = """7 times 7 is 49
            3 times 2 is 6
            8 times 8 is """

#%%
pipe(prompt, max_new_tokens=50)

# %%
prompt = config.template("thinkongraph")
print(prompt)

#%%
output = pipe(prompt, max_new_tokens=100, return_full_text=False)

# %%
print(output[0]['generated_text'])
