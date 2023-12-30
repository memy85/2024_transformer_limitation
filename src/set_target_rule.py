#%%
from utils import load_config

config = load_config()
#%%
from pathlib import Path
import pandas as pd
import os, sys

config = load_config()
projectPath = Path(config['project_path'])
dataPath = projectPath.joinpath("data")
outputPath = dataPath.joinpath('processed/')
contextPath = dataPath.joinpath('processed/')
semantic_path = dataPath.joinpath("raw/medspacy_test/SemanticTypes_2018AB.txt")
quickumls_path = dataPath.joinpath("raw/quickumls")

#%%
def load_semantic_types(path) :
    header = ["abrev", "TUI", "text"]
    df = pd.read_csv(path,sep="|",names = header)
    return df

#%%
semantics = load_semantic_types(semantic_path)


#%%
