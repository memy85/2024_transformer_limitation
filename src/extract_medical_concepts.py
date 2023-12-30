
from medspacy.util import DEFAULT_PIPENAMES
import medspacy
import numpy as np
import pandas as pd
from pathlib import Path
import os, sys
import jsonlines
from utils import load_config
# from quickumls.spacy_component import SpacyQuickUMLS

config = load_config()
projectPath = Path(config['project_path'])
dataPath = projectPath.joinpath("data")
outputPath = dataPath.joinpath('processed/')
contextPath = dataPath.joinpath('processed/')
semantic_path = dataPath.joinpath("raw/medspacy_test/SemanticTypes_2018AB.txt")
quickumls_path = dataPath.joinpath("/home/zhichaoyang/medspacy_test")


def load_semantic_types(path) :
    header = ["abrev", "TUI", "text"]
    df = pd.read_csv(path,sep="|",names = header)
    return df

# data = load_semantic_types(semantic_path)
# data.head()
# tui2abrev = dict(zip(data.TUI, data.abrev))
#%%
# data

#%%

def load_contexts(name) :
    '''
    names : dev, train, test
    '''
    path = contextPath.joinpath(name +".jsonl")

    with jsonlines.open(path) as f:
        contexts = []
        for line in f.iter():
            contexts.append(line['context'])# or whatever else you'd like to do
    return contexts

#%%

def load_semantics() :

    medspacy_pipes = DEFAULT_PIPENAMES.copy()
    if 'medspacy_quickumls' not in medspacy_pipes:
        medspacy_pipes.add('medspacy_quickumls')
    nlp = medspacy.load(enable = medspacy_pipes, quickumls_path=quickumls_path.as_posix())
    return nlp



def extract_medical_concepts(extractor, text):
    context = extractor(text)
    the_enteties = [all_ent for all_ent in context.ents]
    return the_enteties


def main() : 
    #%%
    # data = load_semantic_types(semantic_path)
    for name in ["dev", "train", "test"] :
        context = load_contexts(name)
        nlp = load_semantics()

        #%%
        sample = context[0]
        doc = nlp(sample)

        #%%
        print(sample)

        #%%
        for token in doc.ents : 
            print(token)

    pass

if __name__ == "__main__" :
    main()
