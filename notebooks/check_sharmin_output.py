#%%
import pandas as pd
from pathlib import Path
import jsonlines
import json
import pprint
projectPath = Path(__file__).parents[1]
dataPath = projectPath.joinpath("data")

#%%
professional_path = dataPath.joinpath("processed/chatgpt_response_professional.txt")
student_path = dataPath.joinpath("processed/chatgpt_response_medical_student.txt")
#%%

def load_original() :
    original = dataPath.joinpath("processed/samples_for_sharmin.jsonl")
    with jsonlines.open(original) as f :
        docs = []
        for doc in f: 
            docs.append(doc)
    return docs 

original = load_original()

#%%
original



#%%
class Document :

    def __init__(self, json) :
        self.json = json
        self.question = self.json['question']
        # self.options = self.json['options']
        self.parse_options()

    def parse_options(self)  :
        options = {}
        for idx, (option, val) in enumerate(self.json['options'].items()) :
            options[idx] = (val['text'], val['percentage'])
        self.options = options
    
    def __repr__(self) -> str:
        return "Document Class"


#%%
class Comparator :

    def __init__(self, doc1_path, doc2_path) :
        self.doc1_path = doc1_path
        self.doc2_path = doc2_path

        self.doc1 = self.load_doc(self.doc1_path)
        self.doc2 = self.load_doc(self.doc2_path)

    def load_doc(self, doc_path) :
        with jsonlines.open(doc_path, 'r') as f :
            docs = []
            for doc in f :
                doc = Document(doc)
                docs.append(doc)
        return docs
    
    def compare_doc(self, index) :
        doc1 = self.doc1[index]
        doc2 = self.doc2[index]

        pprint.pprint(doc1.question)
        print("===========================================================")
        print("the answer was : ", original[index]['answer'])
        print("the options ===============================================")
        df = pd.DataFrame({"doc1": self.doc1[index].options,
                      "doc2": self.doc2[index].options})
        print(df)
        pass

#%%
professional_student_comparator = Comparator(professional_path, student_path)

#%%
professional_student_comparator.compare_doc(4)



#%%
docs[0]

#%%


# %%
