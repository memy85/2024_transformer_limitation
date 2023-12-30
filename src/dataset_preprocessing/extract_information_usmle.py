
#%%
import json
import pickle
from pathlib import Path
import os, sys
import json
import jsonlines
import re
import pandas as pd

#%% get file path to configuration
projectPath = Path(os.getcwd())

file_path = Path(__file__).absolute()
projectPath = file_path.parents[2]
print("project path is : ", projectPath)

os.sys.path.append(projectPath.joinpath('src').as_posix())
from utils import load_config
#%%
config = load_config()

#%% define path
projectPath = Path(config['project_path'])
dataPath = projectPath.joinpath('data/')
outPath = dataPath.joinpath("processed")

#%% define functions

def load_jsonl_data(path) :
    '''
    parse the jsonl data and return it in a list with dictionaries
    '''
    with jsonlines.open(path) as f:

        jsonl_list = []
        for line in f.iter() :
            jsonl_list.append(line)
        return jsonl_list

# def filter_out_only_diagnosis_questions(question : str) :

#     question_lower = question.lower()

#     if "diagnosis" in question_lower : 
#         return True
#     else :
#         return False


def parse_answers_returning_distractors(jsonl_data, answer) :
    '''
    parse the options and return only the options without the answer
    '''
    options = jsonl_data['options']
    all_options = list(options.values())
    all_options.remove(answer)

    return all_options


def parse_json_to_extract_context(jsonl_data : dict, question) :
    # all_the_sentences = jsonl_data['question'].split(".")
    # context = ".".join(all_the_sentences[:-1])

    text = jsonl_data['question']
    context = text.replace(question,"")
    context = context.strip()
    return context

def get_the_question(jsonl_data : dict) :
    # all_the_sentences = jsonl_data['question'].split(".")

    # question = all_the_sentences[-1]
    # question = question.lstrip()

    # using regular expression
    p = re.compile("[A-Za-z]+[^.!?]*[.?:;]")
    question = jsonl_data['question']
    question = question.replace("\n",". ")
    q = p.findall(question)

    # extract questions with ?
    extracted = list(filter(lambda x : "?" in x, q))
    if len(extracted) > 0 :
        return extracted[0]
    else :
        q = q[-1]
        q = q.strip()
        return q


def preprocess_questions(question : str) :
    question = question.replace('?"',"?")
    question.replace("\n", "\n.")
    return question


def parse_json_to_extract_answer(jsonl_data : dict) :
    '''
    return the answer parsed from the json
    '''
    return jsonl_data['answer']


def save_to_jsonlines(output_path, name, final_data) :
    '''
    saves the final data to the output path
    input : output_path, name, final_data
    '''
    output_file_name = output_path.joinpath(name)
    
    with jsonlines.open(output_file_name, mode='w') as writer : 
        writer.write_all(final_data)

#%%

def get_diagnosis_questions(jsonl_data) :

    '''
    process the jsonl data to collect the informations needed. 
    '''
    preprocessed_q = preprocess_questions(jsonl_data['question'])
    jsonl_data['question'] = preprocessed_q

    question = get_the_question(jsonl_data)

    context = parse_json_to_extract_context(jsonl_data, question)
    answer = parse_json_to_extract_answer(jsonl_data)
    distractors = parse_answers_returning_distractors(jsonl_data, answer)

    return {
            "context" : context,
            "question" : question, 
            "answer" : answer,
            "distractors" : distractors
            }


#%%
def main() :

    datasets = ['train.jsonl', 'dev.jsonl', 'test.jsonl']
    question = []
    for data in datasets : 
        thePath2jsonl = dataPath.joinpath(f"raw/US/{data}")

        # load the data
        jsonldataList = load_jsonl_data(thePath2jsonl)

        # parse to get context, question, answer, distractors
        resultList = []
        for jsonlData in jsonldataList : 
            result = get_diagnosis_questions(jsonlData)
            q = result['question']
            question.append({"dataset":data, "question":q})

            if result is not None : resultList.append(result)

        save_to_jsonlines(outPath, data, resultList)   
    question_df = pd.DataFrame(question)
    question_df.to_csv(outPath.joinpath("question_df.csv"), index=False)
    
    

if __name__ == "__main__" :
    main()

#%% check questions

# df = pd.read_csv(outPath.joinpath("question_df.csv"))
# df

# #%%
# msk = df['question'].str.endswith("?").astype('bool')
# #%%
# problematic_questions = df[~msk]['question'].reset_index(drop=True)

# #%%
# problematic_questions[13]

# #%%
# df
#%%
# df.iloc[321]['question']
#%%





