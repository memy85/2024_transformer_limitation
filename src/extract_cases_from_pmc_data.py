#%%
import json
import pickle
from pathlib import Path
import os, sys
import json
import jsonlines
import torch
import argparse
import copy
import re

from fastchat.model import load_model, get_conversation_template, add_model_args
from transformers import pipeline

#%% get file path to configuration
file_path = Path(__file__).absolute()
projectPath = file_path.parents[1]
print("project path is : ", projectPath)

os.sys.path.append(projectPath.joinpath('src').as_posix())
from utils import load_config
config = load_config()

#%% define path
projectPath = Path(config['project_path'])
dataPath = projectPath.joinpath('data/')
outPath = dataPath.joinpath("processed")

#%% define the language model that we would want to use
modelPath = "/home/zhichaoyang/llm_share/vicuna/vicuna-13b"
device = "cuda" if torch.cuda.is_available() else  "cpu"
# device = "cpu"
num_gpu = 1
print("the device is ", device)

#%%
@torch.inference_mode()
def ask_to_vicuna(model, tokenizer, patient_scenario, msg : tuple):

    '''
    Here the msg is like : ("What is the most likely diagnosis of this patient?",
                            "Cover the part with the diagnosis of this patient with xxxxxx",
                            "Give 4 possible most likely answers which are not the exact answer for this person's diagnosis")
    '''
    # Load model

    # Build the prompt with a conversation template
    (diagnosis_q, context_without_diagnosis_q, distractors_q) = msg
    patient_scenario_with_q = patient_scenario + "\n" + diagnosis_q

    conv = get_conversation_template(modelPath)
    conv.append_message(conv.roles[0], patient_scenario_with_q)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # Run inference
    answer_diagnosis = decode_output_answers(model, tokenizer, prompt)

    conv = get_conversation_template(modelPath)
    conv.append_message(conv.roles[0], patient_scenario_with_q)
    conv.append_message(conv.roles[1], answer_diagnosis)
    conv.append_message(conv.roles[0], context_without_diagnosis_q)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # Run inference
    answer_question = decode_output_answers(model, tokenizer, prompt)
    
    conv = get_conversation_template(modelPath)
    conv.append_message(conv.roles[0], patient_scenario_with_q)
    conv.append_message(conv.roles[1], answer_diagnosis)
    conv.append_message(conv.roles[0], distractors_q)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # Run inference
    answer_distractor = decode_output_answers(model, tokenizer, prompt)

    return answer_diagnosis, answer_question, answer_distractor

#%%
def decode_output_answers(model, tokenizer, prompt) :

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        do_sample=True if 0.7 > 1e-5 else False,
        temperature=0.7, # args.temperature,
        repetition_penalty= 1, # args.repetition_penalty,
        max_new_tokens=512, # args.max_new_tokens,
    )
    # realOutput = copy.deepcopy(output_ids)

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]

    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs
#%%
def save_to_jsonlines(output_path, name, final_data) :
    '''
    saves the final data to the output path
    input : output_path, name, final_data
    '''
    output_file_name = output_path.joinpath(name)
    with jsonlines.open(output_file_name, mode='w') as writer : 
        writer.write_all(final_data)


def extract_distractor(distractor_string) :
    p = re.compile("(\d). (\w*)")
    distractors = p.findall(distractor_string)
    
    return {int(i) : diagnosis for i, diagnosis in distractors}


def construct_data(original_text, vicuna_output) :
    output = vicuna_output
    context = output[1]
    question = "What is the most likely diagnosis of this patient?"
    answer = output[0]
    distractors = output[2]

    # distractors = extract_distractor(distractors)

    return {
        "original_text" : original_text,
        "context" : context,
        "question" : question,
        "answer" : answer,
        "distractors" :distractors
    }


def main() :
    # write input data
    with open(dataPath.joinpath("raw/pmc_patients.json")) as f :
        data = json.load(f)

    message = ("What is the most likely diagnosis of this patient?, just give the answer. Don't write in full sentence",
            "rewrite the text so that it is impossible to know the diagnosis of this patient. Do minimum change. The diagnosis of the patient should not be explicitly written.",
            "Give 4 possible most likely answers which are not the exact answer for this person's diagnosis")

    model, tokenizer = load_model(
        modelPath,
        device=device,
        num_gpus=num_gpu,
        # max_gpu_memory=args.max_gpu_memory,
        # load_8bit=args.load_8bit,
        # cpu_offloading=args.cpu_offloading,
        # revision=args.revision,
        # debug=args.debug,
    )
    resultList = []
    for idx, jsonfile in enumerate(data) :
        if idx == 100 :
            break

        sampleText = jsonfile['patient']

        output = ask_to_vicuna(model, tokenizer, sampleText, message)
        parsedJson = construct_data(sampleText, output)
        resultList.append(parsedJson)
        print(f"-------------------- finished for case {idx} ----------------------")

    # return resultList
    save_to_jsonlines(outPath, "pmc.jsonl", resultList)   



#%%
if __name__ == "__main__" :
    main()
