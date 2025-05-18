import datasets
import pandas as pd
import numpy as np
import random
# datanames = [ 'nq', 'sciq']
# formal_dataname_list = ['NQ', 'SciQ']
datanames = ['nq', 'sciq', 'triviaqa', 'truthfulqa', 'popqa']
formal_dataname_list = ['NQ', 'SciQ', 'TriviaQA', 'TruthfulQA', 'PopQA']
# datanames = ['truthfulqa']
model_to_full_name = {
    "olmo13b": "OLMo-2-1124-13B-Instruct",
    # "olmo13b-dpo": "OLMo-2-1124-13B-DPO",
    # "olmo13b-sft": "OLMo-2-1124-13B-SFT",
    # "llama8b": "Llama-3.1-Tulu-3-8B",
    # "llama8b-dpo": "Llama-3.1-Tulu-3-8B-DPO",
    # "llama8b-sft": "Llama-3.1-Tulu-3-8B-SFT",
    "olmo": "OLMo-2-1124-7B-Instruct",
    # "olmo-dpo": "OLMo-2-1124-7B-DPO",
    # "olmo-sft": "OLMo-2-1124-7B-SFT",
    "olmo-1": "OLMo-7B-Instruct-hf",
    # "olmo-1-sft": "OLMo-7B-SFT-hf",
}

model_to_formal_name = {
    "olmo13b": "OLMo2-13B-INS",
    # "olmo13b-dpo": "OLMo2-13B-DPO",
    # "olmo13b-sft": "OLMo2-13B-SFT",
    # "llama8b": "Llama3-8B-INS",
    # "llama8b-dpo": "Llama3-8B-DPO",
    # "llama8b-sft": "Llama3-8B-SFT",
    "olmo": "OLMo2-7B-INS",
    # "olmo-dpo": "OLMo2-7B-DPO",
    # "olmo-sft": "OLMo2-7B-SFT",
    "olmo-1": "OLMo-7B-INS",
    # "olmo-1-sft": "OLMo-7B-SFT"
}

for dataname, formal_dataname in zip(datanames, formal_dataname_list):
    exclude_dataset = []
    for name in datanames:
        if name != dataname:
          exclude_dataset.append(name)
    random_dataset = random.choices(exclude_dataset)[0]

    for model in model_to_full_name.keys():
        print(dataname, model)
        data = datasets.load_from_disk( model_to_full_name[model] + "/cos/"+ dataname + '_' + model + '_test/main')
        random_data = datasets.load_from_disk( model_to_full_name[model] + "/cos/"+ random_dataset + '_' + model + '_test/main')
        column = [random.choices(random_data['qa_query'])[0] for i in range(len(data['qa_query']))]

        data = data.add_column("random", column)
  
        data.save_to_disk( dataname + '_' + model + '_test')
