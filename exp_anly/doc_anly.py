import datasets
import pandas as pd
import numpy as np
datanames = ['nq', 'sciq', 'triviaqa', 'truthfulqa', 'popqa'][1:]
formal_dataname_list = ['NQ', 'SciQ', 'TriviaQA', 'TruthfulQA', 'PopQA'][1:]
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
    qa = []
    qa_sim = []
    qa_inf = []

    pc = []
    pc_sim = []
    pc_inf = []

    most_inf_doc = []
    most_inf_score = []

    max_results_df = pd.DataFrame(columns=([ "Model", "Anwer",  "Confidence",  "Max Infl. Score", "Most Influencial Document" ]))
    # max_results_df['Test Data'] = [dataname for dataname in formal_dataname_list for i in list(model_to_formal_name.values())]
    # max_results_df['Search Data'] = ["FT" for i in range(len(model_to_full_name))]
    max_results_df['Model'] = [model for model in list(model_to_formal_name.values())] 


    for model in model_to_full_name.keys():
        
        data = datasets.load_from_disk("/srv/home/groups/dm/share/pre-results_confidence/prob/" + model_to_full_name[model] + "/cos/"+ dataname + '_' + model + '_test/main')
        
        qa.append(data["response"][0])
        # qa_sim.append(data["qa_query"][0][0]["_source"]["text"])
        # max_inf_index = data['qa_query_influence'][0].index(np.nanmax(data['qa_query_influence'][0]))
        # qa_inf.append(data["qa_query"][0][max_inf_index]["_source"]["text"])

        pc.append(str(data["prob_confidence"][0]))
        # pc_sim.append(data["pc_query"][0][0]["_source"]["text"])
        # max_inf_index = data['pc_query_influence'][0].index(np.nanmax(data['pc_query_influence'][0]))
        # pc_inf.append(data["pc_query"][0][max_inf_index]["_source"]["text"]) 
        max_qa_inf = np.nanmax(data['qa_query_influence'][0])
        max_pc_inf = np.nanmax(data['pc_query_influence'][0])
        if max_qa_inf > max_pc_inf:
            text = data["qa_query"][0][data['qa_query_influence'][0].index(max_qa_inf)]["_source"]["text"]
            text = text.replace('_', ' ').replace('\r', ' ').replace('\n', ' ').replace('&', '\&').replace('%', '\%').replace('$', '')
            most_inf_doc.append('[Content-related]: ' + text)
                                
            most_inf_score.append(round(max_qa_inf, 4))
        elif max_qa_inf < max_pc_inf:
            text = data["pc_query"][0][data['pc_query_influence'][0].index(max_pc_inf)]["_source"]["text"]
            text = text.replace('_', ' ').replace('\r', ' ').replace('\n', ' ').replace('&', '\&').replace('%', '\%').replace('$', '')
            most_inf_doc.append('[Confidence-related]: ' + text)
            most_inf_score.append(round(max_pc_inf, 4))
            
    max_results_df["Anwer"] = qa
    max_results_df["Confidence"] = pc
    # max_results_df["qa-Similar"] = qa_sim
    # max_results_df["pc-Similar"] = pc_sim
    # max_results_df["qa-Influencial"] = qa_inf
    # max_results_df["pc-Influencial"] = pc_inf
    max_results_df["Most Influencial Document"] = most_inf_doc
    max_results_df["Max Infl.t Score"] = most_inf_score
    # With more options
    max_latex_table = max_results_df.to_latex(
        index=False,           # Don't show index
        caption="Sample Data", # Add caption
        label="tab:sample",    # Add label for referencing
        position="htbp",       # Table positioning
        column_format="lcr",   # Alignment (left, center, right)
        float_format="%.2f",   # Number formatting
        bold_rows=True,        # Make header row bold
        escape=False           # Don't escape special LaTeX characters
    )

    print(max_latex_table)
    max_results_df.to_csv('../results_exp/' + dataname + '_pre-document_all_model.csv', encoding ='utf-8', index=False)