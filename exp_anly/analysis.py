import datasets
import pandas as pd
import numpy as np
import argparse
from scipy.stats import ttest_ind
from itertools import zip_longest
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

def compare_vectors_with_ttest(vector1, vector2, equal_var=False, nan_policy='omit'):
    """
    Compare two 2D vectors using an independent t-test.
    
    Args:
        vector1 (np.ndarray): 2D array (num_samples × 10).
        vector2 (np.ndarray): 2D array (num_samples × 10).
        equal_var (bool): If True, assumes equal variances (standard t-test).
                         If False (default), uses Welch’s t-test (unequal variances).
        nan_policy (str): How to handle NaN values ('omit' or 'raise').
    
    Returns:
        t_statistic (float): T-test statistic.
        p_value (float): Two-tailed p-value.
    """
    # Flatten the arrays and remove NaNs
    flat1 = vector1.flatten()
    flat2 = vector2.flatten()
    
    # Run independent t-test
    t_stat, p_value = ttest_ind(
        flat1, flat2,
        equal_var=equal_var,
        nan_policy=nan_policy
    )
    
    return t_stat, p_value


def compute_win_ratio(vector1, vector2):
    """
    Compute the win ratio between two 2D vectors.
    
    Args:
        vector1: 2D numpy array (num_samples × 10), may contain NaN values
        vector2: 2D numpy array (num_samples × 10), may contain NaN values
    
    Returns:
        ratio: win ratio of vector1 to vector2
        total_wins1: total winning cases for vector1 across all samples
        total_wins2: total winning cases for vector2 across all samples
    """
    # Ensure inputs are numpy arrays and have the same shape
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    assert vector1.shape == vector2.shape, "Input vectors must have the same shape"
    
    total_wins1 = 0
    total_wins2 = 0
    total_valid_comparisons = 0
    
    for sample1, sample2 in zip(vector1, vector2):
        # For each sample, compare all pairwise scores (10×10 comparisons)
        wins1 = 0
        wins2 = 0
        valid_comparisons = 0
        
        for score1 in sample1:
            for score2 in sample2:
                # Skip if either score is NaN
                if np.isnan(score1) or np.isnan(score2):
                    continue
                
                if score1 > score2:
                    wins1 += 1
                elif score1 < score2:
                    wins2 += 1
                # if equal, no wins for either
                
                valid_comparisons += 1
        
        if valid_comparisons > 0:
            # Compute average wins for this sample
            avg_wins1 = wins1 / valid_comparisons
            avg_wins2 = wins2 / valid_comparisons
            
            total_wins1 += avg_wins1
            total_wins2 += avg_wins2
            total_valid_comparisons += 1
    
    if total_valid_comparisons == 0:
        return np.nan, 0, 0
    
    # Compute the ratio (vector1 wins relative to vector2 wins)
    if total_wins2 == 0:
        ratio = np.inf if total_wins1 > 0 else np.nan
    else:
        ratio = total_wins1 / total_wins2
    
    return ratio, total_wins1, total_wins2

def compare_max_avg_influence_score(search_data):
    max_results_df = pd.DataFrame(columns=(['Data', "Model"] + formal_dataname_list) + ["All"])
    max_results_df['Data'] = [search_data for i in range(len(model_to_full_name))]
    max_results_df['Model'] = list(model_to_formal_name.values())

    avg_results_df = pd.DataFrame(columns=(['Data', "Model"] + formal_dataname_list + ["All"]))
    avg_results_df['Data'] = [search_data for i in range(len(model_to_full_name))]
    avg_results_df['Model'] = list(model_to_formal_name.values())

    max_overall_response_wins = dict.fromkeys(model_to_full_name.keys(), 0)
    max_overall_confidence_wins = dict.fromkeys(model_to_full_name.keys(), 0)
    avg_overall_response_wins = dict.fromkeys(model_to_full_name.keys(), 0)
    avg_overall_confidence_wins = dict.fromkeys(model_to_full_name.keys(), 0)
    for dataname, formal_dataname in zip(datanames, formal_dataname_list):
        max_results = []
        max_tie_results = []
        avg_results = []
        avg_tie_results = []

        for model in model_to_full_name.keys():

            if search_data in ["PT", "FT"]:

                if search_data == "PT":
                    # data = datasets.load_from_disk("/srv/home/groups/dm/share/pre-results_confidence/prob/" + model_to_full_name[model] + "/cos/"+ dataname + '_' + model + '_test/main')
                    data = datasets.load_from_disk("/srv/home/groups/dm/share/results_verb_check/prob/" + model_to_full_name[model] + "/random/"+ dataname + '_' + model + '_test/main')
                elif search_data == "FT":
                    data = datasets.load_from_disk("/srv/home/groups/dm/share/results_confidence/prob/" + model_to_full_name[model] + "/cos/"+ dataname + '_' + model + '_test/main')

                max_tie_cases = 0
                max_response_wins = 0
                max_confidence_wins = 0

                avg_tie_cases = 0
                avg_response_wins = 0
                avg_confidence_wins = 0

                for i  in range(len(data)):
                    # print(i)
                    if len(data['qa_query_influence'][i]) != 0 and len(data['pc_query_influence'][i]) != 0:
                    # print("data['confidence_query_influence'][i]", data['confidence_query_influence'][i])
                    # print("data['response_query_influence'][i]", data['response_query_influence'][i])
                        max_confidence_infl= np.nanmax(data['pc_query_influence'][i])
                        max_response_infl = np.nanmax(data['qa_query_influence'][i])
                        avg_confidence_infl= np.nanmean(data['pc_query_influence'][i])
                        avg_response_infl = np.nanmean(data['qa_query_influence'][i])

                        if max_confidence_infl > max_response_infl:
                            max_confidence_wins+=1
                        elif max_confidence_infl < max_response_infl:
                            max_response_wins+=1
                        else:
                            max_tie_cases +=1

                        if avg_confidence_infl > avg_response_infl:
                            avg_confidence_wins+=1
                        elif avg_confidence_infl < avg_response_infl:
                            avg_response_wins+=1
                        else:
                            avg_tie_cases +=1

                max_results.append(round(max_response_wins/max_confidence_wins, 2))
                max_tie_results.append(round(max_tie_cases/len(data), 2))

                avg_results.append(round(avg_response_wins/avg_confidence_wins, 2))
                avg_tie_results.append(round(avg_tie_cases/len(data), 2))

                max_overall_response_wins[model] += max_response_wins
                max_overall_confidence_wins[model] += max_confidence_wins

                avg_overall_response_wins[model] += avg_response_wins
                avg_overall_confidence_wins[model] += avg_confidence_wins
            
            else:
                ### search name = ft+pt
                pt_data = datasets.load_from_disk("/srv/home/groups/dm/share/pre-results_confidence/prob/" + model_to_full_name[model] + "/cos/"+ dataname + '_' + model + '_test/main')

                ft_data = datasets.load_from_disk("/srv/home/groups/dm/share/results_confidence/prob/" + model_to_full_name[model] + "/cos/"+ dataname + '_' + model + '_test/main')

                max_response_wins = 0
                max_confidence_wins = 0
                max_tie_cases = 0
                avg_response_wins = 0
                avg_confidence_wins = 0
                avg_tie_cases = 0

                for i  in range(len(pt_data)):
                    # print(i)
                    if len(pt_data['qa_query_influence'][i]) != 0 and len(pt_data['pc_query_influence'][i]) != 0 and len(ft_data['qa_query_influence'][i]) != 0 and len(ft_data['pc_query_influence'][i]) != 0:

                        ft_max_confidence_infl= np.nanmax(ft_data['pc_query_influence'][i])
                        ft_max_response_infl = np.nanmax(ft_data['qa_query_influence'][i])
                        ft_avg_confidence_infl= np.nanmean(ft_data['pc_query_influence'][i])
                        ft_avg_response_infl = np.nanmean(ft_data['qa_query_influence'][i])

                        pt_max_confidence_infl= np.nanmax(pt_data['pc_query_influence'][i])
                        pt_max_response_infl = np.nanmax(pt_data['qa_query_influence'][i])
                        pt_avg_confidence_infl= np.nanmean(pt_data['pc_query_influence'][i])
                        pt_avg_response_infl = np.nanmean(pt_data['qa_query_influence'][i])

                        if max(ft_max_confidence_infl,pt_max_confidence_infl) > max(ft_max_response_infl, pt_max_response_infl):
                            max_confidence_wins+=1
                        elif max(ft_max_confidence_infl,pt_max_confidence_infl) < max(ft_max_response_infl, pt_max_response_infl):
                            max_response_wins+=1
                        else:
                            max_tie_cases+=1

                        if max(ft_avg_confidence_infl, pt_avg_confidence_infl) > max(ft_avg_response_infl, pt_avg_response_infl):
                            avg_confidence_wins+=1
                        elif max(ft_avg_confidence_infl, pt_avg_confidence_infl) < max(ft_avg_response_infl, pt_avg_response_infl):
                            avg_response_wins+=1
                        else:
                            avg_tie_cases+=1



                max_results.append(round(max_response_wins/max_confidence_wins, 2))
                # max_tie_results.append(round(max_tie_cases/len(data), 2))

                avg_results.append(round(avg_response_wins/avg_confidence_wins, 2))
                # avg_tie_results.append(round(avg_tie_cases/len(data), 2))

                max_overall_response_wins[model] += max_response_wins
                max_overall_confidence_wins[model] += max_confidence_wins

                avg_overall_response_wins[model] += avg_response_wins
                avg_overall_confidence_wins[model] += avg_confidence_wins


        max_results_df[formal_dataname] = max_results
        # max_results_df[formal_dataname + ' tie'] = max_tie_results

        avg_results_df[formal_dataname] = avg_results
        # avg_results_df[formal_dataname + ' tie'] = max_tie_results

    max_results_df['All'] = [round(max_overall_response_wins[model]/max_overall_confidence_wins[model], 2) for model in model_to_full_name.keys()]
    avg_results_df['All'] = [round(avg_overall_response_wins[model]/avg_overall_confidence_wins[model], 2) for model in model_to_full_name.keys()]
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
    avg_latex_table = avg_results_df.to_latex(
        index=False,           # Don't show index
        caption="Sample Data", # Add caption
        label="tab:sample",    # Add label for referencing
        position="htbp",       # Table positioning
        column_format="lcr",   # Alignment (left, center, right)
        float_format="%.2f",   # Number formatting
        bold_rows=True,        # Make header row bold
        escape=False           # Don't escape special LaTeX characters
    )
    print(search_data)
    print(max_latex_table)
    print(avg_latex_table)
    max_results_df.to_csv('../results_exp/' + search_data + '_max_all_model.csv', encoding ='utf-8', index=False)
    avg_results_df.to_csv('../results_exp/' + search_data + '_avg_all_model.csv', encoding ='utf-8', index=False)

def compare_overall_influence_score(search_data):
    name_list = []
    for name in formal_dataname_list:
        name_list.append(name)
        name_list.append(name+'_p')
    max_results_df = pd.DataFrame(columns=(['Data', "Model"] + name_list) + ["All", "All_p"])
    max_results_df['Data'] = [search_data for i in range(len(model_to_full_name))]
    max_results_df['Model'] = list(model_to_formal_name.values())

    p_results_df = pd.DataFrame(columns=(['Data', "Model"] + formal_dataname_list + ["All"]))
    p_results_df['Data'] = [search_data for i in range(len(model_to_full_name))]
    p_results_df['Model'] = list(model_to_formal_name.values())

    max_overall_response_wins = dict.fromkeys(model_to_full_name.keys(), 0)
    max_overall_confidence_wins = dict.fromkeys(model_to_full_name.keys(), 0)
    corrects = [[] for i in model_to_full_name.keys()]

    if search_data in ["PT", "FT", "random"]:
        dim = 10
    elif search_data in ['orig']:
        dim= 1
    else:
        dim = 20

    all_response = dict.fromkeys(model_to_full_name.keys(), np.empty((0, dim)))
    all_confidence = dict.fromkeys(model_to_full_name.keys(), np.empty((0, dim)))

    for dataname, formal_dataname in zip(datanames, formal_dataname_list):
        max_results = []
        p_values = []
        acces = []
        for i, model in enumerate(model_to_full_name.keys()):

            if search_data in ["PT", "FT", 'orig', 'random']:

                if search_data == "PT":
                    data = datasets.load_from_disk("/srv/home/groups/dm/share/pre-results_confidence/prob/" + model_to_full_name[model] + "/cos/"+ dataname + '_' + model + '_test/main')
                    qa = list(zip(*zip_longest(*data['qa_query_influence'], fillvalue=np.nan))) 
                    pc = list(zip(*zip_longest(*data['pc_query_influence'], fillvalue=np.nan))) 

                elif search_data == "FT":
                    data = datasets.load_from_disk("/srv/home/groups/dm/share/results_confidence/prob/" + model_to_full_name[model] + "/cos/"+ dataname + '_' + model + '_test/main')
                    qa = list(zip(*zip_longest(*data['qa_query_influence'], fillvalue=np.nan))) 
                    pc = list(zip(*zip_longest(*data['pc_query_influence'], fillvalue=np.nan))) 

                elif search_data == "random":
                    print("random")
                    data = datasets.load_from_disk("/srv/home/groups/dm/share/results_verb_check/prob/" + model_to_full_name[model] + "/random/"+ dataname + '_' + model + '_test/main')
                    qa = list(zip(*zip_longest(*data['qa_query_influence'], fillvalue=np.nan))) 
                    pc = list(zip(*zip_longest(*data['random_influence'], fillvalue=np.nan))) 

                else:
                    qa_data = datasets.load_from_disk("/srv/home/groups/dm/share/results_confidence/prob/" + model_to_full_name[model] + "/qa_influence/"+ dataname + '_' + model + '_test/main')        
                    pc_data = datasets.load_from_disk("/srv/home/groups/dm/share/results_confidence/prob/" + model_to_full_name[model] + "/pc_influence/"+ dataname + '_' + model + '_test/main')    

                    qa = [[s] for s in qa_data['qa_influence']]
                    pc = [[s] for s in pc_data['pc_influence']]
            else:
                ### search name = ft+pt
                pt_data = datasets.load_from_disk("/srv/home/groups/dm/share/pre-results_confidence/prob/" + model_to_full_name[model] + "/cos/"+ dataname + '_' + model + '_test/main')

                ft_data = datasets.load_from_disk("/srv/home/groups/dm/share/results_confidence/prob/" + model_to_full_name[model] + "/cos/"+ dataname + '_' + model + '_test/main')

                qa_list = [ a + b for a, b in zip(pt_data['qa_query_influence'],ft_data['qa_query_influence'])]
                pc_list = [ a + b for a, b in zip(pt_data['pc_query_influence'],ft_data['pc_query_influence'])]

                qa = list(zip(*zip_longest(*qa_list, fillvalue=np.nan))) 
                pc = list(zip(*zip_longest(*pc_list, fillvalue=np.nan))) 


            qa_vector = np.array(qa)
            pc_vector = np.array(pc)
            all_response[model] = np.concatenate((all_response[model],qa_vector))
            all_confidence[model] = np.concatenate((all_confidence[model],pc_vector))

            ratio, qa_wins, pc_wins = compute_win_ratio(qa_vector, pc_vector)
            _, p_value = compare_vectors_with_ttest(qa_vector, pc_vector)
            # print(max_response_wins, max_confidence_wins)
            max_results.append(round(ratio, 2))
            p_values.append('$\checkmark$' if p_value<0.05 else '$\times$')
            try:
                acces.append(np.mean(data['prom_score'])*100)
                corrects[i].extend(data['prom_score'])
            except:
                print(model)
            # max_tie_results.append(round(max_tie_cases/len(data), 2))

            max_overall_response_wins[model] += qa_wins
            max_overall_confidence_wins[model] += pc_wins
            

        max_results_df[formal_dataname] = max_results
        # max_results_df[formal_dataname+ '_p'] = acces
        p_results_df[formal_dataname] = p_values
            # max_results_df[formal_dataname + ' tie'] = max_tie_results
    # print(corrects)
   
    max_results_df['All'] = [round(max_overall_response_wins[model]/max_overall_confidence_wins[model], 2) for model in model_to_full_name.keys()]
    # max_results_df['All_p'] = [np.mean(corrects[i])*100 for i in range(len(model_to_full_name))]
    p_results_df['All'] = ['$\checkmark$' if compare_vectors_with_ttest(all_response[model], all_confidence[model])[1] <0.05 else '$\times$' for model in model_to_full_name.keys()]
    # With more options
    max_results_df.to_csv('results/overall_results.csv', encoding='utf-8', float_format='%11.2f', index=False)
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
    print(p_results_df)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Index documents stored in jsonl.gz files"
    )
    parser.add_argument("--search_data", type=str, required=True)
    args = parser.parse_args()

    compare_overall_influence_score(args.search_data)