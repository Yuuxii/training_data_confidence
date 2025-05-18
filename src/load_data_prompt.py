import random
import json
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset
import ast
from fastchat.conversation import get_conv_template
seed = 42
random.seed(seed)

data_path = ''
model_list = ['olmo13b', 'olmo13b-dpo', 'olmo13b-sft', 'olmo', 'olmo-dpo', 'olmo-sft']
formal_model_list = ['OLMo-13B INS', 'OLMo-13B DPO', 'OLMo-13B SFT', 'OLMo-7B INS', 'OLMo-7B DPO', 'OLMo-7B SFT']
dataname_list = ['nq', 'sciq', 'triviaqa', 'truthfulqa', 'popqa']
formal_dataname_list = ['NQ', 'SciQ', 'TriviaQA', 'TruthfulQA', 'PopQA']
formal_prompt_names =['P(1)']
# dataname_list = ['popqa']

def ans_prompt():

    prompt = f"""Answer the question, give ONLY the answer, no other words or explanation:\n\n"""
    
    return prompt


def origin_prob_prompt():

    prompt = f"""Provide the probability that your answer is correct. """
    prompt += f"""Give ONLY the probability between 0.0 and 1.0, no other words or explanation."""

    return prompt


def confi_prompt():

    prompt = f"""Provide the confidence that your answer is correct. """ 
    prompt += f"""Give ONLY the confidence between 0.0 and 1.0, no other words or explanation."""
    
    return prompt

def certain_prompt():

    prompt = f"""Provide the certainty that your answer is correct. """ 
    prompt += f"""Give ONLY the certainty between 0.0 and 1.0, no other words or explanation."""
    
    return prompt


prompt_type_dict = {
    "prob": origin_prob_prompt,
    
}

all_prompt_type = {
    "ans": ans_prompt(),
    "prob": origin_prob_prompt,
}


def load_data(dataname):
    
    train_data = []
    test_data = []

    if dataname == 'truthfulqa':
        ## load test data
        test = load_dataset("truthfulqa/truthful_qa", "generation", split='validation')
        for item in test:
            question = item['question'].strip()
            if question[-1] != '?':
                question = question +'?'
            test_data.append({
                "question": item['question'],
                "target": item['best_answer']
            }) 
        
    if dataname == "popqa":
        ds = load_dataset("akariasai/PopQA", split="test")
        test = ds.shuffle(seed=42).select(range(1000))
        for item in test:
            question = item['question'].strip()
            if question[-1] != '?':
                question = question +'?'
            test_data.append({
                "question": item['question'],
                "target": ast.literal_eval(item['possible_answers'])[0]
            })

    if dataname in ['triviaqa', 'sciq', 'wikiqa', 'nq']:

        dataset_path = data_path + dataname 

        ## load train data
        train = pd.read_csv(dataset_path + "/train.csv", encoding='utf-8')
        
        for i in train.index.tolist():
            train_data.append({
                "question": train['question'][i],
                "target": train['answer'][i][1:-1]
            })

        ## load test data
        test = []
        with open(dataset_path + "/testLlama-2-7b-hftemperature_1topp_1.0_num_demos_15_answers.1.reeval.semantic_uncertainty.gpt4_correctness.jsonl", encoding='utf-8') as f:
            for line in f:
                test.append(json.loads(line))
    
        for item in test:
            question = item['question'].strip()
            if question[-1] != '?':
                question = question +'?'
            test_data.append({
                "question": question,
                "target": item['gold_answer']
            })

    return train_data, test_data

class QDataset(Dataset):
    def __init__(self, dataname, tokenizer=None):

        
        self.train_data, self.data = load_data(dataname)
        
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        question = self.data[idx]['question'].strip()

        prompts = [{"role": "user", "content": ans_prompt() + question}]
        
        if self.tokenizer is not None:
            prompts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

        return prompts

        
 
class QADataset(Dataset):

    def __init__(self, prompt_type, responses, tokenizer=None):

        self.tokenizer = tokenizer
        self.prompt_type = prompt_type
        self.questions = responses['question'].tolist()
        self.answers = responses['response'].tolist()

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):

        question = self.questions[idx].strip()

        answer = self.answers[idx]
        prompts = [{"role": "user", "content": ans_prompt() + question},
                    {"role": "assistant", "content": str(answer)},
                    {"role": "user", "content": prompt_type_dict[self.prompt_type]()}]

        if self.tokenizer is not None:
            prompts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)

        return prompts



class hf_EvalDataset(Dataset):
    def __init__(self, response, apply_temp=False):
        
        self.questions = response['question'].tolist()
        self.answers = response['response'].tolist()
        self.targets = response['target'].tolist()
        self.apply_temp = apply_temp
    
    def _get_conversation_prompt(self, messages):
        """
        From filled prompt, convert it into llama-2 conversation prompt
        """
        conv = get_conv_template("mistral")

        for message in messages:
            if message["role"] == "system":
                conv.set_system_message(message["content"])
            elif message["role"] == "user":
                conv.append_message(conv.roles[0], message["content"])

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt
    
    def __len__(self):
        # this should return the size of the dataset
        return len(self.questions)
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        question = self.questions[idx]
        answer = self.answers[idx]
        target = self.targets[idx]

        ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

        ABSOLUTE_PROMPT = f"""###Task Description:
        An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 1, and a score rubric representing an evaluation criteria are given.
        1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
        2. After writing a feedback, write a score that is an integer between 0 and 1. You should refer to the score rubric.
        3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 0 and 1)\"
        4. Please do not generate any other opening, closing, and explanations.

        ###The instruction to evaluate:
        {question}

        ###Response to evaluate:
        {answer}

        ###Reference Answer (Score 1):
        {target}

        ###Score Rubrics:loloa
        Score 0: for answering the instruction, the response is wrong or not semantically equivalent to the reference answer. 
        Score 1: for answering the instruction, the response is correct or semantically equivalent to the reference answer. 

        ###Feedback:  """
        
        messages = [
                {"role": "system", "content": ABS_SYSTEM_PROMPT},
                {"role": "user", "content": ABSOLUTE_PROMPT},
            ]
        
        if self.apply_temp:
            prompts = self._get_conversation_prompt(messages)
        else:
            prompts = messages

        # prompts = self.tokenizer.apply_chat_template(messages, tokenize=False)

        return prompts