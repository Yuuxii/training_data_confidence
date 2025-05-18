import argparse
import os
from dotenv import load_dotenv
load_dotenv()
import wandb
import pandas as pd
import logging
from datasets import load_dataset,load_from_disk

from logging.handlers import QueueHandler, QueueListener

from  datasets.features.features import Value, Sequence
import datasets
import time
from multiprocessing import Pool, Queue, Manager
from huggingface_hub import list_repo_refs

import functools
os.environ["TOKENIZERS_PARALLELISM"] = "True"



parser = argparse.ArgumentParser("get_influence_confidence")
parser.add_argument("confidence_type", help="Choose one of confidence_prompts (see below)")
parser.add_argument("--model", help="A model on the hf hub. Format: username/name", default="allenai/OLMo-2-1124-7B-Instruct")
parser.add_argument("--dataset", help="A dataset on the hf hub. Format: username/name", default="")
parser.add_argument("--dataset_split", help="The split to access", default="train[0%:100%]")
parser.add_argument("--checkpoint_nr", help="Id of the checkpoint to extract gradients for (inferred from the repo, starting at 0)",type=int, default=0)
parser.add_argument("--output_path", help="The path where to store the result dataset at", default="./results_confidence")
parser.add_argument("--mode",default="cos")
parser.add_argument("--pre",default=False)


parser.add_argument("--batch_size", help="How many examples are assigned to each process (lifetime. no re-use, mainly affects ram requirements)", type=int, default=100)
parser.add_argument("--gpus_per_proc", help="How many gpus to use per model", type=int, default=1)
args = parser.parse_args()


def get_influence_batch(batch, start_idx, stop_idx, completion_times_gradients):
        """Computes pointwise influece in batches

        Args:
            ds: A slice of the dataset (pandas format)

        Returns:
            The dataset with influence scores added as new columns
        """
        # setup
        
   
        logger = setup_logging_subprocess()
        logger.debug(f"Process {start_idx}_{stop_idx} is starting...")
      
        gpus = [gpu_queue.get() for _ in range(args.gpus_per_proc)]

        print(f"Process {start_idx}_{stop_idx} is using gpus {gpus}", flush=True)
        logger.debug(f"Process {start_idx}_{stop_idx} is using gpus {gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in gpus])
        print( ' os.environ["CUDA_VISIBLE_DEVICES"]',os.environ["CUDA_VISIBLE_DEVICES"])

        import torch
        from transformers import  DataCollatorForLanguageModeling
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.debug(f"Process {start_idx}_{stop_idx} is loading model")
        model = AutoModelForCausalLM.from_pretrained(args.model, revision="main", torch_dtype=torch.float16, device_map="auto")
        model.train()
 
        device = "cuda"
       

        logger.debug(f"Process {start_idx}_{stop_idx} is loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if "tulu" in args.model:
            tokenizer.pad_token = tokenizer.eos_token

        data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=False
                )

            

        def preprocess_tulu_confidence(row, include_confidence_prompt=True,include_q=True):
            confidence_prompts = {
                "ans": f"Answer the question, give ONLY the answer, no other words or explanation:\n\n",
                "prob": f"Provide the probability that your answer is correct. Give ONLY the probability between 0.0 and 1.0, no other words or explanation.",
            }
            chat = None
            if include_confidence_prompt and include_q:
                chat = [
                    
                        {"role": "user", "content": f"Answer the question, give ONLY the answer, no other words or explanation:\n\n" + row["question"] if row["question"] is not None else ""},
                            {"role": "assistant", "content": row["response"] if row["response"] is not None else ""},
                            {"role": "user", "content": confidence_prompts[args.confidence_type]}
                    ]
                

            elif include_q and not include_confidence_prompt:
                chat = [
                    
                        {"role": "user", "content": f"Answer the question, give ONLY the answer, no other words or explanation:\n\n" + row["question"] if row["question"] is not None else ""},
                            {"role": "assistant", "content": row["response"] if row["response"] is not None else ""},
             
                ]       
            elif include_confidence_prompt and not include_q:        
                chat = [
                            {"role": "user", "content": confidence_prompts[args.confidence_type]},
                            {"role": "assistant", "content": str(row["prob_confidence"]) if row["prob_confidence"] is not None else ""},
                ]
            else:
                raise NotImplementedError
            return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


        def get_loss_gradient(model, example,device):
            """Computes gradient of the loss function irt to the input embeddings.

            Args:
                model: A model with a `forward` method wich returns `loss`
                example: An instance from the training data
                device: What GPU to use (e.g., cuda:0)

            Returns:
                A 1D tensor: gradient of the loss function irt to the input embeddings
            """
            
            model.zero_grad()
            

            input_ids, labels = data_collator(example).values()
           
            inputs_embeds=model.get_input_embeddings().weight[input_ids].to(device)
            inputs_embeds.retain_grad()

            outputs = model.forward(
                    inputs_embeds=inputs_embeds,
                    labels=labels.to(device)
                )
            loss = outputs.loss
            loss.retain_grad()
            return torch.autograd.grad(loss, inputs_embeds, retain_graph=False)[0].squeeze()
            
        
        def process_row(row):
            """Processes one row (question, answer, a_query [list:string], b_query [list:string], ...). Adds a new column with lists of pointwise influence scores between z and z_i' in x_query.

            Args:
                row: An example of the form (question, answer , a_query [list:string], b_query [list:string], ...)

            Returns:
                A dataset where for each column ending in "_query" a new one ending in "_query_influence" is added with lists of pointwise influence scores
            """
            z_train = tokenizer(preprocess_tulu_confidence(row),return_special_tokens_mask=False, truncation=True, padding="max_length", max_length=4096,return_tensors="pt",padding_side='left')["input_ids"]
            grad_z_train = get_loss_gradient(model, z_train,device).detach().flatten().to(torch.float32)

            def cos(example):
                z_test = tokenizer(example,return_special_tokens_mask=False, truncation=True, padding="max_length", max_length=4096,return_tensors="pt",padding_side='left')["input_ids"]
                grad_z_test = get_loss_gradient(model, z_test,device).detach().flatten().to(torch.float32)
                return torch.nn.functional.cosine_similarity(grad_z_train, grad_z_test,dim=0).cpu().item()
            cos = functools.cache(cos)

            def dot(example):
                z_test = tokenizer(example,return_special_tokens_mask=False, truncation=True, padding="max_length", max_length=4096,return_tensors="pt",padding_side='left')["input_ids"]
                grad_z_test = get_loss_gradient(model, z_test,device).detach().flatten().to(torch.float32)
                return torch.dot(grad_z_train, grad_z_test).cpu().item()
            dot = functools.cache(dot)

            if args.mode in ["cos", "dot"]:
                for test_col in  [ key for key in batch.columns if "_query" in key]: # TODO hotfix
                    def get_chat(x):
                        if "message" in x:
                            return x["message"]
                        elif "messages" in x:
                            return x["messages"]
                        elif "chosen" in x:
                            return x["chosen"]
                        else:
                            raise NotImplementedError
                    
        
                    examples = [tokenizer.apply_chat_template(get_chat(t["_source"]), tokenize=False, add_generation_prompt=True) if not args.pre else t["_source"]["text"] for t in row[test_col]]
                
                    row[test_col+"_influence"] = []
                    for _, example in enumerate(examples):
                        if args.mode == "cos":
                            row[test_col+"_influence"].append(cos(example))
                        else:
                            row[test_col+"_influence"].append(dot(example))
                    
                return row
            elif args.mode == "random":
                test_col = "random"
              
                def get_chat(x):
                    if "message" in x:
                        return x["message"]
                    elif "messages" in x:
                        return x["messages"]
                    elif "chosen" in x:
                        return x["chosen"]
                    else:
                        raise NotImplementedError
                
    
                examples = [tokenizer.apply_chat_template(get_chat(t["_source"]), tokenize=False, add_generation_prompt=True) if not args.pre else t["_source"]["text"] for t in row[test_col]]
            
                row["random_influence"] = []
                for _, example in enumerate(examples):
                    row["random_influence"].append(cos(example))
                    
                    
                return row
            elif args.mode == "qa_influence":   
                example = preprocess_tulu_confidence(row, include_confidence_prompt=False, include_q=True) 
                row["qa_influence"] = cos(example)
                return row
            elif args.mode == "pc_influence":   
                example = preprocess_tulu_confidence(row, include_confidence_prompt=True, include_q=False) 
                row["pc_influence"] = cos(example)
                return row
            
            else:
                raise NotImplementedError


        tmp_result_dir = os.path.join("./tmp" if not args.pre else "./tmp_pre", args.confidence_type, os.path.basename(args.model),args.mode, os.path.basename(args.dataset), "main")
        os.makedirs(tmp_result_dir, exist_ok=True)

        try:
            result =  pd.read_pickle(os.path.join(tmp_result_dir, f"{start_idx}_{stop_idx}.pkl"))
            logger.info(f"Re-using stored results for {start_idx}_{stop_idx}") 
            [gpu_queue.put(i) for i in gpus]
            return result
        except:
            logger.info(f"Processing batch {start_idx}_{stop_idx}")
            start_time = time.time()
            result = batch.apply(process_row, axis=1)
            result.to_pickle(os.path.join(tmp_result_dir, f"{start_idx}_{stop_idx}.pkl"))
            completion_times_gradients.append(time.time() - start_time)
            [gpu_queue.put(i) for i in gpus]
            return result
      
    
    
log_queue = Queue()

import sys

def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        logger.addHandler(handler)

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    queue_listener = QueueListener(log_queue, handler, respect_handler_level=True)
    queue_listener.start()

    return queue_listener


def setup_logging_subprocess():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        queue_handler = QueueHandler(log_queue)
        logger.addHandler(queue_handler)

    return logger





def get_checkpoints_olmo(model_name="allenai/OLMo-2-1124-7B"):

    
    out = list_repo_refs(model_name)
    branches = [b.name for b in out.branches]
    checkpoints_stage_1 = [[branch] + branch.split("-") for branch in branches if branch != "main" and "stage1" in branch]
    checkpoints_stage_1 = [sublist[:2] + ["-"] + sublist[2:] for sublist in checkpoints_stage_1]
    checkpoints_stage_1 = sorted(checkpoints_stage_1, key=lambda checkpoint: int(checkpoint[3].replace("step","")))

    checkpoints_stage_2 = [[branch] + branch.split("-") for branch in branches if branch != "main" and "stage2" in branch]
    checkpoints_stage_2 = sorted(checkpoints_stage_2, key=lambda checkpoint: (int(checkpoint[2].replace("ingredient","")), int(checkpoint[3].replace("step",""))))

    checkpoints = checkpoints_stage_1 + checkpoints_stage_2 + [["main", "-", "-","-","-"]]
    checkpoint_names, _,_,_,_ = zip(*checkpoints)
    return checkpoint_names

def get_checkpoints_hub(model):

    return get_checkpoints_olmo(model)




checkpoints =  get_checkpoints_hub(args.model)



gpu_queue = Queue()



from datasets import Dataset, Features, Value



import psutil
def get_pool_memory_usage(pool):
    memory_usage = 0
    for process in pool._pool:
        try:
            proc = psutil.Process(process.pid)
            memory_info = proc.memory_info()
            memory_usage += memory_info.rss 
        except psutil.NoSuchProcess:
            pass
    return memory_usage / (1024 ** 3)

if __name__ == '__main__':
    from multiprocess import set_start_method
    set_start_method("spawn")
    model_name = args.model.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1]
    dataset_split_name = args.dataset_split

    setup_logging()
    
    run = wandb.init(project="confidence_gradient_extraction")
    run.name = "_".join(["JOB_ID="+ str(os.getenv("SLURM_JOB_ID", "?")), os.path.basename(args.model), os.path.basename(args.dataset), str(args.checkpoint_nr)])
    
    

    logging.info("Loading dataset in main process")
    dataset = None
    try:
        dataset = load_dataset(args.dataset, split=args.dataset_split)
    except:
        dataset = load_from_disk(args.dataset)
        logging.warning("Slicing not supported for local datasets, loading full train split")

    print("len dataset", len(dataset))
    checkpoints =  get_checkpoints_hub(args.model)

    import torch
    
    num_proc = torch.cuda.device_count() // args.gpus_per_proc

    logging.info(f"{num_proc * args.gpus_per_proc} gpus in queue")
    # set up gpu queue to allocate gpus evenly between processes
    for i in range(num_proc * args.gpus_per_proc):
        gpu_queue.put(i)

    
    pool_gradients = Pool(num_proc)
    
    checkpoint = checkpoints[args.checkpoint_nr]


    manager = Manager()
    completion_times_gradients = manager.list() # the processes report back to the main process wich handles web logging

    dataset = dataset.with_format("pandas")
    features = dataset.features.copy()#
    if args.mode in ["cos", "dot"]:
        features.update({feature+"_influence": Sequence(Value(dtype='float64', id=None), length=-1, id=None) for feature in dataset.features if "_query" in feature})
    elif args.mode =="random":
        features.update({"random_influence": Sequence(Value(dtype='float64', id=None), length=-1, id=None)})

    else:
        features.update({
            args.mode: Value(dtype='float64', id=None),
            })
    # create tasks for subprocesses
    batch_indices = [(i,min(i + args.batch_size,len(dataset))) for i in range(0, len(dataset), args.batch_size)]

    print(batch_indices)
    tasks = [(dataset[start_idx:stop_idx], start_idx, stop_idx, completion_times_gradients) for start_idx, stop_idx in batch_indices]
    logging.info("Start processing")
    r = pool_gradients.starmap_async(get_influence_batch, tasks)   

    # web logging
    while not r.ready(): # loop to not block the main process
        while len(completion_times_gradients) > 0:
            run.log({"gradients/time_per_batch": completion_times_gradients.pop()},commit=True)
        run.log({"gradients/pool_ram_usage":get_pool_memory_usage(pool_gradients)}, commit=True)
        time.sleep(10)  
        
    logging.info("Got gradients for checkpoint-{}".format(checkpoint))


    results = r.get()
    print(results)
    p = datasets.concatenate_datasets([Dataset.from_pandas(df, features=features) for df in results])
    p_path = os.path.join(args.output_path, args.confidence_type, os.path.basename(args.model),args.mode, os.path.basename(args.dataset), "main")
    logging.info(f"Saving results to {p_path}")
    p.save_to_disk(p_path)

    pool_gradients.close()
    pool_gradients.join()
