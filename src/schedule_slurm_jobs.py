import argparse
import os
import shutil
import subprocess

from datetime import timedelta

parser = argparse.ArgumentParser("schedule_slurm_jobs")
parser.add_argument("confidence_type", help="Choose one of confidence_prompts")
parser.add_argument("--num_gpus_per_job", help="Number of gpus per job, if unset, then using minimum required for loading model", type=int, nargs="?", const=1, default=1)
parser.add_argument("--num_experiments_per_job", help="Number of experiments per job", type=int, nargs="?", const=1, default=50)
parser.add_argument("--debug", default=False, action='store_true')
parser.add_argument("--force_recompute", help="If set, results get recomputed and overwritten", default=False, action='store_true')
parser.add_argument("--dependency_job_id", help="If specified, an afterok dependency is added to all jobs", type=int, nargs="?", const=1, default=None)

parser.add_argument("--input_folder", help="Input folder with hf datasets", default="/srv/home/groups/dm/share/verb_check_new")
parser.add_argument("--output_folder", help="Output folder for hf datasets", default="/srv/home/groups/dm/share/results_verb_check_new")
parser.add_argument("--batch_size", help="Batch size", type=int, default=100)

parser.add_argument("--mode", help="Either 'cos' or 'dot'", default="cos")
parser.add_argument("--pre",default=False)
args = parser.parse_args()


def resolve_hf_model_string(model):
    if model == 'qwen14b':
        return "Qwen/Qwen2.5-14B-Instruct"
    elif model == 'qwen32b':
        return "Qwen/Qwen2.5-32B-Instruct"
    elif model == 'qwen72b':
        return "Qwen/Qwen2.5-72B-Instruct"
    elif model == 'llama70b':
        return "meta-llama/Llama-3.3-70B-Instruct"
    elif model == "mistral":
        return "mistralai/Mistral-Large-Instruct-2411"
    elif model == 'olmo':
        return "allenai/OLMo-2-1124-7B-Instruct"
    elif model == 'olmo-sft':
        return "allenai/OLMo-2-1124-7B-SFT"
    elif model == 'olmo-dpo':
        return "allenai/OLMo-2-1124-7B-DPO"
    elif model == 'olmo13b':
        return "allenai/OLMo-2-1124-13B-Instruct"
    elif model == 'olmo13b-sft':
        return "allenai/OLMo-2-1124-13B-SFT"
    elif model == 'olmo13b-dpo':
        return "allenai/OLMo-2-1124-13B-DPO"
    elif model == "llama8b-sft":
        return "allenai/Llama-3.1-Tulu-3-8B-SFT"
    elif model == "llama8b":
        return "allenai/Llama-3.1-Tulu-3-8B"
    elif model == "llama8b-dpo":
        return "allenai/Llama-3.1-Tulu-3-8B-DPO"
    elif model == "olmo-1":
        return "allenai/OLMo-7B-Instruct-hf"
    elif model == "olmo-1-sft":
        return "allenai/OLMo-7B-SFT-hf"
        
    else:
        return model # then it is probably a valid hf model string or path

gpu_mappings = [
                (2, ["olmo13b", "olmo13b-dpo", "olmo13b-sft"], "00-12:00:00"),
                (1, ["llama8b-dpo", "llama8b","llama8b-sft","olmo-dpo","olmo", "olmo-sft", "olmo-1", "olmo-1-sft"], "00-12:00:00")
                ]

for gpus_per_proc, models, time_limit_per_experiment in gpu_mappings:
    queue = os.listdir(args.input_folder)
    shutil.copyfile("./slurm_template.sh", "./slurm_combined.sh")
    
    # if args.debug or False:
    #     queue = [queue[0], queue[7]]
                    
    current_num_experiments_per_job = 0
    while len(queue) > 0:
        d = queue.pop()
    
        dataset_shorthand, model_shorthand, split = d.split("_")
        if model_shorthand not in models:
            print("could not resolve", d, "skipping!")
            
        else:  
        

            model = resolve_hf_model_string(model_shorthand)
            dataset = os.path.join(args.input_folder, d)

            r = os.path.join(args.output_folder, os.path.basename(model), args.mode, d)

            c = f"python estimate_training_data_influence.py {args.confidence_type} --batch_size=100 --output_path={args.output_folder} --model={model} --dataset={dataset}  --checkpoint_nr=0 --dataset_split={split+'[0:40]' if args.debug else split}  --gpus_per_proc={int(gpus_per_proc)} --mode={args.mode} --pre={args.pre} --batch_size={args.batch_size}"       
        
            with open("./slurm_combined.sh", 'a') as fd:                
                fd.write(f'\n{c}')
                os.fsync(fd.fileno())
            current_num_experiments_per_job+=1
         

        if (current_num_experiments_per_job >= args.num_experiments_per_job) or len(queue) == 0:   
            time_limit = timedelta(**dict(zip(['days', 'hours', 'minutes', 'seconds'], map(int, time_limit_per_experiment.replace('-', ':').split(':'))))) * current_num_experiments_per_job 
            
            influence_command = [
                "sbatch",
                f"--dependency=afterok:{args.dependency_job_id}" if args.dependency_job_id is not None else "",
                f"--gres=gpu:{max(args.num_gpus_per_job, gpus_per_proc)}",
                # f"--partition=p_datamining",
                f"--time={time_limit.days}-{(time_limit.seconds // 3600):02}:{(time_limit.seconds % 3600) // 60:02}:{(time_limit.seconds % 60):02}",
                "--nodelist=galadriel,dgx-h100-em2",
                f"--job-name=[Yuxi: Verbalized Confidence]:{args.confidence_type}:[{','.join(models)}]",
                "./slurm_combined.sh",
            ]
            influence_command = [c for c in influence_command if c != ""]
            influence_command_str = " ".join([c for c in influence_command])
            if args.debug:
                with open("./slurm_combined.sh", 'r') as fd:
                    content = fd.read()
                    print(content)
            else:
                influence_process = subprocess.run(influence_command, stdout=subprocess.PIPE, text=True, check=True)
            current_num_experiments_per_job = 0
            if len(queue) > 0:
                shutil.copyfile("./slurm_template.sh", "./slurm_combined.sh")
                    



                        
