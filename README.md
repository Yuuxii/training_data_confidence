This repository contains the source code for the paper: 
**_Influential Training Data Retrieval for Explaining Verbalized Confidence of LLMs (ECIR 2026)_**


## Training Dataset Search

We employ the search mechanism developed by WIMBD, which uses Elastic Search to create a keyword-based search engine built on the training data to retrieve similar samples for the search query.

The code for search for pre-training data: `src/search_wimbd.py`.
The code for search for post-training data: `src/search_ft.py`.


## Training Data Influence Experiments

Copy `.env_template` to `.env`

- **Local**: Install conda env from `environment.yaml`
- **SLURM**: `python3 schedule_slurm_jobs.py`


Expects [datasets like this one](https://huggingface.co/datasets/yuxixia/triviaqa-test-tulu3-query), where each row has the fields `question` and `response` that form the *train* document $z_i'$, as well as *n* columns that end in `_query`, each containing a list of documents $\{z'_i,j\}$ of arbritrary lenght. 

For each such column, the script `estimate_training_data_influence.py` adds a new column with lists of pointwise influences $\phi(z_i, z_{i,j})$. 

Results are stored as huggingface datasets in `--output_folder` 


## Evaluation and Analysis

The content-groundness evaluation is implemented in `exp_anly/analysis.py`.
