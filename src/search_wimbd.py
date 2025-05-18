
from functools import cache
import logging
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
import argparse
from elasticsearch import Elasticsearch
import pandas as pd
from load_data_prompt import confi_prompt
import datasets 
import yaml
import os
import urllib3
import concurrent.futures
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path
import time
from load_data_prompt import dataname_list
from collections import OrderedDict
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

DEFAULT_CONFIG_LOCATION = "src/es_config.yml"

@cache
def es_init(config: Path = DEFAULT_CONFIG_LOCATION, cloud_id: str = None, api_key: str=None, timeout: int = 360) -> Elasticsearch:
    """
    :param config: Path to the config yaml file, containing `cloud_id` and `api_key` fields.
    :return: Authenticated ElasticSearch client.
    """
    # global es
    if config:
        with open(config) as file_ref:
            config = yaml.safe_load(file_ref)
    else:
        config = {
            "cloud_id": cloud_id,
            "api_key": api_key
        }
    
    if config == DEFAULT_CONFIG_LOCATION:
        logger.warning("Using default config file. This will unlikely work unless you're the creator of this library. Please make sure to specify the ES config file and provide it to the function you are using.")

    cloud_id = config["cloud_id"]
    api_key = config.get("api_key", os.getenv("ES_API_KEY", None))
    if not api_key:
        raise RuntimeError(
            f"Please specify ES_API_KEY environment variable or add api_key to {DEFAULT_CONFIG_LOCATION}."
        )

    es = Elasticsearch(
        cloud_id=cloud_id,
        api_key=api_key,
        retry_on_timeout=True,
        http_compress=True,
        request_timeout=timeout,
        # connections_per_node=1,   # Isolate connections
        # ssl_show_warn=True,
        # verify_certs=True,       # Always verify certs in cloud
        max_retries=3,
        
    )

    return es

def _query_documents_contain_phrases(
    phrases: List[str],
    index_name: str, 
):
   
    which_bool = "should"
    minimum_should_match = 1

    for phrase in phrases:
        # match_query.append({"match_phrase": {"text": {"query": phrase, "slop": slop}}})
        yield {'index': index_name}
        yield {"query":{"bool": {
            which_bool: {
            "multi_match": {
            "query": phrase,
            "fields": ["text"],
            "type": "best_fields",
            "boost": 2
            }}, "minimum_should_match": minimum_should_match
            }},
            "size": 10}

def get_documents_containing_phrases(
    phrases: List[str],
    index: str,
    batch_size: int = 100,
    # slop: int = 3,
    # return_all_hits: bool = False,
    # sort_field: str = "max_score",
    # subset_filter: Optional[Dict[str, Any]] = None,
    # es: Optional[Elasticsearch] = None,
):
    es = es_init()

    queries = list(_query_documents_contain_phrases(phrases=phrases, index_name=index))
    # Split into batches
    batched_queries = [
        queries[i:i + batch_size*2]  # *2 because each query has header and body
        for i in range(0, len(queries), batch_size*2)
    ]

    all_results = []
    for batch in batched_queries:
        try:
            responses = es.msearch(body=batch)
            for response in responses["responses"]:
                all_results.append(response["hits"]["hits"])
        except Exception as e:
            logger.error(f"Batch search failed: {str(e)}")
    
    return all_results

def parallel_batch_search(
        phrases: List[str],
        index: str,
        batch_size: int = 100,
        max_workers=None):
    """
    Perform batch search using multiprocessing
    
    Args:
        query_terms: List of search terms or query dictionaries
        batch_size: Number of queries per batch
        max_workers: Maximum number of parallel workers (default: CPU count)
        
    Returns:
        List of search results
    """
    # es = es_init()

    if max_workers is None:
        max_workers = cpu_count()
        
    queries = list(_query_documents_contain_phrases(phrases=phrases, index_name=index))
    
    # Split into batches for parallel processing
    batched_queries = [
        queries[i:i + batch_size*2]
        for i in range(0, len(queries), batch_size*2)
    ]
    
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create ordered mapping of batch index to future
        future_map = OrderedDict()
        for i, batch in enumerate(batched_queries):
            future_map[i] = executor.submit(es.msearch, body=batch)
                
        # Process in original order as futures complete
    all_results = [None] * len(batched_queries)  # Pre-allocate result list
    
    for i, future in future_map.items():
        try:
            response = future.result()
            all_results[i] = response["responses"]  # Store in original position
        except Exception as e:
            logger.error(f"Batch {i} failed: {str(e)}")
            all_results[i] = [{"error": str(e)}]
    
    # Flatten results if needed
    
    final_results = [hits["hits"]["hits"] for batch in all_results for hits in batch]

    return final_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Index documents stored in jsonl.gz files"
    )
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--num_documents", type=int, required=False, default=10)
    parser.add_argument("--dataname", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--categories", type=str, default=None, required=False)
    parser.add_argument("--bs", type=int, required=False)


    args = parser.parse_args()
    es = es_init()

    
    num_workers = 4  # Use available CPU cores
    
    if args.categories is not None:
        categories = [args.categories]
    else:
        categories = [ "qa", "pc"]
    
    hf_path = args.dataname + '_' + args.model + '_test'
    if os.path.isdir(hf_path):
        data = datasets.load_from_disk(hf_path)
        data = data.to_pandas()
    else:
        test_save_path = "responses/" + args.dataname + '_' + args.model + '_test.csv'
        data = pd.read_csv(test_save_path, encoding='utf-8', index_col=False)
    
    func = partial(get_documents_containing_phrases, 
                    index=args.index, 
                    all_phrases=False,
                    num_documents = args.num_documents,
                    is_regexp = False)
    
    count_empty_returns = {c: 0 for c in categories}
    for c in categories:
        print(c)
        if c+'_query' not in data.columns.tolist():

            if c in ['question', 'response']:
                phrases = data[c].tolist()   

            elif c == 'qa':
                phrases = [" ".join([i,str(j)]) for i, j in zip(data['question'].tolist(), data['response'].tolist()) ]

            elif c == 'pc':
                phrases = [" ".join([confi_prompt(), str(i)]) for i in data['prob_confidence'].tolist()]  
                
        
            start_time = time.time()
            values = []
            
            # print(cpu_count())
            values = parallel_batch_search(
                phrases=phrases,
                index=args.index,
                batch_size=args.bs,
                max_workers=num_workers  # Adjust based on your cluster capacity
            )
                            
            if len(values) == 0:
                values = ['' for i in range(len(data))]

            
            for i in values:
                if len(i) == 0:
                    count_empty_returns[c] += 1


            data[c+'_query'] = values
            print('1k sample need:', time.time()-start_time)
            print(count_empty_returns)
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
            data.to_csv('retrived_docs/' + args.dataname + '_' + args.model + '_test.csv', encoding ='utf-8', index=False)
            try:
                save_data = datasets.Dataset.from_pandas(data)
                save_data.save_to_disk(args.dataname + '_' + args.model + '_test')
            except:
                pass
  
        
   
