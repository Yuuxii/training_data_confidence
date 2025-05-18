
from functools import cache

from typing import Any, Dict, Generator, Iterable, List, Optional, Union
import argparse
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
import requests
import pandas as pd
from load_data_prompt import confi_prompt
from datasets import Dataset


from huggingface_hub import login

from huggingface_hub import HfApi

login(token=" ")

from load_data_prompt import dataname_list

@cache
def es_init() -> Elasticsearch:
    """
    :param config: Path to the config yaml file, containing `cloud_id` and `api_key` fields.
    :return: Authenticated ElasticSearch client.
    """
    ELASTIC_PASSWORD=""
    es = Elasticsearch('https://localhost:9200', basic_auth=("elastic", ELASTIC_PASSWORD), verify_certs=False)
    requests.packages.urllib3.disable_warnings() 

    return es

def _query_documents_contain_phrases(
    phrases: Union[str, List[str]],
    all_phrases: bool = False,
    do_score: bool = False,
    is_regexp: bool = False,
    slop: int = 1,
) -> Dict:
    if isinstance(phrases, str):
        phrases = [phrases]
    if all_phrases:
        which_bool = "must" if do_score else "filter"
        minimum_should_match = None
    else:
        which_bool = "should"
        minimum_should_match = 1

    if is_regexp:
        match_query = []
        for phrase in phrases:
            match_query.append(
                {
                    "regexp": {
                        "text": {
                            "value": phrase,
                            "case_insensitive": True,
                            "flags": "ALL",
                        }
                    }
                }
            )
        # minimum_should_match = None
    else:
        match_query = []
        for phrase in phrases:
            # match_query.append({"match_phrase": {"text": {"query": phrase, "slop": slop}}})
            match_query.append({ "multi_match": {
            "query": phrase,
            "fields": ["text"],
            "type": "best_fields",
            "boost": 2
          }})
    query = {
        "bool": {which_bool: match_query, "minimum_should_match": minimum_should_match}
    }
    return query



def get_documents_containing_phrases(
    index: str,
    phrases: Union[str, List[str]],
    all_phrases: bool = False,
    num_documents: int = 10,
    is_regexp: bool = False,
    slop: int = 3,
    return_all_hits: bool = False,
    sort_field: str = "max_score",
    subset_filter: Optional[Dict[str, Any]] = None,
    es: Optional[Elasticsearch] = None,
) -> Generator[Dict, None, None]:
    """
    :param index: Name of the index
    :param phrases: A single string or a list of strings to be matched in the `text` field
        of the index.
    :param all_phrases: Whether the document should contain all phrases (AND clause) or any
        of the phrases (OR clause).
    :param num_documents: The number of document hits to return.
    :param is_regexp: Whether the phrases are regular expressions. Note that spaces in regular
        expressions are not supported by ElasticSearch, so if you want to do an exact match for
        spans longer than a single term, set this to False. In most cases, using exp1|exp2 is better
        than specifying [exp1, exp2] as two different `phrases`.
    :param return_all_hits: Whether to return all hits beyond maximum 10k results. This will return an
        iterator.
    :return: An iterable (of length `num_documents` if `return_all_hits` is False),
        containing the relevant hits.

    Examples:

        get_documents_containing_phrases("test-index", "legal", num_documents=50)  # single term, get 50 documents
        get_documents_containing_phrases("test-index", ["legal", "license"])  # list of terms
        get_document_containing_phrases("test-index", ["terms of use", "legally binding"])  # list of word sequences

        # The documents should contain both `winter` and `spring` in the text.
        get_documents_containing_phrases("test-index", ["winter", "spring"], all_phrases=True)
    """
    es = es or es_init()

    query = _query_documents_contain_phrases(phrases, all_phrases, is_regexp=is_regexp, slop=slop)
    if index == "c4":
        if not subset_filter:
            subset_filter = [{"subset": "en"}]
        else:
            subset_filter.append({"subset": "en"})

    if subset_filter:
        if "filter" not in query["bool"]:
            query["bool"]["filter"] = []
        for filter_ in subset_filter:
            query["bool"]["filter"].append({"term": filter_})

    if return_all_hits:
        sort = [{sort_field: "asc"}]
        pit = es.open_point_in_time(index=index, keep_alive="1m")
        # pit_search = {"id": pit["id"], "keep_alive": "1m"}
        # all_results = []
        results = es.search(index=index, query=query, size=num_documents, sort=sort)[
            "hits"
        ]["hits"]
        yield from results
        while len(results) > 0:
            # todo: perhaps we need to refresh pit?
            results = es.search(
                index=index,
                query=query,
                size=num_documents,
                sort=sort,
                search_after=results[-1]["sort"],
            )["hits"]["hits"]
            yield from results
        try:
            es.close_point_in_time(id=pit["id"])
        except NotFoundError:
            # Already closed.
            pass
    else:
        yield from es.search(index=index, query=query, size=num_documents)["hits"][
            "hits"
        ]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Index documents stored in jsonl.gz files"
    )
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--num_documents", type=int, required=False, default=10)
    # parser.add_argument("--dataname", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)


    args = parser.parse_args()

    ### get similar documents to confidence prompt 
    wrap = get_documents_containing_phrases(
                    index=args.index, 
                    phrases=confi_prompt(), 
                    num_documents=args.num_documents
                    )
    confidence_prompt_pd = pd.DataFrame({"confidence_prompt": [confi_prompt()], "confidence_prompt_query":[[item for item in wrap]]})
    confidence_prompt_pd = confidence_prompt_pd.loc[:, ~confidence_prompt_pd.columns.str.contains('^Unnamed')]
    confidence_prompt_pd.to_csv('responses/confidence_prompt_query.csv', encoding ='utf-8')

    # for slop in [1,2,3,4,5]:
    for dataname in dataname_list:
        # for dataname in ['popqa']:
            
        ### get similar documents to different contents
        categories = ['question', 'response', 'prob_confidence', 'qa', 'pc']

        test_save_path = "responses/" + dataname + '_' + args.model + '_test.csv'
        data = pd.read_csv(test_save_path, encoding='utf-8', index_col=False)

        count_empty_returns = {c: 0 for c in categories}
        for c in categories:
            values = []
            if c in ['question', 'response', 'prob_confidence']:
                for i in data[c].tolist():       
                    wrap = get_documents_containing_phrases(
                        index=args.index, 
                        phrases=str(i), 
                        num_documents=args.num_documents,
                        # slop=slop
                        )
                
                    values.append([item for item in wrap])

            elif c == 'qa':
                for i, j in zip(data['question'].tolist(), data['response'].tolist()):       
                    wrap = get_documents_containing_phrases(
                        index=args.index, 
                        phrases=" ".join([i,str(j)]), 
                        num_documents=args.num_documents,
                        # slop=slop
                        )
                
                    values.append([item for item in wrap])
                
            elif c == 'pc':
                for i in data['prob_confidence'].tolist():       
                    wrap = get_documents_containing_phrases(
                        index=args.index, 
                        phrases=" ".join([confi_prompt(), str(i)]), 
                        num_documents=args.num_documents,
                        # slop=slop
                        )
                
                    values.append([item for item in wrap])
            
            if len(values) == 0:
                values = ['' for i in range(len(data))]

            
            for i in values:
                if len(i) == 0:
                    count_empty_returns[c] += 1


            data[c+'_query'] = values

        print(count_empty_returns)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        data = Dataset.from_pandas(data)
        data.save_to_disk(dataname + '_' + args.model + '_test')
