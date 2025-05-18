import argparse
import gzip
import json
import logging
from functools import cache
import h5py
import numpy as np
from elasticsearch.helpers import parallel_bulk
from tqdm import tqdm
import yaml
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
import os
import requests

@cache
def es_init() -> Elasticsearch:
    """
    :param config: Path to the config yaml file, containing `cloud_id` and `api_key` fields.
    :return: Authenticated ElasticSearch client.
    """
    ELASTIC_PASSWORD="43tEcVyvw3*kgudJekNf"
    es = Elasticsearch('https://localhost:9200', basic_auth=("elastic", ELASTIC_PASSWORD), verify_certs=False, timeout=60)
    requests.packages.urllib3.disable_warnings() 

    return es


def get_indices(
    return_mapping: bool = False, es: Optional[Elasticsearch] = None
) -> Dict:
    """
    :param return_mapping: Whether to return mapping along with index information.
    :return: Dictionary of existing indices.

    Note that this function won't work in case you were given an API key from AI2 since these are restrictive.
    Please refer to the README for the indices mapping.
    """
    es = es or es_init()

    indices = es.cat.indices(format="json")
    exclude = [
        "search-test",
        "test-index-2",
        "metrics-endpoint.metadata_current_default",
    ]
    indices = {
        index["index"]: {key: index[key] for key in ["docs.count"]}
        for index in indices
        if not index["index"].startswith(".") and not index["index"] in exclude
    }

    if return_mapping:
        mappings = es.indices.get_mapping(index=list(indices.keys()))
        for key in mappings:
            indices[key]["properties"] = list(
                mappings[key]["mappings"]["properties"].keys()
            )

    return indices



def main():
    parser = argparse.ArgumentParser(
        description="Index documents stored in jsonl.gz files"
    )
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--id-field", type=str, required=False, default=None)
    parser.add_argument("--skip", type=int, required=False, default=0)
    parser.add_argument("--bulk-docs", type=int, required=False, default=500)
    parser.add_argument(
        "--max-doc-size", type=int, required=False, default=1024 * 1024 * 99
    )
    parser.add_argument("--text-field", type=str, required=False, default="text")
    parser.add_argument(
        "--skip-fields",
        type=str,
        required=False,
        default="",
        help="comma-separated list of fields to NOT include in the index",
    )
    parser.add_argument("--save-ids-file", type=str, required=False, default=None)
    parser.add_argument("--filenames-path", type=str, required=False)
    parser.add_argument("--num-shards", type=int, required=False, default=1)


    args = parser.parse_args()

    if args.filenames_path:
        # with open(args.filenames_path) as f:
        #     filenames = f.readlines()
        #     filenames = [path.strip() for path in filenames]
        filenames = os.listdir(args.filenames_path)
        filenames = [args.filenames_path + name for name in filenames if '.config' not in name]
    else:
        filenames = args.filename

    with open("es_index.config", "w+") as f:
        json.dump(args.__dict__, f, indent=4)

    es = es_init()
    index = args.index.lower()

    skip_fields = args.skip_fields.split(",")
    skip_fields = [s for s in skip_fields if s != ""]

    def make_action(filename, line_number, line):
        doc = json.loads(line)
        doc["text"] = doc.pop(args.text_field)[: args.max_doc_size]
        for field in skip_fields:
            doc.pop(field, None)
        if args.id_field is None:
            doc_id = f"{filename}-{line_number}"
        else:
            doc_id = doc.pop(args.id_field)
        return {"_source": doc, "_index": index, "_id": doc_id, "_op_type": "create"}

    assert (
        len(filenames) == 1 or args.skip == 0
    ), "You can't skip when you specify more than one file."

    if args.save_ids_file:
        with h5py.File(args.save_ids_file, "w") as f:
            dset = f.create_dataset("ids", shape=(0,), maxshape=(None,), dtype="S40")
    print(get_indices(es=es), index)
    # es.indices.delete(index="rlvr-7b")
    # es.indices.delete(index="rlvr-13b")
    # es.indices.delete(index="rlvr-8b")
    try:
        es.indices.delete(index=index)
    except:
        pass
    # es.indices.delete(index="olmo13b-preference")
    # es.indices.delete(index="llama-tulu3")
    # es.indices.delete(index="olmo-tulu3")

    if index not in get_indices(es=es):
        print(
            f"The index '{index}' is being created with {args.num_shards} shards"
        )
        es.indices.create(
            index=index, settings={"index.number_of_shards": args.num_shards}
        )
    else:
        print(f"'{index}' already exists. Indexing more documents into it...")

    for filename in filenames:
        with gzip.open(filename, "rt", encoding="UTF8") as f:
            actions = (
                make_action(filename, line_number, line)
                for line_number, line in enumerate(f)
            )
            actions = (
                action
                for action in actions
                if action["_source"]["text"] is not None
                and len(action["_source"]["text"]) > 0
            )
            if args.skip > 0:
                import itertools

                actions = itertools.islice(actions, args.skip, None)
            results = parallel_bulk(
                es,
                actions,
                ignore_status={409},
                # max_retries=10,
                thread_count=32,
                raise_on_error=False,
                chunk_size=args.bulk_docs,
            )
            result_counts = {True: 0, False: 0}
            results_tqdm = tqdm(results, desc=f"Processing {filename}")
            result_ids = []
            for result in results_tqdm:
                status = result[1]["create"]["status"]
                result_ids.append(result[1]["create"]["_id"])
                assert status in {201, 409}, repr(result)
                result_counts[result[0]] += 1
                results_tqdm.set_postfix(
                    {str(k): v for k, v in result_counts.items()}, refresh=False
                )

            if args.save_ids_file:
                with h5py.File(args.save_ids_file, "a") as f:
                    dset = f["ids"]
                    new_size = dset.shape[0] + len(result_ids)
                    dset.resize((new_size,))
                    dset[-len(result_ids) :] = np.array(result_ids, dtype="S40")


if __name__ == "__main__":
    main()