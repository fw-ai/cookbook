# All Rights Reserved.

import json

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from omegaconf import DictConfig
from recipes.eval.common.bq_data import load_and_parse_data_from_query
from recipes.eval.common.util import read_files


def load_data(config: DictConfig) -> Dataset:
    """
    Loads data based on the configuration settings provided.

    Args:
        config: The configuration object containing parameters for data loading.

    Returns:
        The loaded dataset.
    """
    split = config.get("split")
    path = config.get("huggingface_name", config.get("format", config.get("path")))
    kwargs = {}
    if "data_files" in config:
        kwargs["data_files"] = config["data_files"]
    if "huggingface_revision" in config:
        kwargs["revision"] = str(config["huggingface_revision"])

    if config.get("format") == "bigquery":
        query = config.get("query")
        if not query:
            # TODO(pawel): remove this branch after moving the query to the config
            dataset_id = config.get("dataset_id")
            table_id = config.get("table_id")
            dataset_names = config.get("dataset_name")
            version = config.get("version")
            query = f"""
                SELECT
                    *
                FROM `{dataset_id}.{table_id}`
                WHERE
                    version="{version}"
                    AND dataset_name IN ({",".join('"' + str(dataset_name) + '"' for dataset_name in dataset_names)})
            """
        print(f"running query: {query}")
        results = load_and_parse_data_from_query(query)
        for row in results:
            functions = row.get("functions")
            if functions:
                row["functions"] = json.dumps(functions, indent=4, ensure_ascii=False)
        dataset = Dataset.from_dict(
            {key: [row[key] for row in results] for key in results[0].keys()}
        )
    elif config.get("path"):
        dataset = load_from_disk(path)
        if isinstance(dataset, DatasetDict) and split is not None:
            dataset = dataset[split]
    elif config.get("subset"):
        dataset = load_dataset(
            path, config.subset, split=split, trust_remote_code=True, **kwargs
        )
    elif path in ["json", "jsonl"]:
        assert (
            kwargs.get("data_files", None) is not None
        ), "In JSON format, data_files should exist"
        data_files = (
            [kwargs.get("data_files")]
            if isinstance(kwargs.get("data_files"), str)
            else kwargs.get("data_files")
        )
        dataset_dict = {}
        for data_file in data_files:
            if path == "json":
                with open(data_file, "r") as f:
                    data_json = json.load(f)
            else:
                data_json = [
                    json.loads(row)
                    for row in list(read_files(data_file).values())[0].split("\n")
                    if len(row) > 0
                ]

            dataset_dict[data_file] = Dataset.from_list(
                [
                    {k: json.dumps(v, ensure_ascii=False) for k, v in row.items()}
                    for row in data_json
                ]
            )
        dataset = concatenate_datasets(dataset_dict.values())
    else:
        dataset = load_dataset(path, split=split, trust_remote_code=True, **kwargs)

    print(f"loaded dataset {dataset}")

    if isinstance(dataset, DatasetDict):
        dataset = concatenate_datasets(dataset.values())

    return dataset
