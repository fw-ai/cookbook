import logging
from typing import Any, Dict, List

from google.cloud import bigquery


def load_and_parse_data_from_query(
    query: str, project_id: str = "nomadic-bison-363821"
) -> List[Dict[str, Any]]:
    """
    This function takes a SQL query as input, executes this query against Google BigQuery, and
    waits for the result.

    Args:
        query: The SQL query string to be executed. This query should follow the syntax and
            conventions of Google BigQuery SQL.

    Returns:
        A list of dictionaries, where each dictionary represents a row from
        the query result. The keys of the dictionary correspond to the column names of the result
        set.
    """
    client = bigquery.Client(project=project_id)
    query_job = client.query(query)
    query_job.result()
    if query_job.errors and len(query_job.errors):
        logging.error(f"{query=} failed with errors {query_job.errors}")

    results = [dict(row) for row in query_job]

    return results


def update_data(query: str, project_id: str = "nomadic-bison-363821") -> None:
    """
    Executes a given BigQuery SQL query.

    Args:
        query: The SQL query string to be executed.
        project_id: The Google Cloud project ID.
    """
    client = bigquery.Client(project=project_id)
    query_job = client.query(query)
    query_job.result()  # Wait for the job to complete

    if query_job.errors:
        logging.error(f"{query=} failed with errors: {query_job.errors}")
        raise RuntimeError(f"{query=} failed with errors: {query_job.errors}")
