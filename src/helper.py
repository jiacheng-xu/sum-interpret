
from util import *


@dec_print_wrap
def get_sum_data(dataset_name='xsum', split='validation'):
    """
    Load dataset with huggingface datasets
    """
    from datasets import load_dataset
    dataset = load_dataset(dataset_name, split=split)
    logging.info(dataset.features)
    logging.info(f"First Example in {dataset_name} {split}: {dataset[0]}")
    return dataset


def get_summ_prefix(tgt_token: str, raw_output_summary: str):
    """
    Get the prefix summary given the query
    """
    lower_tgt_token = tgt_token.lower()
    start_index = raw_output_summary.lower().find(lower_tgt_token)
    if start_index >= 0:
        summary_prefix = raw_output_summary[:start_index]
        logging.info(f"Prefix: {summary_prefix}")
    else:
        summary_prefix = None
        logging.warn(
            f"No match found! {tgt_token} not in {raw_output_summary}")
    return summary_prefix
