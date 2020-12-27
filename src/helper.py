
from util import *


@dec_print_wrap
def get_sum_data(dataset_name='xsum', split='validation'):
    """
    Load dataset with huggingface datasets
    """
    from datasets import load_dataset
    dataset = load_dataset(dataset_name, split=split)
    logger.info(dataset.features)
    # logger.debug(f"First Example in {dataset_name} {split}: {dataset[0]}")
    return dataset


def get_summ_prefix(tgt_token: str, raw_output_summary: str,start_matching_index:int) -> str:
    """
    Get the prefix summary given the query
    """
    lower_tgt_token = tgt_token.lower()
    start_index = raw_output_summary.lower().find(lower_tgt_token,start_matching_index)
    if start_index >= 0:
        summary_prefix = raw_output_summary[:start_index]
        logger.info(f"Prefix: {summary_prefix}")
    else:
        summary_prefix = ""
        logger.warn(
            f"No match found! {tgt_token} not in {raw_output_summary}")
    return summary_prefix


def init_vocab_distb_fix(tokenizer) -> torch.Tensor:
    trans_mat = np.eye(tokenizer.vocab_size-1,dtype=np.float)
    cnt = 0
    for vocab_idx in range(tokenizer.vocab_size-1):
        tok = tokenizer.convert_ids_to_tokens(vocab_idx)
        if tok.startswith('Ġ'):
            no_space_tok = tok[1:]
            no_space_id = tokenizer.convert_tokens_to_ids(no_space_tok)
            if no_space_id == 3:
                continue
            logging.debug(
                f"{vocab_idx}:{tok} -> {no_space_id}:{tokenizer.convert_ids_to_tokens(no_space_id)}")
            trans_mat[vocab_idx][vocab_idx] = 0
            trans_mat[vocab_idx][no_space_id] = 1
            cnt += 1
    period_id = tokenizer.convert_tokens_to_ids(".")
    trans_mat[period_id][period_id] = 0
    logging.info(f"Lines of change: {cnt}")
    return torch.from_numpy(trans_mat)
