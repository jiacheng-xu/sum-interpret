from typing import List


def prepare_concat_input_seq(sel_budget: int, input_doc: List[int], top_indicies: List[int], ctx_window=2, sep_token_id=6) -> List[int]:
    # .=4  ,=6  space=1437
    selected_tokens = []
    cursor = 0
    max_doc_len = len(input_doc) - 1  # <eos>
    min_doc_len = 1  # <s>
    while sel_budget > 0:
        sel_idx = top_indicies[cursor]
        cursor += 1
        left, right = max(
            min_doc_len, sel_idx-ctx_window), min(sel_idx + ctx_window, max_doc_len)
        if selected_tokens:
            span_bpe = [sep_token_id]+input_doc[left:right]
        else:
            span_bpe = input_doc[left:right]
        selected_tokens += span_bpe
        sel_budget -= 1
    return selected_tokens

def yield_random_rank(input_doc):
    pass