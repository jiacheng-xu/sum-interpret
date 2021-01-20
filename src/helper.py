
import re
from util import *


@dec_print_wrap
def get_sum_data(dataset_name='xsum', split='validation'):
    """
    Load dataset with huggingface datasets
    """
    if dataset_name == 'xsum':
        from datasets import load_dataset
        dataset = load_dataset(dataset_name, split=split)
        logger.info(dataset.features)
        # logger.debug(f"First Example in {dataset_name} {split}: {dataset[0]}")
    elif dataset_name == 'cnndm':
        with open('/mnt/data0/jcxu/dataset_cnndm/validation.pkl', 'rb') as fd:

            dataset = pickle.load(fd)
    return dataset


WORD = re.compile(r'\w+')


@dec_print_wrap
def show_top_k(prob, prefix, name, tokenizer, k=5):
    prob = prob.squeeze()
    topk_v, top_idx = torch.topk(prob, k=k)
    index = top_idx.tolist()
    toks = [tokenizer.convert_ids_to_tokens(i) for i in index]

    logger.info(f"Type: {name}")
    result = []
    for i, t in enumerate(toks):
        logger.info(f"{i}: {pnum(topk_v[i].item())} {prefix}{t}")
        result.append((pnum(topk_v[i].item()), t))
    return result


def reg_tokenize(text):
    words = WORD.findall(text)
    return words


def get_summ_prefix(tgt_token: str, raw_output_summary: str, start_matching_index: int) -> str:
    """
    Get the prefix summary given the query
    """
    lower_tgt_token = tgt_token.lower()
    start_index = raw_output_summary.lower().find(
        lower_tgt_token, start_matching_index)
    if start_index >= 0:
        summary_prefix = raw_output_summary[:start_index]
        logger.debug(f"Prefix: {summary_prefix}")
    else:
        summary_prefix = ""
        logger.warn(
            f"No match found! {tgt_token} not in {raw_output_summary}")
    return summary_prefix


def init_vocab_distb_fix(tokenizer) -> torch.Tensor:
    trans_mat = np.eye(tokenizer.vocab_size-1, dtype=np.float)
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


def fix_distribution(prob_distb, remapping_mat, truncate_vocab_size=50264, device='cuda:1', neu_samp=True):
    prob_distb = prob_distb.squeeze().to(device)

    prob_distb = prob_distb[..., :truncate_vocab_size]

    mapped_prob = torch.matmul(prob_distb, remapping_mat)
    if neu_samp:
        mapped_prob -= 1e-4
        mapped_prob = torch.nn.functional.relu(mapped_prob)
    else:
        mapped_prob += + 1e-8
    mapped_prob = mapped_prob/torch.sum(mapped_prob, dim=-1, keepdim=True)
    return mapped_prob


def compute_group_jaccard(distb, distb_signature, k=1):
    # distribution = [x.unsqueeze(0) for x in distb]
    topk_indicies = [torch.topk(x, k=k) for x in distb]
    l = len(distb)
    # create source batch
    src = []
    src_sig = []
    for idx in range(l):
        src += [topk_indicies[idx]] * (l-1)
        src_sig += [distb_signature[idx]] * (l-1)
    tgt = []
    tgt_sig = []
    for idx in range(l):
        for jdx in range(l):
            if idx == jdx:
                continue
            tgt += [topk_indicies[jdx]]
            tgt_sig += [distb_signature[jdx]]
    assert len(src) == len(tgt)
    distance = []
    for idx in range(len(src)):
        jac = len(src[idx] & tgt[idx]) / len(src[idx] | tgt[idx])
        # print(jac)
        distance.append(jac)
    # klv = torch.sum(torch.abs(src - tgt), dim=-1).cpu().tolist()

    # klv = batch_kl_value.mean(dim=-1).cpu().tolist()

    name = [f"{x}_{y}" for (x, y) in zip(src_sig, tgt_sig)]
    # Create a zip object from two lists
    zipbObj = zip(name, distance)
    # Create a dictionary from zip object
    dictOfWords = dict(zipbObj)
    return dictOfWords


def compute_group_deduct(distb, distb_signature) -> Dict:
    log_distributions = [x.unsqueeze(0) for x in distb]
    l = len(log_distributions)
    # create source batch
    src = []
    src_sig = []
    for idx in range(l):
        src += [log_distributions[idx]] * (l-1)
        src_sig += [distb_signature[idx]] * (l-1)
    tgt = []
    tgt_sig = []
    for idx in range(l):
        for jdx in range(l):
            if idx == jdx:
                continue
            tgt += [log_distributions[jdx]]
            tgt_sig += [distb_signature[jdx]]
    src = torch.cat(src)
    tgt = torch.cat(tgt)

    klv = torch.sum(torch.abs(src - tgt), dim=-1).cpu().tolist()

    # klv = batch_kl_value.mean(dim=-1).cpu().tolist()

    name = [f"{x}_{y}" for (x, y) in zip(src_sig, tgt_sig)]
    # Create a zip object from two lists
    zipbObj = zip(name, klv)
    # Create a dictionary from zip object
    dictOfWords = dict(zipbObj)
    return dictOfWords


def feat_perturb(p_pert, p_full, distb_fix, device):
    # p_pert batch, vocab
    # p_full vocab

    value, indices = torch.topk(p_full, k=1, dim=-1)
    indi = indices.tolist()[0]
    p_pert = fix_distribution(p_pert, distb_fix, device=device)
    sel_values = p_pert[:, indi].tolist()
    # print(sel_values)
    assert len(sel_values) >= 1
    top1 = max(sel_values)

    max_value = max(sel_values)

    var = statistics.mean([abs(this_v - max_value)
                            for this_v in sel_values])
    return top1, var, sel_values


def check_exist_file(dir_to_search, fname):
    files = os.listdir(dir_to_search)
    files = [f for f in files if f.startswith(fname)]
    if any(files):
        return True
    else:
        return False
