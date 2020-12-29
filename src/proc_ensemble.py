# analyze the output of the ensemble
from retrieve_compare import map_summary_xsum, map_summary_cnndm
from itertools import combinations
from scipy.stats import wasserstein_distance
# from util import *
from helper import *
from helper import init_vocab_distb_fix
distb_fix = init_vocab_distb_fix(tokenizer).float()
device = 'cuda:1'
# device = 'cpu'
distb_fix = distb_fix.to(device)

# bigram match in train doc and test doc


def load_pickle(dir, fname) -> Dict:
    with open(os.path.join(dir, fname), 'rb') as rfd:
        data = pickle.load(rfd)
    return data


def pnum(num):
    return "{:.2f}".format(num)


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


def fix_distribution(prob_distb, remapping_mat, truncate_vocab_size=50264, device='cuda:1', neu_samp=True):
    prob_distb = prob_distb.squeeze().to(device)
    prob_distb = prob_distb[:truncate_vocab_size]

    mapped_prob = torch.matmul(prob_distb, remapping_mat)
    if neu_samp:
        mapped_prob -= 1e-2
        mapped_prob = torch.nn.functional.relu(mapped_prob)
    else:
        mapped_prob += + 1e-8
    mapped_prob = mapped_prob/torch.sum(mapped_prob)
    return mapped_prob


def compute_group_kl(distb, distb_signature) -> Dict:
    log_distributions = [torch.log(x).unsqueeze(0) for x in distb]
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
    batch_kl_value = kld(src, tgt)
    klv = batch_kl_value.mean(dim=-1).cpu().tolist()

    name = [f"{x}2{y}" for (x, y) in zip(src_sig, tgt_sig)]
    # Create a zip object from two lists
    zipbObj = zip(name, klv)
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


def comp_wasserstein(distb, distb_signature):
    distb = [x.cpu().numpy() for x in distb]
    l = len(distb)
    rt_dict = {}
    perm = combinations(range(l), 2)
    for per in perm:
        a, b = per
        name = f"{distb_signature[a]}_{distb_signature[b]}"
        name_alternative = f"{distb_signature[b]}_{distb_signature[a]}"
        distance = wasserstein_distance(distb[a], distb[b])
        distance = float(distance)
        rt_dict[name] = distance
        rt_dict[name_alternative] = distance
    return rt_dict


def feat_ngram_ref_domain(prefix: str, token: str, map_in_domain: Dict, map_ood: Dict):
    tok = token.lower()
    try:
        key_word = reg_tokenize(prefix)[-1]
    except:
        return 0
    rate_in, rate_ood = 0, 0
    if key_word in map_in_domain and tok in map_in_domain[key_word]:
        rate_in = map_in_domain[key_word][tok]
    if key_word in map_ood and tok in map_ood[key_word]:
        rate_in = map_ood[key_word][tok]
    if rate_in == 0:
        return 0
    else:
        ratio = (rate_in + 1e-2) / (rate_ood+1e-2)
        return ratio


def feat_perturb(p_pert, p_full):
    # p_pert batch, vocab
    # p_full vocab
    value, indices = torch.topk(p_full, k=1, dim=-1)
    indi = indices.tolist()[0]
    sel_values = p_pert[:, indi].tolist()
    # print(sel_values)
    assert len(sel_values) >= 1
    top1 = max(sel_values)
    try:
        var = statistics.variance(sel_values)
    except:
        var = 0
    return top1, var, sel_values


def analyze_one_p_file(dir, fname):
    data_pkg = load_pickle(dir, fname)
    data = data_pkg['data']
    T = len(data)
    meta = data_pkg['meta']
    logger.info(f"{meta['id']}\nSUM: {meta['output']}")
    document, output, uid = meta['document'], meta['output'], meta['id']
    logger.info(f"Doc:{document[:2000]}\nOutput:{output}\nID:{uid}")
    stat_output = []
    key, values = None, []
    for t, step_data in enumerate(data):
        pos = step_data['pos']
        token = step_data['token']

        p_lm = step_data['p_lm']
        p_imp = step_data['p_imp']
        p_attn = step_data['p_attn']
        p_full = step_data['p_full']

        p_imp_ood = step_data['p_imp_ood']
        p_full_ood = step_data['p_full_ood']

        prefix = step_data['prefix']
        interest = step_data['interest']
        logger.info(f"Prefix: {prefix}")

        # feature 1: prefix[-1] + output token, get the ratio of this pair in in-domain train ref (rate_in) and ood train ref (rate_out).
        # if rate_in = 0, ignore
        # if rate_in >0, compute

        domain_rate = feat_ngram_ref_domain(
            prefix, token, map_in_domain=map_summary_xsum, map_ood=map_summary_cnndm)

        max_vocab_size = 50264
        p_lm = fix_distribution(p_lm, distb_fix, device=device)
        p_imp = fix_distribution(p_imp, distb_fix, device=device)
        p_full = fix_distribution(p_full, distb_fix, device=device)

        p_imp_ood = fix_distribution(p_imp_ood, distb_fix, device=device)
        p_full_ood = fix_distribution(p_full_ood, distb_fix, device=device)
        p_attn = fix_distribution(p_attn, distb_fix, device=device)
        signature = ['lm', 'imp', 'full', 'imp_cnn', 'full_cnn', 'attn']

        distributions = [p_lm, p_imp, p_full, p_imp_ood, p_full_ood, p_attn]
        # result = compute_group_kl(distributions, signature)
        result = compute_group_deduct(distributions, signature)
        # result = comp_wasserstein(distributions, signature)

        # feature 2: perturbation
        pert_sents = step_data['pert_sents']
        p_pert = step_data['p_pert']  # num_pertb, vocab size
        pert_top1, pert_var, pert_distb = feat_perturb(p_pert, p_full)
        result['pert_top'] = pert_top1
        result['pert_var'] = pert_var
        result['pert_distb'] = pert_distb
        result['pert_sents'] = pert_sents
        top_lm = show_top_k(p_lm, prefix, 'lm', tokenizer)
        top_imp = show_top_k(p_imp, prefix, 'imp', tokenizer)
        top_full = show_top_k(p_full, prefix, 'full', tokenizer)
        top_impood = show_top_k(
            p_imp_ood, prefix, 'imp_ood', tokenizer=tokenizer)
        top_fullood = show_top_k(
            p_full_ood, prefix, 'full_ood', tokenizer=tokenizer)
        top_attn = show_top_k(p_attn, prefix, 'attn', tokenizer)
        result['top_lm'] = top_lm
        result['top_imp'] = top_imp
        result['top_full'] = top_full
        result['top_impood'] = top_impood
        result['top_fullood'] = top_fullood
        result['top_attn'] = top_attn
        result['domain_rate'] = domain_rate
        result['pos'] = pos
        result['tok'] = token
        result['t'] = t
        result['T'] = T
        result['prefix'] = prefix
        k = list(result.keys())
        v = list(result.values())
        values.append(v)

    return k, values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", help="device to use", default='cuda:1')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')
    parser.add_argument("-method", default='fast',
                        help='distance metric to use')
    parser.add_argument('-output_file', default='output_fast.csv')
    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/interpret_output",
                        help="The location to save output data. ")
    args = parser.parse_args()
    print(args)
    logger.info(args)
    all_files = os.listdir(args.dir_save)
    random.shuffle(all_files)
    all_files = all_files[:200]

    all_outs = []
    k = None
    for f in all_files:
        key, value = analyze_one_p_file(args.dir_save, f)
        if not k:
            k = key
        all_outs += value
    """
    nprocess = multiprocessing.cpu_count() - 2
    with multiprocessing.Pool(processes=nprocess) as pool:
        results = pool.starmap(analyze_one_p_file, [
                               (args.dir_save, f) for f in all_files])
    k = results[0][0]
    all_outs = [x[1] for x in results]
    """

    df = pd.DataFrame(all_outs, columns=k)
    df.to_csv(args.output_file)
    logger.info(f"write to {os.getcwd()}/{args.output_file}")
