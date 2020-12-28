# analyze the output of the ensemble
from itertools import combinations
from scipy.stats import wasserstein_distance
from util import *
from helper import init_vocab_distb_fix
distb_fix = init_vocab_distb_fix(tokenizer).float()
device = 'cuda:2'
distb_fix = distb_fix.to(device)


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
    for i, t in enumerate(toks):
        logger.info(f"{i}: {pnum(topk_v[i].item())} {prefix}{t}")


def fix_distribution(prob_distb, remapping_mat, truncate_vocab_size=50264, device='cuda:2', neu_samp=True):
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

    name = [f"{x}2{y}" for (x, y) in zip(src_sig, tgt_sig)]
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
        pos = step_data['pos'][t]
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
        max_vocab_size = 50264
        p_lm = fix_distribution(p_lm, distb_fix)
        p_imp = fix_distribution(p_imp, distb_fix)
        p_full = fix_distribution(p_full, distb_fix)

        p_imp_ood = fix_distribution(p_imp_ood, distb_fix)
        p_full_ood = fix_distribution(p_full_ood, distb_fix)
        p_attn = fix_distribution(p_attn, distb_fix)

        signature = ['lm', 'imp', 'full', 'imp_cnn', 'full_cnn', 'attn']

        distributions = [p_lm, p_imp, p_full, p_imp_ood, p_full_ood, p_attn]
        # result = compute_group_kl(distributions, signature)
        result = compute_group_deduct(distributions, signature)
        # result = comp_wasserstein(distributions, signature)

        result['pos'] = pos
        result['tok'] = token
        result['t'] = t
        result['T'] = T
        result['prefix'] = prefix
        k = list(result.keys())
        v = list(result.values())
        values.append(v)
        show_top_k(p_lm, prefix, 'lm', tokenizer)
        show_top_k(p_imp, prefix, 'imp', tokenizer)
        show_top_k(p_full, prefix, 'full', tokenizer)
        show_top_k(p_imp_ood, prefix, 'imp_ood', tokenizer=tokenizer)
        show_top_k(p_full_ood, prefix, 'full_ood', tokenizer=tokenizer)
        show_top_k(p_attn, prefix, 'attn', tokenizer)
    return k, values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", help="device to use", default='cuda:1')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')
    parser.add_argument("-method",default='fast',help='distance metric to use')
    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/interpret_output_fix_lm",
                        help="The location to save output data. ")
    args = parser.parse_args()
    print(args)
    logger.info(args)
    all_files = os.listdir(args.dir_save)
    random.shuffle(all_files)
    all_files = all_files[:100]

    all_outs = []
    k = None
    for f in all_files:
        key, value = analyze_one_p_file(args.dir_save, f)
        if not k:
            k = key
        all_outs += value

    # nprocess = multiprocessing.cpu_count()
    # with multiprocessing.Pool(processes=nprocess) as pool:
    #     results = pool.starmap(analyze_one_p_file, [
    #                            (args.dir_save, f) for f in all_files])

    # print(results)

    df = pd.DataFrame(all_outs, columns=k)
    df.to_csv("output_fast.csv")
    print(f"write to {os.getcwd()}/output_fast.csv")
