# analyze the output of the ensemble
from util import *


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
    toks = [tokenizer.decode(i) for i in index]
    logger.info(f"Type: {name}")
    for i, t in enumerate(toks):
        logger.info(f"{i}: {pnum(topk_v[i].item())} {prefix}{t}")


def analyze_one_p_file(dir, fname):
    data_pkg = load_pickle(dir, fname)
    data = data_pkg['data']
    meta = data_pkg['meta']
    logger.info(f"{meta['id']}\nSUM: {meta['output']}")
    document, output, uid = meta['document'], meta['output'], meta['id']
    logger.info(f"Doc:{document[:2000]}\nOutput:{output}\nID:{uid}")
    stat_output = []
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
        p_lm = p_lm.squeeze()[:max_vocab_size]
        p_imp = p_imp.squeeze()[:max_vocab_size]
        p_full = p_full.squeeze()[:max_vocab_size]
        p_attn = p_attn.squeeze()[:max_vocab_size]
        p_attn = p_attn + 1e-8
        p_attn = p_attn/torch.sum(p_attn)

        distributions = [p_lm, p_imp, p_full, p_imp_ood]
        log_distributions = [torch.log(x) for x in distributions]
        kl_full_to_lm = kld(
            log_distributions[2], log_distributions[0]).item() * 1e5
        kl_full_to_imp = kld(
            log_distributions[2], log_distributions[1]).item() * 1e5
        kl_imp_to_lm = kld(
            log_distributions[1], log_distributions[0]).item() * 1e5
        stat_output.append(
            [pos, token, kl_full_to_lm, kl_full_to_imp, kl_imp_to_lm])
        show_top_k(p_lm, prefix, 'lm', tokenizer)
        show_top_k(p_imp, prefix, 'imp', tokenizer)
        show_top_k(p_full, prefix, 'full', tokenizer)

        show_top_k(p_imp_ood, prefix, 'imp_ood', tokenizer=tokenizer)
        show_top_k(p_full_ood, prefix, 'full_ood', tokenizer=tokenizer)

        show_top_k(p_attn, prefix, 'attn', tokenizer)
    return stat_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", help="device to use", default='cuda:1')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')

    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/interpret_output_fix_lm",
                        help="The location to save output data. ")
    args = parser.parse_args()

    all_files = os.listdir(args.dir_save)
    random.shuffle(all_files)
    all_files = all_files[:50]
    all_outs = []
    for f in all_files:
        cur_output = analyze_one_p_file(args.dir_save, f)
        all_outs += cur_output
    pt = []
    d = {}
    for x in all_outs:
        if x[0] in d:
            d[x[0]] = d[x[0]] + [x[2]]
        else:
            d[x[0]] = [x[2]]

    for k, v in d.items():
        print(f"{k}\t{statistics.mean(v)}")

    """
    nprocess = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=nprocess) as pool:
        results = pool.starmap(analyze_one_p_file, [
                               (args.dir_save, f) for f in all_files])
    """
