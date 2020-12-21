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
    for t, step_data in enumerate(data):
        p_lm = step_data['p_lm']
        p_imp = step_data['p_imp']
        p_attn = step_data['p_attn']
        p_full = step_data['p_full']
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
        show_top_k(p_lm, prefix, 'lm')
        show_top_k(p_imp, prefix, 'imp')
        show_top_k(p_full, prefix, 'full')
        show_top_k(p_attn, prefix, 'attn')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", help="device to use", default='cuda:1')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')

    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/interpret_output",
                        help="The location to save output data. ")
    args = parser.parse_args()
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    all_files = os.listdir(args.dir_save)
    for f in all_files:
        analyze_one_p_file(args.dir_save, f)
        break
    """
    nprocess = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=nprocess) as pool:
        results = pool.starmap(analyze_one_p_file, [
                               (args.dir_save, f) for f in all_files])
    """
