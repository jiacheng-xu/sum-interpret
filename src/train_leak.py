from analyze_ensemble import load_pickle, pnum, show_top_k
from util import *

kld = torch.nn.KLDivLoss(log_target=True)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

from cache_ref_sum import retrieve_train
def analyze_one_p_file(dir, fname):
    data_pkg = load_pickle(dir, fname)
    data = data_pkg['data']
    meta = data_pkg['meta']
    # logger.info(f"{meta['id']}\nSUM: {meta['output']}")
    document, output, uid = meta['document'], meta['output'], meta['id']

    for t, step_data in enumerate(data):
        p_lm = step_data['p_lm']
        p_imp_ood = step_data['p_imp_ood']

        p_imp = step_data['p_imp']
        p_attn = step_data['p_attn']
        p_full = step_data['p_full']
        prefix = step_data['prefix']
        interest = step_data['interest']
        # logger.info(f"Prefix: {prefix} ({interest})")

        max_vocab_size = 50264
        p_lm = p_lm.squeeze()[:max_vocab_size]
        p_imp = p_imp.squeeze()[:max_vocab_size]
        p_full = p_full.squeeze()[:max_vocab_size]
        p_imp_ood = p_imp_ood.squeeze()[:max_vocab_size]
        distributions = [p_lm, p_imp, p_full, p_imp_ood]
        log_distributions = [torch.log(x) for x in distributions]

        kl_imp_to_full = kld(log_distributions[1], log_distributions[2]) * 1e5
        kl_imp_to_lm = kld(log_distributions[1], log_distributions[0]) * 1e5
        kl_imp_to_imp_ood = kld(
            log_distributions[1], log_distributions[3]) * 1e5

        if kl_imp_to_full < 0.6 and (kl_imp_to_lm > 6 and kl_imp_to_imp_ood > 6):
            logger.info(f"\n\n")
            logger.info(f"Doc:{document[:500]}\nOutput:{output}\nID:{uid}")
            logger.info(
                f"IMP->FULL: {pnum(kl_imp_to_full)}\t\tIMP->LM: {pnum(kl_imp_to_lm)}\t\tIMP->IMP_OOD: {pnum(kl_imp_to_imp_ood)}")
            logger.info(f"Potential Prefix: {prefix} | {interest}")
            show_top_k(p_lm, prefix, 'lm', tokenizer=tokenizer)
            show_top_k(p_imp, prefix, 'imp', tokenizer=tokenizer)
            show_top_k(p_imp_ood, prefix, 'imp_ood', tokenizer=tokenizer)
            show_top_k(p_full, prefix, 'full', tokenizer=tokenizer)

            retrieve_train(q=prefix.strip().split(" ")[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", help="device to use", default='cuda:1')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')

    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/interpret_output_fix_lm",
                        help="The location to save output data. ")
    args = parser.parse_args()

    all_files = os.listdir(args.dir_save)
    random.shuffle(all_files)

    for f in all_files:
        analyze_one_p_file(args.dir_save, f)
