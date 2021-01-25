# batch evaluation
from helper_run_bart import write_pkl_to_disk
import time
import datetime
from util import *
from helper_run_bart import run_full_model_slim, init_bart_family
from eval_util import *


def run_eval(model_sum, data_feeder):
    result_bag = []
    for data_stream in data_feeder:
        input_ids, attn_masks, gather_prefix, gather_tgt, gather_metas = data_stream
        output_tok, _, _, loss = run_full_model_slim(
            model_sum,
            input_ids=input_ids,
            attention_mask=attn_masks,
            decoder_input_ids=gather_prefix,
            targets=gather_tgt,
            device=device)
        loss = loss.tolist()
        result_bag.append({'loss': loss,
                           'meta': gather_metas})
    return result_bag




if __name__ == "__main__":

    debug = False
    parser = common_args()
    args = parser.parse_args()
    args = fix_args(args)
    logger.info(args)
    device = args.device

    if debug:
        args.device = 'cpu'
        args.mname_sum = 'sshleifer/distilbart-xsum-6-6'
    result_file = '/mnt/data0/jcxu/eval.txt'
    # # init BART models
    if 'sent' in args.eval_mode:
        budget_ig = [0, 1, 2, 3, 4]
    else:
        budget_ig = [0, 1, 2, 4, 8, 16]

    _, model_sum, _, tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device, no_lm=True, no_ood=True)
    logger.info("Done loading BARTs.")
    model_pkg = {'sum': model_sum, 'tok': tokenizer}

    all_files_base = os.listdir(args.dir_base)
    if args.task in ['random', 'lead']:
        viable_files = all_files_base
    else:
        all_files_eval = os.listdir(args.dir_task)
        viable_files = list(set(all_files_base) & set(all_files_eval))
    # viable_files = viable_files[:100]
    random.shuffle(viable_files)
    print(len(viable_files))
    all_result = []
    try:
        for f in viable_files:
            try:
                uid = f.split('.')[0]
                exist = check_exist_file(args.dir_eval_save, f)
                if exist:
                    logger.debug(f"{f} already exists")
                    continue
                output_base_data = load_pickle(args.dir_base, f)
                step_data, meta_data = read_meta_data(args.dir_meta, f)
                if args.task in ['int_grad', 'inp_grad', 'occ', 'int_grad_sent_sel', 'inp_grad_sent_sel', 'occ_sent_sel','attn','attn_sent_sel']:
                    task_output = load_pickle(args.dir_task, f)
                    pack_data_inp = extract_from_task_output(
                        task_output, meta_data, step_data, args, budget_ig, device)
                elif args.task in ['random', 'lead']:
                    pack_data_inp = extract_from_baseline(
                        meta_data, step_data, args, budget_ig, device)
                data_gen = batch_data(pack_data_inp, tokenizer, device)
                if random.random() < 0.1:
                    print("working ...")
                this_result = run_eval(model_pkg['sum'], data_gen)
            except RuntimeError:
                logger.warning(f"Runtime error captured in {uid}")
                continue
            # write evaluation output to disk to reuse
            write_pkl_to_disk(args.dir_eval_save, uid, this_result)
            all_result += this_result
            
    except KeyboardInterrupt:
        logger.info(f"Done {len(all_result)}")
    one_line_out = summarize_result(all_result, args)
    with open(result_file, 'a') as fd:
        ct = time.strftime('%m%d%H')
        fd.write('\n')
        fd.write(ct + ","+one_line_out)
