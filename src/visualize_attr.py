# batch evaluation
from helper_run_bart import write_pkl_to_disk
import time
import datetime
from util import *
from helper_run_bart import run_full_model_slim, init_bart_family
from eval_util import *


def extract_from_task_output(task_output, meta_data, step_data, args, budget: List[int], device):
    # 'doc_token_ids': new_input_doc,
    # 'map_index':meta_data['map_index'],
    # 'sent_token_ids':meta_data['sent_token_ids'],
    # 'output': outputs
    if 'occ' in args.task:
        task_output = process_occlusion(task_output)
    document_tokens = task_output['doc_token_ids'][:args.hard_max_len]
    map_index = task_output['map_index']
    # Truncate sent
    sent_token_ids = task_output['sent_token_ids']
    doc_tok_len = len(document_tokens)
    max_sent_idx = map_index[doc_tok_len-1]
    sent_token_ids = sent_token_ids[:max_sent_idx+1]
    map_index = map_index[:doc_tok_len]
    pred_output = task_output['output']
    eval_mode = args.eval_mode
    uid = meta_data['id']
    return_data = []
    # for token, get index of token to add or remove
    if args.sent_pre_sel:
        agg_pred_output = [t_pred['output'] for t_pred in pred_output]
    else:
        agg_pred_output = agg_time_step_task_output(pred_output, device)
        agg_pred_output = agg_pred_output.cpu().tolist()
    for t, step_pred in enumerate(agg_pred_output):
        rank_of_tok_index = argsort(step_pred)[::-1]
        if args.sent_pre_sel:
            document_tokens = pred_output[t]['doc_token_ids']
        else:
            pass
        rendered_tokens = prepare_concat_input_seq(
            bud, document_tokens, rank_of_tok_index)
        unit = assemble_units(
            rendered_tokens, step_data, t, bud, eval_mode, uid)
        return_data.append(unit)
    return return_data


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

    _, model_sum, _, tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device, no_lm=True, no_ood=True)
    logger.info("Done loading BARTs.")
    model_pkg = {'sum': model_sum, 'tok': tokenizer}

    all_files_base = os.listdir(args.dir_base)
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
                if args.task in ['int_grad', 'inp_grad', 'occ', 'int_grad_sent_sel', 'inp_grad_sent_sel', 'occ_sent_sel', 'attn', 'attn_sent_sel']:
                    task_output = load_pickle(args.dir_task, f)
                    pack_data_inp = extract_from_task_output(
                        task_output, meta_data, step_data, args, budget_ig, device)
                data_gen = batch_data(pack_data_inp, tokenizer, device)
                if random.random() < 0.1:
                    print("working ...")
                this_result = run_eval(model_pkg['sum'], data_gen)
            except RuntimeError:
                logger.warning(f"Runtime error captured in {uid}")
                continue
            # write evaluation output to disk to reuse
            all_result += this_result

    except KeyboardInterrupt:
        logger.info(f"Done {len(all_result)}")
