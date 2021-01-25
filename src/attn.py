from helper_run_bart import write_pkl_to_disk
from transformers.modeling_outputs import BaseModelOutput
from helper_run_bart import init_bart_sum_model, init_bart_family
from util import *
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
import torch
from captum.attr._utils.visualization import format_word_importances
from helper import *

from helper_run_bart import run_full_model_slim


if __name__ == "__main__":

    parser = common_args()

    args = parser.parse_args()
    args = fix_args(args)
    logger.info(args)

    device = args.device

    model_lm, model_sum, model_sum_ood, tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device, no_lm=True, no_ood=True)
    logger.info("Done loading BARTs.")
    model_pkg = {'sum': model_sum, 'tok': tokenizer}
    all_files = os.listdir(args.dir_base)

    for f in all_files:
        outputs = []
        exist = check_exist_file(args.dir_task, f)
        if exist:
            logger.debug(f"{f} already exists")
            continue
        output_base_data = load_pickle(args.dir_base, f)
        step_data, meta_data = read_meta_data(args.dir_meta, f)
        sent_token_ids = meta_data['sent_token_ids']
        uid = meta_data['id']
        print(f"Input size: {len(meta_data['doc_token_ids'])}")
        acc_durations = 0
        for t, step in enumerate(step_data):
            output_base_step = output_base_data[t]
            if args.sent_pre_sel:
                input_doc = prepare_filtered_input_document(
                    output_base_step, sent_token_ids)
            else:
                input_doc = meta_data['doc_token_ids'][:args.hard_max_len]
            start_time = time.time()
            prefix = step['prefix_token_ids']
            input_tensor = torch.LongTensor([input_doc])
            result = run_full_model_slim(model=model_pkg['sum'], input_ids=input_tensor, output_attentions=True,decoder_input_ids=prefix, special_attn=True, device=device)

            acc_durations += (time.time() - start_time)
            result = result.cpu().detach()
            if args.sent_pre_sel:
                rt_step = {
                    'doc_token_ids': input_doc,
                    'output': result
                }
                outputs.append(rt_step)
            else:
                outputs.append(result)
            # outputs.append(ig_enc_result)
        skinny_meta = {
            'doc_token_ids': input_doc,
            'map_index': meta_data['map_index'],
            'sent_token_ids': meta_data['sent_token_ids'],
            'output': outputs,
            'time': acc_durations,
        }
        write_pkl_to_disk(args.dir_task, uid, skinny_meta)
        print(f"Done {uid}.pkl")
