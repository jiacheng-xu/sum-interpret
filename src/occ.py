from torch.utils.data.dataloader import default_collate
from helper_run_bart import run_full_model_slim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from helper import *
from helper_run_bart import write_pkl_to_disk
from transformers.modeling_outputs import BaseModelOutput
from helper_run_bart import init_bart_sum_model, init_bart_family
from util import *
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
import torch
from captum.attr._utils.visualization import format_word_importances

from int_grad import forward_enc_dec_step, bart_decoder_forward_embed, summarize_attributions, simple_viz_attribution


def step_input_grad(input_ids, actual_word_id, prefix_token_ids, model_pkg, device):
    input_ids = torch.LongTensor(input_ids).to(device).unsqueeze(0)
    # input_ids = input_ids[:, :400]
    batch_size, seq_len = input_ids.size()
    assert batch_size == 1
    # encode enc input
    model_encoder = model_pkg['sum'].model.encoder

    # encode dec input
    decoder_input_ids = prefix_token_ids.to(device)
    # decoder_input_ids = prefix_token_ids.repeat((num_run_cut, 1))

    dec_seq_len = decoder_input_ids.size()[-1]
    model_decoder = model_pkg['sum'].model.decoder
    embed_scale = model_decoder.embed_scale
    embed_tokens = model_decoder.embed_tokens
    # dec input embedding
    dec_inp_embedding = bart_decoder_forward_embed(
        decoder_input_ids, embed_tokens, embed_scale)

    with torch.enable_grad():
        encoder_outputs = model_encoder(input_ids.to(device), return_dict=True)
        encoder_outputs.last_hidden_state.retain_grad()

        interp_out = forward_enc_dec_step(
            model_pkg['sum'], encoder_outputs=encoder_outputs, decoder_inputs_embeds=dec_inp_embedding)

        logits = interp_out.logits[:, -1, :]
        target = torch.LongTensor([actual_word_id]).to(device)

        loss = torch.nn.functional.cross_entropy(logits, target)

        loss.backward()
        logger.info(f"Loss: {loss.tolist()}")
        raw_grad = encoder_outputs.last_hidden_state.grad
        # print(raw_grad.size())  # 1, 563, 1024
        # print(encoder_outputs.last_hidden_state.size())
        result_inp_grad = raw_grad * encoder_outputs.last_hidden_state
        # result_inp_grad = torch.mean(result_inp_grad, dim=-1)

    ig_enc_result = summarize_attributions(result_inp_grad)
    # ig_dec_result = summarize_attributions(ig_dec_result)
    if random.random() < 0.01:
        extracted_attribution = ig_enc_result.squeeze(0)
        input_doc = input_ids.squeeze(0)
        viz = simple_viz_attribution(
            tokenizer, input_doc, extracted_attribution)
        logger.info(viz)
        # extracted_attribution = ig_dec_result
        # viz = simple_viz_attribution(
        #     tokenizer, decoder_input_ids[0], extracted_attribution)
        # logger.info(viz)
    return ig_enc_result


def iterate_occlusions(input_tok_ids: List[int], context_window=2, mask_token_id=1):
    l = len(input_tok_ids)
    min_op, max_op = 1, l - 1
    full_occ = [input_tok_ids.copy() for _ in range(l)]    # make a dup copy
    attn_mask = [[1]*l for _ in range(l)]
    for idx, tok in enumerate(input_tok_ids):
        left, right = max(
            min_op,  idx-context_window), min(idx + context_window, max_op)
        for jdx in range(left, right):
            full_occ[idx][jdx] = mask_token_id
            attn_mask[idx][jdx] = 0
    return full_occ, attn_mask


class MyDataset(Dataset):
    def __init__(self, raw_data, attn):
        self.data = raw_data
        self.attn = attn

    def __getitem__(self, index):
        return (self.data[index], self.attn[index])

    def __len__(self):
        return len(self.data)


def my_collate(batch, max_len=500):
    batch = [(b[0][:max_len], b[1][:max_len]) for b in batch]
    inps = [b[0] for b in batch]
    attns = [b[1] for b in batch]
    inps = torch.LongTensor(inps)
    attns = torch.LongTensor(attns)
    return (inps, attns)


def step_occlusion(input_tok_ids, actual_word_id, prefix_token_ids, batch_size, model_pkg, device):

    combination_input_tok_ids, combination_attn_mask = iterate_occlusions(
        input_tok_ids)

    total_len = len(combination_input_tok_ids)

    # combination_input_tok_ids = torch.LongTensor(combination_input_tok_ids)
    # combination_attn_mask = torch.LongTensor(combination_attn_mask)

    dataset = MyDataset(combination_input_tok_ids, combination_attn_mask)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=my_collate)
    rec_probs = []
    for batch_idx, inp_data in enumerate(data_loader):
        input_ids, inp_attn = inp_data
        input_ids = input_ids.to(device)
        inp_attn = inp_attn.to(device)
        current_batch_size = input_ids.size()[0]
        decoder_input_ids = prefix_token_ids.to(device).repeat((current_batch_size, 1))
        gather_tgt = torch.LongTensor(
            [actual_word_id]*current_batch_size)
        output_tok, prob, _, loss = run_full_model_slim(
            model_sum,
            input_ids=input_ids,
            attention_mask=inp_attn,
            decoder_input_ids=decoder_input_ids,
            targets=gather_tgt,
            device=device)
        sel_prob = prob[:, actual_word_id].tolist()
        rec_probs += sel_prob
    assert len(rec_probs) == total_len
    return rec_probs


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
        step_data, meta_data = read_meta_data(args.dir_meta, f)
        exist = check_exist_file(args.dir_task, f)
        if exist:
            logger.debug(f"{f} already exists")
            continue
        uid = meta_data['id']
        for step in step_data:

            result = step_occlusion(input_tok_ids=meta_data['doc_token_ids'], actual_word_id=step['tgt_token_id'],
                                    prefix_token_ids=step['prefix_token_ids'], batch_size=args.batch_size, model_pkg=model_pkg, device=device)
            print(f"Mean:{statistics.mean(result) } Max: {max(result) }")
            outputs.append(result)
        skinny_meta = {
            'doc_token_ids': meta_data['doc_token_ids'],
            'map_index': meta_data['map_index'],
            'sent_token_ids': meta_data['sent_token_ids'],
            'output': outputs
        }
        write_pkl_to_disk(args.dir_task, uid, skinny_meta)
        print(f"Done {uid}.pkl")
