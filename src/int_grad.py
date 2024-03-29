from helper_run_bart import write_pkl_to_disk
from transformers.modeling_outputs import BaseModelOutput
from helper_run_bart import init_bart_sum_model, init_bart_family
from util import *
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
import torch
from captum.attr._utils.visualization import format_word_importances
from helper import *


def simple_viz_attribution(tokenizer, input_ids, attribution_scores):
    token_in_list = input_ids.tolist()
    if isinstance(token_in_list[0], list):
        token_in_list = token_in_list[0]
    words = [tokenizer.decode(x) for x in token_in_list]
    attribution_scores_list = attribution_scores.tolist()
    # for w, ascore in zip(words, attribution_scores_list):
    #     logging.info('{:10} {:02.2f}'.format(w, ascore))

    output = format_word_importances(words, attribution_scores_list)
    return output


def interpolate_vectors(ref_vec, inp_vec, num_steps, device):
    interp_step_vec = (inp_vec - ref_vec)/num_steps
    ranges = torch.arange(
        1, num_steps + 1).unsqueeze(-1).unsqueeze(-1).to(device)
    rows = ranges * interp_step_vec
    interp_final = rows + ref_vec
    return interp_final


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1)
    attributions = attributions / torch.norm(attributions)
    return attributions


def forward_enc_dec_step(model, encoder_outputs, decoder_inputs_embeds):
    # expanded_batch_idxs = (
    #         torch.arange(batch_size)
    #             .view(-1, 1)
    #             .repeat(1, 1)
    #             .view(-1)
    #             .to(device)
    #     )
    # encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
    #         0, expanded_batch_idxs
    #     )
    model_inputs = {"input_ids": None,
                    "past_key_values": None,
                    "encoder_outputs": encoder_outputs,
                    "decoder_inputs_embeds": decoder_inputs_embeds,
                    }
    outputs = model(**model_inputs, use_cache=False,
                    return_dict=True, output_attentions=True)
    return outputs


def ig_dec(
    bart_model, dec_inp_embedding, encoder_outputs,
    dec_ref_embedding, tgt_class, device, num_steps=51
):
    interp_dec_embedding = interpolate_vectors(
        dec_ref_embedding, dec_inp_embedding, num_steps, device)
    encoder_outputs['last_hidden_state'] = encoder_outputs['last_hidden_state'].repeat(
        (num_steps, 1, 1))
    with torch.enable_grad():
        interp_dec_embedding.retain_grad()
        interp_out = forward_enc_dec_step(
            bart_model, encoder_outputs=encoder_outputs, decoder_inputs_embeds=interp_dec_embedding)

        logits = interp_out.logits[:, -1, :]
        target = torch.ones(num_steps, dtype=torch.long,
                            device=device) * tgt_class

        loss = torch.nn.functional.cross_entropy(logits, target)
        # loss.backward(retain_graph=True)
        loss.backward()
        logger.info(f"Loss: {loss.tolist()}")
        raw_grad = interp_dec_embedding.grad

        # Approximate the integral using the trapezodal rule
        approx_grad = (raw_grad[:-1] + raw_grad[1:]) / 2
        # print(approx_grad.size())
        avg_grad = torch.mean(approx_grad, dim=0)  # input len, hdim
        # print(encoder_outputs.last_hidden_state.size())
        # print(ref_encoder_outputs.last_hidden_state.size())
        integrated_gradient = (
            dec_inp_embedding[0] - dec_ref_embedding[0]) * avg_grad  # seq_len, hdim
    return integrated_gradient


def bart_decoder_forward_embed(input_ids, embed_tokens, embed_scale):
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    inputs_embeds = embed_tokens(input_ids) * embed_scale
    return inputs_embeds


def ig_enc(bart_model, dec_inp_embedding, tgt_class: int, encoder_outputs, ref_encoder_outputs, device, num_steps=51):
    hid_states_ref_encoder_outputs = ref_encoder_outputs.last_hidden_state
    hid_states_inp_encoder_outputs = encoder_outputs.last_hidden_state
    interp_last_hidden_state = interpolate_vectors(
        hid_states_ref_encoder_outputs, hid_states_inp_encoder_outputs, num_steps, device)

    # interp_encoder_outputs = BaseModelOutput()
    interp_encoder_outputs = ref_encoder_outputs
    interp_encoder_outputs.last_hidden_state = interp_last_hidden_state

    # interp_decoded = decoded_inputs.repeat(num_steps, 1)
    # interp_decoder_input_ids = torch.LongTensor(interp_decoded).to(device)

    with torch.enable_grad():
        interp_encoder_outputs.last_hidden_state.retain_grad()
        interp_out = forward_enc_dec_step(
            bart_model, encoder_outputs=interp_encoder_outputs, decoder_inputs_embeds=dec_inp_embedding)

        logits = interp_out.logits[:, -1, :]
        target = torch.ones(num_steps, dtype=torch.long,
                            device=device) * tgt_class

        loss = torch.nn.functional.cross_entropy(logits, target)
        # loss.backward(retain_graph=True)
        loss.backward()
        logger.info(f"Loss: {loss.tolist()}")
        raw_grad = interp_encoder_outputs.last_hidden_state.grad

        # Approximate the integral using the trapezodal rule
        approx_grad = (raw_grad[:-1] + raw_grad[1:]) / 2
        # print(approx_grad.size())
        avg_grad = torch.mean(approx_grad, dim=0)  # input len, hdim
        # print(encoder_outputs.last_hidden_state.size())
        # print(ref_encoder_outputs.last_hidden_state.size())
        integrated_gradient = (encoder_outputs.last_hidden_state - ref_encoder_outputs.last_hidden_state[
            0]) * avg_grad  # seq_len, hdim
    return integrated_gradient


def gen_ref_input(batch_size, seq_len, bos_token_id, pad_token_id, eos_token_id, device) -> torch.Tensor:
    token_reference = TokenReferenceBase(
        reference_token_idx=pad_token_id)
    ref_input = token_reference.generate_reference(
        seq_len, device=device)
    ref_input = ref_input.unsqueeze(0)
    expanded_input = ref_input.repeat((batch_size, 1))
    expanded_input[:, 0] = bos_token_id
    expanded_input[:, -1] = eos_token_id
    return expanded_input


def new_step_int_grad(input_ids, actual_word_id, prefix_token_ids, num_run_cut, model_pkg, device):

    input_ids = torch.LongTensor(input_ids).to(device).unsqueeze(0)
    # input_ids = input_ids[:,:400]
    batch_size, seq_len = input_ids.size()
    assert batch_size == 1
    # encode enc input
    enc_reference = gen_ref_input(num_run_cut, seq_len, tokenizer.bos_token_id,
                                  tokenizer.pad_token_id, tokenizer.eos_token_id, device)
    start_time = time.time()
    model_encoder = model_pkg['sum'].model.encoder
    encoder_outputs = model_encoder(
        input_ids.to(device), return_dict=True)

    ref_encoder_outputs = model_encoder(
        enc_reference, return_dict=True)
    assert isinstance(ref_encoder_outputs, BaseModelOutput)

    # encode dec input
    prefix_token_ids = prefix_token_ids.to(device)
    decoder_input_ids = prefix_token_ids.repeat((num_run_cut, 1))

    dec_seq_len = decoder_input_ids.size()[-1]
    model_decoder = model_pkg['sum'].model.decoder
    embed_scale = model_decoder.embed_scale
    embed_tokens = model_decoder.embed_tokens
    # dec input embedding
    dec_inp_embedding = bart_decoder_forward_embed(
        decoder_input_ids, embed_tokens, embed_scale)
    # # dec ref embedding
    # dec_ref_embedding = bart_decoder_forward_embed(
    #     dec_reference, embed_tokens, embed_scale)

    ig_enc_result = ig_enc(model_pkg['sum'],
                           dec_inp_embedding=dec_inp_embedding, encoder_outputs=encoder_outputs,
                           ref_encoder_outputs=ref_encoder_outputs, tgt_class=actual_word_id, device=device, num_steps=num_run_cut)
    # ig_dec_result = ig_dec(
    #     model_pkg['sum'],
    #     dec_inp_embedding=dec_inp_embedding, encoder_outputs=encoder_outputs,
    #     dec_ref_embedding=dec_ref_embedding, tgt_class=actual_word_id, device=device, num_steps=num_run_cut)
    ig_enc_result = summarize_attributions(ig_enc_result)
    duration = time.time() - start_time
    # ig_dec_result = summarize_attributions(ig_dec_result)
    if random.random() < 0.1:
        extracted_attribution = ig_enc_result.squeeze(0)
        input_doc = input_ids.squeeze(0)
        viz = simple_viz_attribution(
            tokenizer, input_doc, extracted_attribution)
        logger.info(viz)
        # extracted_attribution = ig_dec_result
        # viz = simple_viz_attribution(
        #     tokenizer, decoder_input_ids[0], extracted_attribution)
        # logger.info(viz)
    return ig_enc_result, duration


if __name__ == "__main__":

    parser = common_args()

    args = parser.parse_args()
    args = fix_args(args)
    logger.info(args)

    device = args.device
    acc_duration = 0
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

            ig_enc_result, duration = new_step_int_grad(input_doc, actual_word_id=step['tgt_token_id'], prefix_token_ids=step['prefix_token_ids'],
                                                        num_run_cut=args.num_run_cut, model_pkg=model_pkg, device=device)
            acc_duration += duration
            result = ig_enc_result.squeeze(0).cpu().detach()
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
