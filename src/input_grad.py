from helper_run_bart import write_pkl_to_disk
from transformers.modeling_outputs import BaseModelOutput
from helper_run_bart import init_bart_sum_model, init_bart_family
from util import *
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
import torch
from captum.attr._utils.visualization import format_word_importances

from int_grad import forward_enc_dec_step, bart_decoder_forward_embed, summarize_attributions, simple_viz_attribution


def step_input_grad(input_ids, actual_word_id, prefix_token_ids, model_pkg, device):
    input_ids = input_ids[:, :400]
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-device", help="device to use", default='cuda:1')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')
    parser.add_argument("-mname_lm", default='facebook/bart-large')
    parser.add_argument("-mname_sum", default='facebook/bart-large-xsum')
    parser.add_argument("-batch_size", default=40)
    parser.add_argument('-max_samples', default=1000)
    parser.add_argument('-num_run_cut', default=50)
    parser.add_argument('-truncate_sent', default=15,
                        help='the max sent used for perturbation')
    parser.add_argument('-dir_read', default='/mnt/data0/jcxu/meta_data_ref',
                        help="Path of the meta data to read.")
    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/output_inpg",
                        help="The location to save output data. ")
    args = parser.parse_args()
    logger.info(args)
    args.dir_save = args.dir_save + '_' + args.data_name
    args.dir_read = args.dir_read + '_' + args.data_name
    if not os.path.exists(args.dir_save):
        os.makedirs(args.dir_save)
    device = args.device

    model_lm, model_sum, model_sum_ood, tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device, no_lm=True, no_ood=True)
    logger.info("Done loading BARTs.")
    model_pkg = {'sum': model_sum, 'tok': tokenizer}
    all_files = os.listdir(args.dir_read)
    for f in all_files:
        outputs = []
        step_data, meta_data = read_meta_data(args.dir_read, f)
        uid = meta_data['id']
        for step in step_data:
            inp_grad_result = step_input_grad(meta_data['doc_token_ids'], actual_word_id=step['tgt_token_id'],
                                              prefix_token_ids=step['prefix_token_ids'], model_pkg=model_pkg, device=device)

            result = inp_grad_result.squeeze(0).cpu().detach()
            outputs.append(result)
        skinny_meta = {
            'doc_token_ids': meta_data['doc_token_ids'].squeeze(),
            'output': outputs
        }
        write_pkl_to_disk(args.dir_save, uid, skinny_meta)
        print(f"Done {uid}.pkl")
