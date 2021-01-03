from transformers.modeling_outputs import BaseModelOutput
from helper_run_bart import init_bart_sum_model
from util import *
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
import torch


def interpolate_vectors(ref_vec, inp_vec, num_steps, device):
    interp_step_vec = (inp_vec - ref_vec)/num_steps
    ranges = torch.arange(
        1, num_steps + 1).unsqueeze(-1).unsqueeze(-1).to(device)
    rows = ranges * interp_step_vec
    interp_final = rows + ref_vec
    return interp_final


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


def ig_dec():
    pass


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
        target = torch.ones(num_steps, dtype=torch.long) * tgt_class

        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward(retain_graph=True)
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


def _step_ig(interest: str, actual_word_id: int, summary_prefix: List[str], document: List[str], document_sents: List[str], num_run_cut: int, model_pkg, device):
    # assume the batch size is one because we are going to use batch_size as the num trials for ig
    input_doc = tokenizer(document, max_length=300,
                          return_tensors='pt', truncation=True, padding=True)
    # input_doc = input_doc.to(device)

    batch_size, seq_len = input_doc['input_ids'].size()
    assert batch_size == 1

    # encode enc input

    enc_reference = gen_ref_input(num_run_cut, seq_len, tokenizer.bos_token_id,
                                  tokenizer.pad_token_id, tokenizer.eos_token_id, device)
    model_encoder = model_pkg['sum'].model.encoder
    encoder_outputs = model_encoder(
        input_doc['input_ids'].to(device), return_dict=True)

    ref_encoder_outputs = model_encoder(
        enc_reference, return_dict=True)
    assert isinstance(ref_encoder_outputs, BaseModelOutput)

    # encode dec input
    decoder_input_ids = torch.LongTensor(tokenizer.encode(
        summary_prefix[0], return_tensors='pt')).to(device)

    decoder_input_ids = decoder_input_ids.expand(
        (num_run_cut, decoder_input_ids.size()[-1]))
    # ATTN: remove the EOS token from the prefix!
    dec_seq_len = decoder_input_ids.size()[-1]
    decoder_input_ids = decoder_input_ids[:, :-1]
    model_decoder = model_pkg['sum'].model.decoder
    dec_reference = gen_ref_input(num_run_cut, dec_seq_len, tokenizer.bos_token_id,
                                  tokenizer.pad_token_id, tokenizer.eos_token_id, device)

    embed_scale = model_decoder.embed_scale
    embed_tokens = model_decoder.embed_tokens
    # dec input embedding
    dec_inp_embedding = bart_decoder_forward_embed(
        decoder_input_ids, embed_tokens, embed_scale)
    # dec ref embedding
    dec_ref_embedding = bart_decoder_forward_embed(
        dec_reference, embed_tokens, embed_scale)

    ig_enc(model_pkg['sum'],
           dec_inp_embedding=dec_inp_embedding, encoder_outputs=encoder_outputs,
           ref_encoder_outputs=ref_encoder_outputs, tgt_class=actual_word_id, device=device)
    ig_dec(
        model_pkg['sum'],
           dec_inp_embedding=dec_inp_embedding, encoder_outputs=encoder_outputs,
           dec_ref_embedding=dec_ref, tgt_class=actual_word_id, device=device
    )
    exit()


if __name__ == "__main__":
    interest = "Google"
    summary_prefix = "She didn't know her kidnapper but he was using"
    document = 'Google Maps is a web mapping service developed by Google. It offers satellite imagery, aerial photography, street maps, 360 interactive panoramic views of streets, real-time traffic conditions, and route planning for traveling by foot, car, bicycle, air and public transportation.'
    document_sents = ['Google Maps is a web mapping service developed by Google. ',
                      'It offers satellite imagery, aerial photography, street maps, 360 interactive panoramic views of streets, real-time traffic conditions, and route planning for traveling by foot, car, bicycle, air and public transportation.']
    mname = 'facebook/bart-large-xsum'
    device = 'cpu'
    model_sum, bart_tokenizer = init_bart_sum_model(mname, device)
    model_pkg = {'lm': None, 'sum': model_sum, 'ood': None, 'tok': bart_tokenizer,
                 'spacy': None}
    actual_word_id = tokenizer.encode(" "+interest)[1]
    _step_ig(interest, actual_word_id, summary_prefix=[summary_prefix], document=[
             document], document_sents=document_sents, num_run_cut=51, model_pkg=model_pkg, device=device)
