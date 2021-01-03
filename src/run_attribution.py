from typing import List
from torch.nn import CrossEntropyLoss
from captum.attr import TokenReferenceBase
from scipy.stats import entropy

from attr_ig import simple_viz_attribution
import torch

from transformers import BatchEncoding, PreTrainedTokenizer
from typing import Optional, List
from captum.attr import LayerIntegratedGradients, LayerGradientShap, LayerGradientXActivation, TokenReferenceBase
from argparse import Namespace

from util import *


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1)
    attributions = attributions / torch.sum(attributions)
    return attributions


def init_model(mname='sshleifer/distilbart-cnn-6-6', device='cuda:0'):
    from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
    model = BartForConditionalGeneration.from_pretrained(mname).to(device)
    tokenizer = BartTokenizer.from_pretrained(mname)
    return model, tokenizer


def tokenize_text(tokenizer, raw_string, max_len=500):
    token_ids = tokenizer(raw_string, max_length=max_len,
                          return_tensors='pt', truncation=True)
    token_ids_list = token_ids['input_ids'].tolist()[0]
    doc_str = "".join([tokenizer.decode(x) for x in token_ids_list])
    doc_str_lower = doc_str.lower()
    # reverse_eng_token_str = [tokenizer.decode(token) for token in token_ids_list]
    # lowercased_token_str = [x.lower() for x in reverse_eng_token_str]
    # lower_token_ids = [tokenizer.encode(x) for x in lowercased_token_str]
    # return token_ids, lower_token_ids, reverse_eng_token_str, lowercased_token_str
    return token_ids, doc_str, doc_str_lower


def get_xsum_data(split='validation'):
    # Load sentiment 140 dataset and return
    from datasets import load_dataset

    dataset = load_dataset('xsum', split=split)
    # # def renorm_label(example):
    # #     example['sentiment'] = 1 if example['sentiment'] >=3 else 0
    # #     return example
    # updated_dataset = dataset.map(renorm_label)
    logger.info(dataset.features)
    logger.info(dataset[0])
    logger.info("=" * 40)
    return dataset


"""
Setting: LayerIntegratedGradient over embedding. 
No cache and batch size = 1 due to captum issues. 
"""


def fast_ig_enc_dec(decoded_inputs, tgt_class: int, encoder_outputs, ref_encoder_outputs, device, num_steps=51, ):
    loss_fct = CrossEntropyLoss()
    interp_step_vec = (encoder_outputs.last_hidden_state -
                       ref_encoder_outputs.last_hidden_state) / num_steps
    ranges = torch.arange(
        1, num_steps + 1).unsqueeze(-1).unsqueeze(-1).to(device)
    repeated_raw = ranges * interp_step_vec
    interp_last_hidden_state = repeated_raw + ref_encoder_outputs.last_hidden_state
    interp_encoder_outputs = ref_encoder_outputs
    interp_encoder_outputs.last_hidden_state = interp_last_hidden_state

    # interp_decoded = [decoded_inputs for _ in range(num_steps)]
    interp_decoded = decoded_inputs.repeat(num_steps, 1)
    interp_decoder_input_ids = torch.LongTensor(interp_decoded).to(device)
    with torch.enable_grad():
        interp_encoder_outputs.last_hidden_state.retain_grad()
        interp_out = forward_enc_dec_step(
            model, interp_encoder_outputs, interp_decoder_input_ids)
        logits = interp_out.logits[:, -1, :]
        target = torch.ones(num_steps, dtype=torch.long) * tgt_class

        loss = loss_fct(logits, target)
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


def forward_enc_dec_step(model, encoder_outputs, decoder_input_ids):
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
                    "decoder_input_ids": decoder_input_ids,
                    }
    outputs = model(**model_inputs, use_cache=False,
                    return_dict=True, output_attentions=True)
    return outputs


def forward_step(model, encoder_outputs, past_key_values, decoder_input_ids, inp_attn_mask=None):
    model_inputs = {"input_ids": None,
                    "past_key_values": past_key_values,
                    "attention_mask": inp_attn_mask,
                    "encoder_outputs": encoder_outputs,
                    "decoder_input_ids": decoder_input_ids,
                    }
    outputs = model(**model_inputs, use_cache=True, return_dict=True)
    next_token_logits = outputs.logits[:, -1, :]
    pred_distribution = torch.nn.functional.softmax(next_token_logits, dim=-1)
    numpy_pred_distb = pred_distribution.cpu().detach().numpy()
    ent = entropy(numpy_pred_distb, axis=-1)
    top5 = torch.topk(pred_distribution, 5, dim=-1, largest=True, sorted=True)
    next_token = torch.argmax(next_token_logits, dim=-1)
    next_token = next_token.unsqueeze(-1)
    cur_decoded = next_token.tolist()

    if "past_key_values" in outputs:
        past_key_values = outputs.past_key_values

    decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
    return ent, top5, cur_decoded, pred_distribution, past_key_values, decoder_input_ids


def run_one_example(data, device, model, tokenizer):
    document, ref_sum = data['document'], data['summary']
    token_ids, doc_str, _ = tokenize_text(tokenizer, document)

    input_doc = token_ids['input_ids'].to(device)
    encoder_outputs = model.model.encoder(input_doc, return_dict=True)

    batch_size = input_doc.shape[0]
    cur_len = 1
    max_len = 20
    has_eos = [False for _ in range(batch_size)]
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    decoded = [[bos_token_id] for _ in range(batch_size)]
    decoder_input_ids = torch.LongTensor(decoded).to(device)
    past_key_values = None
    seq_length = input_doc.shape[1]
    token_reference = TokenReferenceBase(
        reference_token_idx=tokenizer.pad_token_id)
    reference_indice = token_reference.generate_reference(
        seq_length, device=device)
    reference_indices = torch.stack(
        [reference_indice for _ in range(batch_size)], dim=0)
    # reference_indices[:, 0] = self.tokenizer.bos_token_id
    # reference_indices[:, -1] = self.tokenizer.eos_token_id
    ref_encoder_outputs = model.model.encoder(
        reference_indices, return_dict=True)

    all_entropy = []
    all_topk = []
    while cur_len < max_len and (not all(has_eos)):
        cur_len += 1
        logger.debug(f"Step: {cur_len}")
        last_decoder_input_ids = decoder_input_ids
        ent, top5, cur_decoded, pred_distribution, past_key_values, decoder_input_ids = forward_step(model,
                                                                                                     encoder_outputs,
                                                                                                     past_key_values,
                                                                                                     decoder_input_ids)
        all_entropy.append(ent[0])
        all_topk.append(top5)
        logger.info(f"Entropy: {ent[0]}")
        logger.info(f"Decoded token: {tokenizer.decode(cur_decoded[0])}")
        cur_decoded = [cur_dec_token[0] for cur_dec_token in cur_decoded]
        for idx in range(batch_size):
            if cur_decoded[idx] == tokenizer.eos_token_id or cur_decoded[idx] == 479:
                has_eos[idx] = True
        # Attribution
        # print(last_decoder_input_ids)
        ig = fast_ig_enc_dec(decoded_inputs=last_decoder_input_ids,
                             tgt_class=cur_decoded[0],
                             encoder_outputs=encoder_outputs,
                             ref_encoder_outputs=ref_encoder_outputs, device=device)
        extracted_attribution = summarize_attributions(ig)
        # process for viz
        extracted_attribution = extracted_attribution.squeeze(0)
        input_doc = input_doc.squeeze(0)
        viz = simple_viz_attribution(
            tokenizer, input_doc, extracted_attribution)
        logger.info(viz)

    decoder_input_ids = decoder_input_ids[:, 1:]  # REMOVE <s>
    all_decoded_tokens = decoder_input_ids.tolist()
    decoded_sents = [[tokenizer.convert_ids_to_tokens(
        x) for x in s] for s in all_decoded_tokens]

    decoded_sents_str = tokenizer.decode(all_decoded_tokens[0])
    logger.info(decoded_sents_str)


if __name__ == '__main__':

    all_data = get_xsum_data()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    mname = 'facebook/bart-large-xsum'
    model, tokenizer = init_model(mname=mname, device=device)

    # data = all_data[0]
    for data in all_data:
        run_one_example(data, device, model, tokenizer)
