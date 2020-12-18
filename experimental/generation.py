
import torch

from transformers import BatchEncoding, PreTrainedTokenizer
from typing import Optional,List
from captum.attr import LayerIntegratedGradients, LayerGradientShap, LayerGradientXActivation, TokenReferenceBase
from argparse import Namespace

"""
Setting: LayerIntegratedGradient over embedding. 
No cache and batch size = 1 due to captum issues. 
"""

# layer wise: inputs, baselines, target, additional_forward_args
# for each summary, the model needs to encode the document again and again?

class SumGen(torch.nn.Module):
    def __init__(self, model, tokenizer: PreTrainedTokenizer, attribution_func, use_cache=False, max_len=50):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_cache = use_cache
        self.attr = attribution_func(self.forward_step, self.model.model.shared)
        self.encoder = self.model.get_encoder()

    def prepare_batch_inp(self, input_articles: List[str], tgt_summaries: Optional[List[str]]) -> BatchEncoding:
        return self.tokenizer.prepare_seq2seq_batch(src_texts=input_articles, tgt_texts=tgt_summaries,
                                                    return_tensors='pt')

    def run_attribution(self, input_doc, attn_mask, tgt_sum=None):

        device = input_doc.device
        batch_size = input_doc.shape[0]
        cur_len = 1
        has_eos = [False for _ in range(batch_size)]
        bos_token_id = self.tokenizer.bos_token_id
        decoded = [[bos_token_id] for _ in range(batch_size)]
        decoder_input_ids = torch.LongTensor(decoded).to(device)
        past_key_values = None
        seq_length = input_doc.shape[1]
        token_reference = TokenReferenceBase(reference_token_idx=self.tokenizer.pad_token_id)
        reference_indice = token_reference.generate_reference(seq_length, device=device)
        reference_indices = torch.stack([reference_indice for _ in range(batch_size)], dim=0)
        reference_indices[:, 0] = self.tokenizer.bos_token_id
        reference_indices[:, -1] = self.tokenizer.eos_token_id

        while cur_len < self.max_len and (not all(has_eos)):

            additional_input = {
                "attn_mask": attn_mask,
                "past_key_values": past_key_values,
                "decoder_input_ids": decoder_input_ids,
                "attr_mode": False
            }
            cur_decoded, cur_past_key_values, cur_decoder_input_ids = self.forward_step(input_doc, additional_input
                                                                                        )
            print('run normal')
            # cur_decoded is just a list with token id
            for idx, cur_dec_tok in enumerate(cur_decoded):
                if cur_dec_tok == self.tokenizer.eos_token_id:
                    has_eos[idx] = True
            if tgt_sum is None:
                # target = cur_decoder_input_ids[:, -1].unsqueeze(0)
                target = cur_decoded[0][0]  # assume batch size = 1
                print(f'target : {target}')
            else:
                pass
            additional_input['attr_mode'] = True
            additional_input['target'] = target
            print()
            attribution_result = self.attr.attribute(inputs=input_doc, baselines=reference_indices,
                                                     additional_forward_args=additional_input
                                                     )
            print(attribution_result)
            # past_key_values = cur_past_key_values
            past_key_values = None
            decoder_input_ids = cur_decoder_input_ids
        print("end of decoding")

    def forward_step(self, input_doc: torch.LongTensor,
                     additional_input_args: dict,
                     ):
        """The forward pass for one single time step

        """
        attn_mask, past_key_values, decoder_input_ids, attr_mode = \
            additional_input_args['attn_mask'], additional_input_args['past_key_values'], additional_input_args[
                'decoder_input_ids'], additional_input_args['attr_mode'],
        if 'target' in additional_input_args:
            target = additional_input_args['target']
        else:
            target = None
        print("encoder:")
        print(input_doc.size())
        if attn_mask:
            print(attn_mask.size())
        encoder_outputs = self.encoder(input_doc, attention_mask=attn_mask, return_dict=True)
        batch_size = input_doc.shape[0]
        device = input_doc.device

        expanded_batch_idxs = (
            torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, 1)
                .view(-1)
                .to(device)
        )
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_batch_idxs
        )
        model_inputs = {"input_ids": None,
                        "past_key_values": past_key_values,
                        "attention_mask": attn_mask,
                        "encoder_outputs": encoder_outputs,
                        "decoder_input_ids": decoder_input_ids,
                        }
        print(model_inputs)
        outputs = self.model(**model_inputs, use_cache=self.use_cache, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        if attr_mode:
            print(next_token_logits.shape)
            # vector_next_token_logits = next_token_logits.squeeze(0)
            return next_token_logits[:,target]  # assume batch size = 1
        next_token = next_token.unsqueeze(-1)
        cur_decoded = next_token.tolist()
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values

        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
        return cur_decoded, past_key_values, decoder_input_ids
