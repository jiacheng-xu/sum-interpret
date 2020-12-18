#%%
import torch


from transformers import BartTokenizer, BartForConditionalGeneration
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

model_name = 'google/pegasus-xsum'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

encoder = model.get_encoder()
#%%

# BART function
def forward_step(input_doc,decoder_input_ids,
                 additional_input_args: dict,
                 # attn_mask, past_key_values, decoder_input_ids, attr_mode: bool
                 ):
    attn_mask, past_key_values, _, attr_mode = \
        additional_input_args['attn_mask'], additional_input_args['past_key_values'], additional_input_args[
            'decoder_input_ids'], additional_input_args['attr_mode']
    print(input_doc.size())
    print(decoder_input_ids.size())
    # input_doc = input_doc.permute(1,0)
    # decoder_input_ids = decoder_input_ids.permute(1,0)
    encoder_outputs = encoder(input_doc, attention_mask=None, return_dict=True)
    batch_size = input_doc.shape[0]
    print(f"Batch size {batch_size}")
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

    outputs = model(**model_inputs, use_cache=False, return_dict=True)
    next_token_logits = outputs.logits[:, -1, :]
    if attr_mode:
        return next_token_logits.unsqueeze(0)

#%%


from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from captum.attr import LayerGradientShap,LayerDeepLift
token_reference = TokenReferenceBase(reference_token_idx=tokenizer.pad_token_id)

lig = LayerDeepLift(forward_step, model.model.encoder.embed_tokens)

TXT ="The Pegasus model was proposed in PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization by Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu on Dec 18, 2019. According to the abstract, Pegasus’ pretraining task is intentionally similar to summarization: important sentences are removed/masked from an input document and are generated together as one output sequence from the remaining sentences, similar to an extractive summary."
input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']

seq_length = input_ids.shape[1]
reference_indices = token_reference.generate_reference(seq_length, device='cpu').unsqueeze(0)
# reference_indices[:,0] = tokenizer.bos_token_id # BART
reference_indices[:,0] = tokenizer.eos_token_id # PEGASUS
reference_indices[:,-1] = tokenizer.eos_token_id

device = input_ids.device
decoded = [tokenizer.encode("The model is proposed by someone else")]
decoder_input_ids = torch.LongTensor(decoded).to(device)
dec_len = len(decoded[0])
dec_reference_indices = token_reference.generate_reference(dec_len, device='cpu').unsqueeze(0)
# dec_reference_indices[:,0] = tokenizer.bos_token_id  # BART

# input_ids = input_ids.permute(1,0)
# decoder_input_ids = decoder_input_ids.permute(1,0)
# reference_indices = reference_indices.permute(1,0)

additional_input = {
                "attn_mask": None,
                "past_key_values": None,
                "decoder_input_ids": decoder_input_ids, "attr_mode": True
            }
forward_step(input_doc=input_ids,decoder_input_ids=decoder_input_ids,additional_input_args=additional_input)
print("!-"*20)
attributions_ig, delta = lig.attribute((input_ids,decoder_input_ids), additional_forward_args=additional_input,
                                       baselines=(reference_indices,dec_reference_indices), \
                                           # n_steps=50,
                                       return_convergence_delta=True)
#%%


