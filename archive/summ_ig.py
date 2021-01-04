# Running Integreated Gradient for an encoder-decoder model
from util import *

def run_model_core():
    
    pass

def ig_full_model(model, tokenizer, input_text: List[str], sum_prefix: List[str], device='cuda:0', output_attentions=False, output_dec_hid=False):
    
    inputs = tokenizer(input_text, max_length=300,
                       return_tensors='pt', truncation=True, padding=True)
    encoder_outputs = model.model.encoder(
        inputs['input_ids'].to(device), return_dict=True)

    sum_prefix = [_sum_prefix.strip() for _sum_prefix in sum_prefix]
    batch_size = len(input_text)
    assert batch_size == len(sum_prefix)

    if batch_size > 1 and sum_prefix[0] != sum_prefix[1]:
        raise NotImplementedError('So far we assume the prefix are duplicates')
    decoder_input_ids = torch.LongTensor(tokenizer.encode(
        sum_prefix[0], return_tensors='pt')).to(device)

    decoder_input_ids = decoder_input_ids.expand(
        (batch_size, decoder_input_ids.size()[-1]))

    # ATTN: remove the EOS token from the prefix!
    decoder_input_ids = decoder_input_ids[:, :-1]

    model.output_attentions = output_attentions
    model_inputs = {"input_ids": None,
                    "past_key_values": None,
                    "encoder_outputs": encoder_outputs,
                    "decoder_input_ids": decoder_input_ids,
                    }
    outputs = model(**model_inputs, output_attentions=output_attentions, output_hidden_states=output_dec_hid,
                    use_cache=False, return_dict=True)

    if output_attentions:
        # use cross attention as the distribution
        # last layer.   batch=1, head, dec len, enc len
        # by default we use the last layer of attention
        output, p = get_cross_attention(
            outputs['cross_attentions'], inputs['input_ids'], device=device)
        return output, p
    else:
        # batch, dec seq, vocab size
        next_token_logits = outputs.logits[:, -1, :]
        prob = next_token_logits.softmax(dim=-1)
        next_token = torch.argmax(next_token_logits, dim=-1)
        # next_token = next_token.unsqueeze(-1)
        next_token = next_token.tolist()    # confrim nested list?

        output = [tokenizer.decode(tk) for tk in next_token]
        logging.info(f"Next token: {output}")
        outputs['output'] = output
        return outputs, prob
