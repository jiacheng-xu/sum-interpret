from util import *


def init_bart_sum_model(mname='sshleifer/distilbart-cnn-6-6', device='cuda:0'):
    model = BartForConditionalGeneration.from_pretrained(mname).to(device)
    tokenizer = BartTokenizer.from_pretrained(mname)
    return model, tokenizer


def init_bart_lm_model(mname='facebook/bart-large', device='cuda:0'):
    model = BartForConditionalGeneration.from_pretrained(mname).to(device)
    tokenizer = BartTokenizer.from_pretrained(mname)
    return model, tokenizer


# TODO below ready to clean

def run_model(model, tokenizer, input_text: List, device, sum_prefix=""):
    inputs = tokenizer(input_text, max_length=300,
                       return_tensors='pt', truncation=True, padding=True)
    # Generate Summary
    # logging.info(torch.max(inputs['input_ids']))
    if sum_prefix:
        encoder_outputs = model.model.encoder(
            inputs['input_ids'].to(device), return_dict=True)
        if sum_prefix:
            sum_prefix = sum_prefix.strip()
            batch_size = len(input_text)
            decoder_input_ids = torch.LongTensor(tokenizer.encode(
                sum_prefix, return_tensors='pt')).to(device)
            decoder_input_ids = decoder_input_ids.expand(
                (batch_size, decoder_input_ids.size()[-1]))
            decoder_input_ids = decoder_input_ids[:, :-1]
        else:
            decoder_input_ids = torch.LongTensor(
                tokenizer.bos_token_id).to(device)

        model_inputs = {"input_ids": None,
                        "past_key_values": None,
                        "encoder_outputs": encoder_outputs,
                        "decoder_input_ids": decoder_input_ids,
                        }
        outputs = model(**model_inputs, use_cache=False, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        prob = next_token_logits.softmax(dim=-1)
        next_token = torch.argmax(next_token_logits, dim=-1)
        # next_token = next_token.unsqueeze(-1)
        next_token = next_token.tolist()

        output = [tokenizer.decode(tk) for tk in next_token]
        logging.info(f"Next token: {output}")
    else:
        summary_ids = model.generate(inputs['input_ids'].to(
            device), num_beams=1, max_length=30, early_stopping=True)
        output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                  summary_ids]
        prob = None
    return output, prob


def run_full_model(model, tokenizer, input_text, sum_prefix, encoder_outputs=None, device='cuda:0', output_attentions=False, output_dec_hid=False):

    if not encoder_outputs:
        inputs = tokenizer(input_text, max_length=300,
                           return_tensors='pt', truncation=True, padding=True)
        encoder_outputs = model.model.encoder(
            inputs['input_ids'].to(device), return_dict=True)

    if sum_prefix:
        sum_prefix = sum_prefix.strip()
        batch_size = len(input_text)
        decoder_input_ids = torch.LongTensor(tokenizer.encode(
            sum_prefix, return_tensors='pt')).to(device)
        decoder_input_ids = decoder_input_ids.expand(
            (batch_size, decoder_input_ids.size()[-1]))
        decoder_input_ids = decoder_input_ids[:, :-1]
    else:
        decoder_input_ids = torch.LongTensor(tokenizer.bos_token_id).to(device)
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
        cross_attns = outputs['cross_attentions'][-1]
        attn = cross_attns[0, :, -1, :]    # head, enc len

        mean_attn = torch.mean(attn, dim=0)
        assert len(mean_attn.size()) == 1

        topk = min(30, mean_attn.size()[0])

        values, indices = torch.topk(mean_attn, k=topk)
        values = values.detach().cpu().tolist()
        indices = indices.detach().cpu().tolist()
        input_ids_list = inputs['input_ids'][0].tolist()  # batch=1, enc len
        output = tokenizer.decode(
            int(input_ids_list[indices[0]]))
        logging.info(f"Most attention: {output}")
        p_list = [0.0 for _ in range(tokenizer.vocab_size)]
        for v, i in zip(values, indices):
            p_list[input_ids_list[i]] += v
        p = torch.as_tensor(p_list, device=device)
        p = p / p.sum()
        return output, p
    else:
        next_token_logits = outputs.logits[:, -1, :]
        prob = next_token_logits.softmax(dim=-1)
        next_token = torch.argmax(next_token_logits, dim=-1)
        # next_token = next_token.unsqueeze(-1)
        next_token = next_token.tolist()

        output = [tokenizer.decode(tk) for tk in next_token]
        logging.info(f"Next token: {output}")
        outputs['output'] = output
        return outputs, prob


def run_implicit(model, tokenizer, device, sum_prefix=""):
    output, prob = run_full_model(
        model, tokenizer, " ", sum_prefix, device=device)
    return output, prob


def run_explicit(model, tokenizer, original_doc, sum_prefix):
    output, prob = run_full_model(model, tokenizer, input_text, sum_prefix)
    pass


def run_attn(model, tokenizer, input_text, sum_prefix="", device='cuda:0'):
    model.output_attentions = True
    output, prob = run_full_model(
        model, tokenizer, input_text, sum_prefix, device=device, output_attentions=True)
    model.output_attentions = False  # reset
    return output, prob


def run_lm(model, tokenizer, device, sum_prefix=""):
    sum_prefix = sum_prefix.strip()
    # Mask filling only works for bart-large
    TXT = f"{sum_prefix} <mask> "
    input_ids = tokenizer([TXT], return_tensors='pt')['input_ids'].to(device)
    logits = model(input_ids, return_dict=True)['logits']
    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    probs = logits[0, masked_index].softmax(dim=0)
    values, predictions = probs.topk(10)
    top_output = tokenizer.decode(predictions).split()
    return top_output[0], probs


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
