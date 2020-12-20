"""Run a classification model with gradient attribution methods like Integrated gradient or InputGrad"""
import torch

from lib.model_wrap import WrapExplainEnc, WrapDistillBERT, WrapBERT
from lib.attr_ig import IntGrad
from lib.util import get_sentiment_data

DISC_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
# DISC_MODEL = "barissayil/bert-sentiment-analysis-sst"
# DISC_MODEL = 'bert-base-uncased'

if __name__ == '__main__':
    name_model = DISC_MODEL
    name_tokenizer = name_model
    device = 'cuda:0'

    if 'distil' in name_model:
        model_wrap = WrapDistillBERT(device=device, name_model=name_model, name_tokenizer=name_tokenizer)
    else:
        model_wrap = WrapBERT(device=device, name_model=name_model, name_tokenizer=name_tokenizer)
    my_ig = IntGrad(model_wrap)
    sentiment_dataset = get_sentiment_data()
    EXAMPLES_TO_RUN = 25
    total_outputs = []
    for example in sentiment_dataset:
        input_doc = example['text']
        label = example['label']

        input_doc_tok_result = model_wrap.tokenizer(text=input_doc, truncation=True, max_length=100,
                                                    return_tensors='pt')
        input_doc_input_ids, input_doc_input_attn = input_doc_tok_result['input_ids'], input_doc_tok_result[
            'attention_mask']
        # input_doc_input_ids = input_doc_input_ids
        # input_doc_input_ids = input_doc_input_ids[:100]
        input_doc_input_ids = input_doc_input_ids.to(device)
        result_grad = my_ig.entrance_explain_enc(input_doc_input_ids)
        total_outputs.append(result_grad)
        if len(total_outputs) >= EXAMPLES_TO_RUN:
            break
