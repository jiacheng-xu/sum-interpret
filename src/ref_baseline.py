# from pprint import pprint
# from transformers import pipeline
# nlp = pipeline("fill-mask")
# pprint(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased", return_dict=True)
original_seq = "Distilled models are smaller than the models they mimic. "
inputs = tokenizer.batch_encode_plus([original_seq, original_seq], return_tensors='pt')
mask_token_index = torch.where(inputs == tokenizer.mask_token_id)[1]
token_logits = model(inputs).logits
print(token_logits)
print(token_logits.size())
exit()
sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."
input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
token_logits = model(input).logits
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

def slot_fill(input_ids, lm_model, tokenzier):
    len_of_seq = input_ids.size()