import datasets

from datasets import load_dataset

dataset = load_dataset("xsum")
dataset = dataset.shuffle()
dev_set = dataset['validation']
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

max_samples = 5000
batch_size = 300

from transformers import BartTokenizer, BartForConditionalGeneration
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-xsum')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum')


import torch

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
from lib.run_lime import run_model

model = model.to(torch_device)

batch_for_model = []
ref_sums = []
dec_summaries = []
for idx in range(max_samples):
    inp_doc, ref_sum = dev_set['document'][idx], dev_set['summary'][idx]
    ref_sums.append([[ref_sum]])
    batch_for_model.append(inp_doc)
    if len(batch_for_model) == batch_size:
        print(f"Run {len(batch_for_model)}")
        model_output = run_model(model, tokenizer, batch_for_model, device=torch_device)
        dec_summaries += model_output
        batch_for_model = []
if len(batch_for_model) != 0:
    model_output = run_model(model, tokenizer, batch_for_model, device=torch_device)
    dec_summaries += model_output
    batch_for_model = []
print("\n".join(dec_summaries))

summary = [ [x] for x in dec_summaries]

from pythonrouge.pythonrouge import Pythonrouge

# system summary(predict) & reference summary
# summary = [[" Tokyo is the one of the biggest city in the world."]]
# reference = [[["The capital of Japan, Tokyo, is the center of Japanese economy."]]]

# initialize setting of ROUGE to eval ROUGE-1, 2, SU4
# if you evaluate ROUGE by sentence list as above, set summary_file_exist=False
# if recall_only=True, you can get recall scores of ROUGE
rouge = Pythonrouge(summary_file_exist=False,
                    summary=summary, reference=ref_sums,
                    n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                    recall_only=False, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=50,
                    use_cf=True, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
score = rouge.calc_score()
print(score)