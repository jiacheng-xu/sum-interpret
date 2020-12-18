import torch
from captum.attr import LayerIntegratedGradients

from transum.configs.config_env import bpe_tokenizer


class GPT_IG:
    def __init__(self):
        # setup

        from transformers import GPT2LMHeadModel
        from transformers import GPT2Tokenizer, GPT2Model
        self.device = torch.device('cuda:0')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.lig = LayerIntegratedGradients(self._wrap_function, self.model.transformer.wte)
        # layer_to_att = self.model.transformer.h[-1].attn
        # self.lig_attn = LayerIntegratedGradients(self._wrap_function, layer_to_att)

    def _wrap_function(self, _input_ids):
        outputs = self.model(_input_ids)
        logits = outputs[0]
        rt = logits[:, -1, :]
        # print(f"output size: {rt.size()}")
        return rt


def gpt_ig():
    ig = GPT_IG()

    try:
        while True:
            val = input("Enter a sentence: ")
            bpes = ig.tokenizer.encode(val, add_special_tokens=True)
            for t in range(1, len(bpes)):
                prefix = bpes[:t]
                predict = bpes[t]

                input_ids = torch.tensor(prefix).unsqueeze(0).to(ig.device)  # Batch size 1
                target_code = predict

                pred_logit = ig._wrap_function(input_ids)  # 1, 50257
                val, indices = torch.topk(pred_logit, 5, dim=1, largest=True)
                indices = indices.squeeze().tolist()

                """
                # attention attribution

                attributions_ig, delta = ig.lig_attn.attribute(input_ids, bpe_tokenizer.encode(".")[0],
                                                               target=target_code,
                                                               n_steps=500,
                                                               return_convergence_delta=True)
                print(attributions_ig)
                """
                """
                attributions_ig, delta = ig.lig.attribute(input_ids, bpe_tokenizer.encode(".")[0], target=target_code,
                                                          n_steps=500,
                                                          return_convergence_delta=True)
                visualize_attribution(attributions_ig, input_ids, target_code)
                attributions_ig, delta = ig.lig.attribute(input_ids, bpe_tokenizer.encode("a")[0], target=target_code,
                                                          n_steps=500,
                                                          return_convergence_delta=True)
                visualize_attribution(attributions_ig, input_ids, target_code)
                print("<p>Show Top K attribution</p>")
                for ind in indices:
                    print(bpe_tokenizer.decode(ind))
                    attributions_ig, delta = ig.lig.attribute(input_ids, 1234, target=int(ind), n_steps=500,
                                                              return_convergence_delta=True)
                    visualize_attribution(attributions_ig, input_ids, int(ind))
                print("<br><br>")
                # abs_attributions = torch.abs(attributions_ig)
                # visualize_attribution(abs_attributions, input_ids, target_code)
                """
    except KeyboardInterrupt:
        print("done")
