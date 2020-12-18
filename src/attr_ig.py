import torch
from torch import Tensor

from torch.nn import CrossEntropyLoss

from lib.model_wrap import WrapExplain
import logging

from captum.attr._utils.visualization import format_word_importances

DISC_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
GEN_MODEL = 'sshleifer/distilbart-xsum-12-1'


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1)
    attributions = attributions / torch.sum(attributions)
    return attributions


def simple_viz_attribution(tokenizer, input_ids, attribution_scores):
    token_in_list = input_ids.tolist()
    if isinstance(token_in_list[0], list):
        token_in_list = token_in_list[0]
    words = [tokenizer.decode(x) for x in token_in_list]
    attribution_scores_list = attribution_scores.tolist()
    # for w, ascore in zip(words, attribution_scores_list):
    #     logging.info('{:10} {:02.2f}'.format(w, ascore))

    output = format_word_importances(words, attribution_scores_list)
    return output


class IntGrad():
    def __init__(self, model):
        self.model: WrapExplain = model
        self.loss_fct = CrossEntropyLoss()

    def entrance_explain_enc(self, input_tensor, target_idx=None, baseline=None, num_steps=11):
        batch_size, seq_len = input_tensor.size()  # assume batch size =1 for now
        input_tensor = input_tensor.to(self.model.device)
        assert batch_size == 1
        expanded_input_tensor = input_tensor.repeat(num_steps, 1)

        # get the reference baseline
        ref_input_tensor = self.model.fill_pad_ref_baseline(num_steps, seq_len)
        # ref_input_tensor = self.model.fill_ref_baseline_lm_mask_filling(input_tensor, num_steps, seq_len)
        single_ref_inp = ref_input_tensor[:1]
        # load embeddings from the model   this should be word+type+position
        embed_input_tensor = self.model.get_full_embeddings(expanded_input_tensor)
        embed_ref_tensor = self.model.get_full_embeddings(ref_input_tensor)

        # do linear interpolation of these embeddings
        interp_step_vec = (embed_input_tensor - embed_ref_tensor) / num_steps
        ranges = torch.arange(1, num_steps + 1).unsqueeze(-1).unsqueeze(-1).to(self.model.device)
        repeated_raw = ranges * interp_step_vec
        interpolated_tensor = repeated_raw + embed_ref_tensor

        # run the model forward
        with torch.enable_grad():
            interpolated_tensor.retain_grad()
            logits = self.model.forward(inputs_embeds=interpolated_tensor, return_dict=True)

        logging.info(f"Logits: {logits.tolist()}")

        viz_pos = self.run_w_target_n_exaplain(logits, target=torch.ones(num_steps).long().to(self.model.device),
                                               tensor_to_attr=interpolated_tensor, input_ids_tensor=input_tensor,
                                               input_tensor=embed_input_tensor[0],
                                               ref_tensor=embed_ref_tensor[0],
                                               num_steps=num_steps)
        viz_neg = self.run_w_target_n_exaplain(logits, target=torch.zeros(num_steps).long().to(self.model.device),
                                               tensor_to_attr=interpolated_tensor, input_ids_tensor=input_tensor,
                                               input_tensor=embed_input_tensor[0],
                                               ref_tensor=embed_ref_tensor[0],
                                               num_steps=num_steps)
        logging.info('-' * 30 + '<br><br>')
        return viz_pos, viz_neg

    def run_w_target_n_exaplain(self, logits, target, tensor_to_attr, input_ids_tensor, input_tensor, ref_tensor,
                                num_steps, ):
        # Postive label

        loss = self.loss_fct(logits, target)
        loss.backward(retain_graph=True)
        logging.info(f"Loss: {loss.tolist()}")
        raw_grad = tensor_to_attr.grad

        # Approximate the integral using the trapezodal rule
        approx_grad = (raw_grad[:-1] + raw_grad[1:]) / 2
        avg_grad = torch.mean(approx_grad, dim=0)  # input len, hdim
        integrated_gradient = (input_tensor - ref_tensor) * avg_grad  # seq_len, hdim
        extracted_attribution = summarize_attributions(integrated_gradient)
        viz = simple_viz_attribution(self.model.tokenizer, input_ids_tensor, extracted_attribution)
        logging.info(viz)
        return viz


def load_dataset():
    pass


if __name__ == '__main__':
    name_model = 'facebook/bart-base'
    name_tokenizer = name_model
    model_wrap = WrapExplainDec \
        (name_model=name_model, name_tokenizer=name_tokenizer)

    my_ig = IntGrad(model_wrap)
    input_doc = "Integrated Gradients is a technique for attributing a classification model's prediction to its input features. It is a model interpretability technique: you can use it to visualize the relationship between input features and model predictions."
    input_doc_input_ids = model_wrap.tokenizer(text=input_doc, return_tensors='pt')
    length = input_doc_input_ids.size()

    ref_input_ids = [model_wrap.tokenizer.pad_token_id] * length
    ref_input_ids = torch.LongTensor(ref_input_ids)

    input_doc_input_ids = input_doc_input_ids.unsqueeze(0)
    ref_input_ids = ref_input_ids.unsqueeze(0)

    my_ig.get_integrated_gradients(input_doc_input_ids, ref_input_ids)
