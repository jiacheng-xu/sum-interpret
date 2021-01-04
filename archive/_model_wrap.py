import torch
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelWithLMHead

import logging


class WrapExplain():
    def __init__(self, device, name_tokenizer, name_model):
        self.tokenizer = AutoTokenizer.from_pretrained(name_tokenizer)
        self.model = AutoModelForPreTraining.from_pretrained(name_model)

        self.device = device
        self.model.to(device)

        self.mname = name_model
        try:
            if 'distilbert' in name_model:
                self.lm_model = AutoModelWithLMHead.from_pretrained(
                    'distilbert-base-uncased', return_dict=True)
            elif 'bert' in name_model:
                self.lm_model = AutoModelWithLMHead.from_pretrained(
                    'bert-base-uncased', return_dict=True)
            self.lm_model.to(device)
        except:
            print('Model does not support LM')

    def load_input_embed_from_input_ids(self, input_ids):
        # Wrap different encoder model
        pass


class WrapExplainEnc(WrapExplain):
    def __init__(self, name_tokenizer, name_model, device):
        super().__init__(device, name_tokenizer, name_model)

    def fill_pad_ref_baseline(self, batch, seq_len):
        ref = [self.tokenizer.cls_token_id] + [self.tokenizer.pad_token_id] * (seq_len - 2) + [
            self.tokenizer.sep_token_id]
        # TODO  need to pad if things are getting longer
        rt_pads = torch.tensor([ref for _ in range(batch)],
                               dtype=torch.long, device=self.device)
        return rt_pads

    def fill_ref_baseline_lm_mask_filling(self, input_tensor, batch_sz, seq_len):
        # size of input_tensor: 1, seqlen
        input_token_list = input_tensor.tolist()[0]
        repeated_input = input_tensor.repeat(seq_len, 1).to(self.device)

        repeated_input[range(seq_len), range(
            seq_len)] = self.tokenizer.mask_token_id
        lm_output_logits = self.lm_model(repeated_input).logits
        selected_logits = lm_output_logits[range(seq_len), range(seq_len), :]
        lm_topk_result = torch.topk(selected_logits, 100, dim=-1)
        indicies = lm_topk_result.indices  # [int] seqleno, topk
        indicies_list = indicies.tolist()

        for t in range(seq_len):
            original_tok_id = input_token_list[t]
            original_tok = self.tokenizer.decode([original_tok_id])
            mask_fill_token_ids = indicies_list[t]
            mask_fill_tokens = [self.tokenizer.decode(
                [x]) for x in mask_fill_token_ids]
            logging.info('Time: {:2d} {:_<10}{:_>10}{:_>10}{:_>10}'.format(t, original_tok,mask_fill_tokens[0],
                                                                           mask_fill_tokens[1],
                                                                           mask_fill_tokens[2]))
        ref = indicies[:, 99].tolist()
        logging.info("New Sentence: {}".format(self.tokenizer.decode(ref)))
        rt_pads = torch.tensor(
            [ref for _ in range(batch_sz)], dtype=torch.long, device=self.device)
        return rt_pads


class WrapDistillBERT(WrapExplainEnc):
    def __init__(self, name_tokenizer, name_model, device):
        super().__init__(name_tokenizer, name_model, device)
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            name_tokenizer, truncation=True, max_length=100)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            name_model, return_dict=True)

        self.model.to(device)

    def get_full_embeddings(self, input_ids):
        embed_input_ids = self.model.distilbert.embeddings.forward(input_ids)
        input_shape = embed_input_ids.size()[:-1]
        seq_length = input_shape[1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.model.distilbert.embeddings.position_embeddings(
            position_ids)

        embeddings = embed_input_ids + position_embeddings
        embeddings = self.model.distilbert.embeddings.LayerNorm(embeddings)
        embeddings = self.model.distilbert.embeddings.dropout(embeddings)
        return embeddings

    def forward(self, inputs_embeds, **kwargs):
        model_output = self.model.forward(
            inputs_embeds=inputs_embeds, **kwargs)

        logits = model_output['logits']  # batch size, class num = 2
        return logits


class WrapBERT(WrapExplainEnc):
    def __init__(self, name_tokenizer, name_model, device):
        super().__init__(name_tokenizer, name_model, device)
        from transformers import BertTokenizer, BertForSequenceClassification
        self.tokenizer = BertTokenizer.from_pretrained(
            name_tokenizer, truncation=True, max_length=30)
        self.model = BertForSequenceClassification.from_pretrained(
            name_model, return_dict=True)
        self.model.to(device)

    def get_full_embeddings(self, input_ids):
        embeddings = self.model.bert.embeddings.forward(input_ids=input_ids)
        return embeddings

    def forward(self, inputs_embeds, **kwargs):
        model_output = self.model.forward(
            inputs_embeds=inputs_embeds, **kwargs)

        logits = model_output['logits']  # batch size, class num = 2
        return logits


class WrapExplainDec(WrapExplain):
    def __init__(self, name_tokenizer, name_model):
        pass
