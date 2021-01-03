
from transformers.models.bart.modeling_bart import (
    BartDecoder,
    BartEncoder)


def bart_decoder_forward_embed(input_ids, embed_tokens, embed_scale):
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    inputs_embeds = embed_tokens(input_ids) * embed_scale
    return inputs_embeds



if __name__ == "__main__":
    print("Pass")
