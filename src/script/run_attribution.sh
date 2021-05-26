
python produce_prob_output.py -mname_sum='facebook/bart-large-cnn' -data_name='cnndm' -device='cuda:1'


python occ.py -task='occ'
python occ.py -task='occ' -sent_pre_sel -batch_size=100 -device='cuda:3'

python int_grad.py -task='int_grad' -sent_pre_sel -device='cuda:3'
python int_grad.py -task='int_grad'  -device='cuda:1'


python input_grad.py -task='inp_grad' -sent_pre_sel -device='cuda:0'
python input_grad.py -task='inp_grad'  -device='cuda:0'


python attn.py -task='attn'  -device='cuda:1'  -sent_pre_sel 


python output_meta.py -data_name='cnndm'    # extract data and add category to meta.csv => viz.csv
python map.py -data_name='cnndm'    # how to draw map


python output_meta.py -data_name='xsum'    # extract data and add category to meta.csv => viz.csv
python map.py -data_name='xsum'    # how to draw map

python post_eval.py -data_name='cnndm'
