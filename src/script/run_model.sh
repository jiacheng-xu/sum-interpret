
python occ.py -task='occ'
python occ.py -task='occ' -sent_pre_sel -batch_size=100 -device='cuda:3'

python int_grad.py -task='int_grad' -sent_pre_sel -device='cuda:3'
python int_grad.py -task='int_grad'  -device='cuda:1'


python input_grad.py -task='inp_grad' -sent_pre_sel -device='cuda:0'
python input_grad.py -task='inp_grad'  -device='cuda:0'


python evaluation.py -task=random -eval_mode=sel_sent

python evaluation.py -task=int_grad -eval_mode=sel_sent
python post_eval.py