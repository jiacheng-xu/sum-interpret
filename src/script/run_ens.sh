# python src/ensemble.py -dir_save='/mnt/data0/jcxu/interpret_output' 
# python src/proc_ensemble.py -dir_save='/mnt/data0/jcxu/interpret_output'
python set_data.py -dir_save='/mnt/data0/jcxu/meta_data_ref' -data_name='xsum'

python occ.py -sent_pre_sel -task='occ' -eval_mode='sel_tok'

python evaluation.py -task=int_grad -eval_mode=sel_sent 
python evaluation.py -task=int_grad -eval_mode=rm_sent 
python evaluation.py -task=inp_grad -eval_mode=sel_sent 
python evaluation.py -task=random -eval_mode=sel_sent -device=cuda:1 

python evaluation.py -task=lead -eval_mode=sel_sent -device=cuda:1 


python evaluation.py -task=lead -eval_mode=rm_sent -device=cuda:1 
python evaluation.py -task=random -eval_mode=rm_sent -device=cuda:1 
python evaluation.py -task=inp_grad -eval_mode=rm_sent -device=cuda:1

python evaluation.py -task=occ -eval_mode=rm_sent -device=cuda:0


python evaluation.py -task=random -eval_mode=sel_sent

python evaluation.py -task=int_grad -eval_mode=sel_sent
python post_eval.py

python evaluation.py -task=occ -eval_mode=sel_tok -device=cuda:0 -sent_pre_sel
python evaluation.py -task=occ -eval_mode=rm_tok -device=cuda:0

python evaluation.py -task=attn -eval_mode=sel_tok -device=cuda:0 -sent_pre_sel