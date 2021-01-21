
# python evaluation.py -task=inp_grad -eval_mode=sel_sent
# python evaluation.py -task=inp_grad -eval_mode=rm_sent

# python evaluation.py -task=int_grad -eval_mode=sel_sent -device='cuda:1'
# python evaluation.py -task=int_grad -eval_mode=rm_sent -device='cuda:1'
import random

for task in ['int_grad', 'inp_grad', 'random', 'lead', 'occ']:
    for setting in ['sel_tok', 'rm_tok', 'sel_sent', 'rm_sent']:
        gpuid = random.randint(0,0)
        # print(f"python evaluation.py -task={task}  -eval_mode={setting} -device=cuda:{gpuid}")
        print(f"python post_eval.py -task={task} -eval_mode={setting}")
        print()
    print()
"""

for task in ['int_grad', 'inp_grad', 'occ']:
    for setting in ['sel_tok', 'rm_tok']:
        gpuid = random.randint(1,1)
        print(f"python evaluation.py -task={task} -sent_pre_sel -eval_mode={setting} -device=cuda:{gpuid}")
        print(f"python post_eval.py -task={task} -eval_mode={setting}")
        print()
    print()
"""

"""

python evaluation.py -task=int_grad -device=cuda:0
python post_eval.py -task=int_grad -eval_mode=sel_tok

python evaluation.py -task=int_grad -device=cuda:0
python post_eval.py -task=int_grad -eval_mode=rm_tok

python evaluation.py -task=int_grad -device=cuda:0
python post_eval.py -task=int_grad -eval_mode=sel_sent

python evaluation.py -task=int_grad -device=cuda:0
python post_eval.py -task=int_grad -eval_mode=rm_sent


python evaluation.py -task=inp_grad -device=cuda:0
python post_eval.py -task=inp_grad -eval_mode=sel_tok

python evaluation.py -task=inp_grad -device=cuda:0
python post_eval.py -task=inp_grad -eval_mode=rm_tok

python evaluation.py -task=inp_grad -device=cuda:0
python post_eval.py -task=inp_grad -eval_mode=sel_sent

python evaluation.py -task=inp_grad -device=cuda:0
python post_eval.py -task=inp_grad -eval_mode=rm_sent


python evaluation.py -task=random -device=cuda:0
python post_eval.py -task=random -eval_mode=sel_tok

python evaluation.py -task=random -device=cuda:0
python post_eval.py -task=random -eval_mode=rm_tok

python evaluation.py -task=random -device=cuda:0
python post_eval.py -task=random -eval_mode=sel_sent

python evaluation.py -task=random -device=cuda:0
python post_eval.py -task=random -eval_mode=rm_sent


python evaluation.py -task=lead -device=cuda:0
python post_eval.py -task=lead -eval_mode=sel_tok

python evaluation.py -task=lead -device=cuda:0
python post_eval.py -task=lead -eval_mode=rm_tok

python evaluation.py -task=lead -device=cuda:0
python post_eval.py -task=lead -eval_mode=sel_sent

python evaluation.py -task=lead -device=cuda:0
python post_eval.py -task=lead -eval_mode=rm_sent


python evaluation.py -task=occ -device=cuda:0
python post_eval.py -task=occ -eval_mode=sel_tok

python evaluation.py -task=occ -device=cuda:0
python post_eval.py -task=occ -eval_mode=rm_tok

python evaluation.py -task=occ -device=cuda:0
python post_eval.py -task=occ -eval_mode=sel_sent

python evaluation.py -task=occ -device=cuda:0
python post_eval.py -task=occ -eval_mode=rm_sent

"""