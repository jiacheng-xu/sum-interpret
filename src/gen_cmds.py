
# python evaluation.py -task=inp_grad -eval_mode=sel_sent
# python evaluation.py -task=inp_grad -eval_mode=rm_sent

# python evaluation.py -task=int_grad -eval_mode=sel_sent -device='cuda:1'
# python evaluation.py -task=int_grad -eval_mode=rm_sent -device='cuda:1'
import random

for task in ['int_grad', 'inp_grad', 'random', 'lead', 'occ']:
    for setting in ['sel_tok', 'rm_tok', 'sel_sent', 'rm_sent']:
        gpuid = random.randint(0,3)
        print(
            f"python evaluation.py -task={task} -eval_mode={setting} -device=cuda:{gpuid}")
    print()