
# python evaluation.py -task=inp_grad -eval_mode=sel_sent
# python evaluation.py -task=inp_grad -eval_mode=rm_sent

# python evaluation.py -task=int_grad -eval_mode=sel_sent -device='cuda:1'
# python evaluation.py -task=int_grad -eval_mode=rm_sent -device='cuda:1'
import random
import random
tasks = ['int_grad', 'inp_grad', 'random', 'lead', 'occ', 'attn']
random.shuffle(tasks)
# settings = ['rm_sent'] 
settings = ['sel_tok', 'rm_tok', 'sel_sent', 'rm_sent'] 
for task in tasks:
    for setting in settings:
        gpuid = random.randint(0, 3)
        print(f"python evaluation.py -task={task}  -eval_mode={setting} -device=cuda:{gpuid}")
        print(f"python post_eval.py -task={task} -eval_mode={setting}")
        # print()
    print()


for task in ['int_grad', 'inp_grad', 'occ', 'attn']:
    for setting in ['sel_tok', ]:
        gpuid = random.randint(0, 3)
        print(f"cd /home/jcxu/sum-interpret/src; python evaluation.py -task={task} -sent_pre_sel -eval_mode={setting} -device=cuda:{gpuid}")
        print(f"python post_eval.py -task={task} -sent_pre_sel -eval_mode={setting}")
        print()
    print()
