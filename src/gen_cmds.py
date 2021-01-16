
# python evaluation.py -task=inp_grad -eval_mode=sel_sent
# python evaluation.py -task=inp_grad -eval_mode=rm_sent

# python evaluation.py -task=int_grad -eval_mode=sel_sent -device='cuda:1'
# python evaluation.py -task=int_grad -eval_mode=rm_sent -device='cuda:1'

for task in ['int_grad','inp_grad','random','lead']:
    for setting in ['sel_tok','rm_tok']:
    # for setting in ['sel_sent','rm_sent']:

        print(f"python evaluation.py -task={task} -eval_mode={setting} -device=cuda:1")