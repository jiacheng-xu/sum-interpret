
from helper_run_bart import run_full_model_slim, init_bart_family
import time
import datetime
from util import *
from helper_run_bart import run_full_model_slim, init_bart_family
from eval_util import *

if __name__ == "__main__":
    debug = True
    parser = common_args()
    args = parser.parse_args()
    args = fix_args(args)
    logger.info(args)
    
    if debug:
        args.device = 'cuda:3'
        # args.mname_sum = 'sshleifer/distilbart-xsum-6-6'
    device = args.device
    # # init BART models

    _, model_sum, _, tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device, no_lm=True, no_ood=True)
    logger.info("Done loading BARTs.")
    model_pkg = {'sum': model_sum, 'tok': tokenizer}

    while True:
        x = input("name")
        input_ids = torch.LongTensor(tokenizer.encode(x)).unsqueeze(0)
        dec_inp = torch.LongTensor(tokenizer.encode(' David')[:-1]).unsqueeze(0)
        out = run_full_model_slim(model=model_pkg['sum'],input_ids=input_ids, decoder_input_ids=dec_inp,device=device)
        print(out[0])