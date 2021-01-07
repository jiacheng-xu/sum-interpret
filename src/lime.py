from helper_run_bart import write_pkl_to_disk
from main import init_bart_family
from transformers.modeling_outputs import BaseModelOutput
from helper_run_bart import init_bart_sum_model
from util import *
from captum.attr import LayerIntegratedGradients, TokenReferenceBase
import torch
from captum.attr._utils.visualization import format_word_importances


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-device", help="device to use", default='cuda:0')
    parser.add_argument("-data_name", default='xsum', help='name of dataset')
    parser.add_argument("-mname_lm", default='facebook/bart-large')
    parser.add_argument("-mname_sum", default='facebook/bart-large-xsum')
    parser.add_argument("-batch_size", default=40)
    parser.add_argument('-max_samples', default=1000)
    parser.add_argument('-num_run_cut', default=50)
    parser.add_argument('-truncate_sent', default=15,
                        help='the max sent used for perturbation')
    parser.add_argument('-dir_read', default='/mnt/data0/jcxu/meta_data_ref',
                        help="Path of the meta data to read.")
    parser.add_argument('-dir_save', default="/mnt/data0/jcxu/output_ig",
                        help="The location to save output data. ")
    args = parser.parse_args()
    logger.info(args)
    if not os.path.exists(args.dir_save):
        os.makedirs(args.dir_save)
    device = args.device

    model_lm, model_sum, model_sum_ood, tokenizer = init_bart_family(
        args.mname_lm, args.mname_sum, device, no_lm=True, no_ood=True)
    logger.info("Done loading BARTs.")
    model_pkg = {'sum': model_sum, 'tok': tokenizer}
    all_files = os.listdir(args.dir_read)
    for f in all_files:
        outputs = []
        step_data, meta_data = read_meta_data(args.dir_read, f)
        uid = meta_data['id']
        for step in step_data:
            ig_enc_result = new_step_int_grad(meta_data['doc_token_ids'], actual_word_id=step['tgt_token_id'], prefix_token_ids=step['prefix_token_ids'],
                                              num_run_cut=args.num_run_cut, model_pkg=model_pkg, device=device)
            ig_enc_result = ig_enc_result.squeeze(0).cpu().detach()
            outputs.append(ig_enc_result)
        skinny_meta = {
            'doc_token_ids': meta_data['doc_token_ids'].squeeze(),
            'output': outputs
        }
        write_pkl_to_disk(args.dir_save, uid, skinny_meta)
        print(f"Done {uid}.pkl")

    """
    interest = "Google"
    summary_prefix = "She didn't know her kidnapper but he was using"
    document = 'Google Maps is a web mapping service developed by Google. It offers satellite imagery, aerial photography, street maps, 360 interactive panoramic views of streets, real-time traffic conditions, and route planning for traveling by foot, car, bicycle, air and public transportation.'
    document_sents = ['Google Maps is a web mapping service developed by Google. ',
                      'It offers satellite imagery, aerial photography, street maps, 360 interactive panoramic views of streets, real-time traffic conditions, and route planning for traveling by foot, car, bicycle, air and public transportation.']
    mname = 'facebook/bart-large-xsum'
    device = 'cpu'
    model_sum, bart_tokenizer = init_bart_sum_model(mname, device)
    model_pkg = {'lm': None, 'sum': model_sum, 'ood': None, 'tok': bart_tokenizer,
                 'spacy': None}
    actual_word_id = tokenizer.encode(" "+interest)[1]
    step_int_grad(interest, actual_word_id, summary_prefix=[summary_prefix], document=[
        document], document_sents=document_sents, num_run_cut=51, model_pkg=model_pkg, device=device)
    """