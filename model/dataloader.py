from fastNLP.io import JsonLoader
from transformers import AutoTokenizer
from preprocess.preprocess import TOKENIED_FILE_DIR

def get_data_path(PTM, dataset):   
    paths = {'train': f'{TOKENIED_FILE_DIR}/{dataset}/train.{PTM}.jsonl', 'val': f'{TOKENIED_FILE_DIR}/{dataset}/val.{PTM}.jsonl'}
    return paths


def process_data(data_bundle, args):
    target = "tgt_id"
    source = "src_id"

    def get_tgt_inp(instance):
        if args.PTM == "bart":
            return [args.sep_id] + instance[target][:-1]
        else:
            return [args.pad_id] + instance[target][:-1]

    def get_tgt_outp(instance):
        return instance[target][:-1] + [args.pad_id]

    def truncate_tgt_id(instance, max_len):
        if args.PTM == "bart":
            if len(instance[target]) > max_len:
                return instance[target] + [args.sep_id]
            else:
                return instance[target] + [args.sep_id]
        else:
            if len(instance[target]) > max_len:
                return instance[target] + [args.eos_id]
            else:
                return instance[target] + [args.eos_id]

    def truncate_src_id(instance, max_len):
        if len(instance[source]) > max_len:
            if args.PTM == "bart":
                return instance[source][:max_len - 1] + [args.sep_id] + [args.pad_id]
            else:
                return instance[source][:max_len - 1] + [args.eos_id]
        else:
            return instance[source] + [args.pad_id]

    data_bundle.apply(lambda ins: truncate_tgt_id(ins, args.max_tgt_len), new_field_name=target)
    data_bundle.apply(lambda ins: get_tgt_inp(ins), new_field_name='target_inp')
    data_bundle.apply(lambda ins: get_tgt_outp(ins), new_field_name='target_outp')
    data_bundle.apply(lambda ins: truncate_src_id(ins, args.max_src_len), new_field_name='src_inp')
    data_bundle.apply(lambda ins: len(ins["target_outp"]), new_field_name="seq_len")

    return data_bundle


def get_data_bundle(args):

    data_paths = get_data_path(args.PTM, args.dataset)
    init_bundle = JsonLoader().load(data_paths)
    # load the datasets
    if args.PTM == "codet5":
        tokenizer_name = "Salesforce/codet5-base"
    elif args.PTM == "t5":
        if "mengzi" in args.model_name:
            tokenizer_name = "t5-mengzi"
        else:
            tokenizer_name = "t5-small-chinese"
    elif args.PTM == "pegasus":
        tokenizer_name = "google/pegasus-xsum"
    elif args.PTM == "bart":
        tokenizer_name = "bart-base-chinese"
    else:
        raise NotImplementedError("--- please add this tokenizer here ---  ")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    args.pad_id = tokenizer.pad_token_id
    args.eos_id = tokenizer.eos_token_id
    args.bos_id = tokenizer.bos_token_id

    if args.PTM == "bart":
        args.sep_id = tokenizer.sep_token_id
        args.ignore_id = -100
    data_bundle = process_data(init_bundle, args)
    return data_bundle
