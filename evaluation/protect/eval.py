import glob
import json
import os
import time
from multiprocessing import Pool

import shutil
import sys
import codecs
import nltk
import logging
from pyrouge import Rouge155
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import argparse

# import datasets

from nltk.tokenize import word_tokenize


def load_jsonl(path):
    inst_list = []
    ref_list = []
    source_list = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            if 'sys' in path:
                sys_temp = json.loads(line)['sys_out'] + '\t' + json.loads(line)['source']
                inst_list.append(sys_temp)
            else:
                ref_list.append(json.loads(line)['ref_out'])
    if 'sys' in path:
        return inst_list
    else:
        return ref_list
    


def eval_by_model_batch(predict, dataset, verbose=True):
    print(predict[111])
    print(dataset[111])
    import unicodedata
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    pos_num = 0
    neg_num = 0
    total_num = 0
    import time
    start_time = time.time()
    srcs = []
    tgts = []
    for line in predict:
        line = line.strip()
        # 转换中文符号
        line = unicodedata.normalize('NFKC', line)
        parts = line.split('\t')
        if len(parts) != 2:
            src = parts
            srcs.append(src)
            tgts.append(src)
        else:
            src = parts[1].lower()
            tgt = parts[0].lower()
            srcs.append(src)
            tgts.append(tgt)

    res = dataset
    id = 0
    for each_res, src, tgt in zip(res, srcs, tgts):
        # each_res = each_res.split('\t')
        if len(each_res) == 2:
            tgt_pred, pred_detail = each_res
        else:
            tgt_pred = each_res.lower()
        if verbose:
            print()
            print('id     :', id)
            id += 1
            print('input  :', src)
            print('truth  :', tgt)
            print('predict:', each_res)

        # 负样本
        if src == tgt:
            neg_num += 1
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
                if verbose:
                    print('neg right')
            # 预测为正
            else:
                FP += 1
                if verbose:
                    print('neg wrong')
        # 正样本
        else:
            pos_num += 1
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
                if verbose:
                    print('pos right')
            # 预测为负
            else:
                FN += 1
                if verbose:
                    print('pos wrong')
        total_num += 1

    spend_time = time.time() - start_time
    print(total_num)
    print(TP)
    print(TN)
    acc = (TP + TN) / total_num
    precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    if args.wandb:
        wandb.log({'acc':acc, 'precision':precision, 'recall':recall, 'f1':f1})
    print(
        f'Sentence Level: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, '
        f'cost time:{spend_time:.2f} s, total num: {total_num}, pos num: {pos_num}, neg num: {neg_num}')
    return acc, precision, recall, f1

def evaluate(prompt_model, dataloader, verbose=False):
        generated_sentence = []
        groundtruth_sentence = []
        prompt_model.eval()

        for step, inputs in tqdm(enumerate(dataloader)):
            if use_cuda:
                inputs = inputs.cuda()
            _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
            
            generated_sentence.extend(output_sentence)
            # groundtruth_sentence.extend(inputs['tgt_text'])
            # print(output_sentence)
            # print(inputs['tgt_text'])
        
        acc, precision, recall, f1 =  eval_by_model_batch(generated_sentence, dataset=validation_data, verbose=verbose)
        # if args.wandb:
        #     wandb.log({"acc": acc})
        print("dev_acc {}, dev_precision {} dev_recall: {} dev_f1: {}".format(acc, precision, recall, f1), flush=True)
        return generated_sentence, acc, precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sys_file', default=None, type=str)
    parser.add_argument('--sys_path', default="/media/HD0/CoNT/results/hybrid/t5/2023-10-13-22_45_52_984514/", type=str)
    parser.add_argument('--wandb', default=False, type=bool)
    args = parser.parse_args()
    if args.sys_path is not None:
        candidate_files = glob.glob(os.path.join(args.sys_path, "*.sys"))
        for candidate_file in candidate_files:
            sys_files = os.path.join(args.sys_path, candidate_file)
    else:
        candidate_files = [args.sys_file]
    for cand_file in candidate_files:
        ref_path = cand_file.replace('sys', 'ref')
        sys_path = cand_file
        print("evaluate: ", sys_path)
        sys_outputs = load_jsonl(sys_path)
        ref_outputs = load_jsonl(ref_path)
        eval_by_model_batch(sys_outputs, ref_outputs)






