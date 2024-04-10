# Self-purify
 Self-Purify: Generative Text Purification for Robust Defence

## Dependencies

Main libraries

- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.7 +
- [transformers](https://github.com/huggingface/transformers) 4.21.0
- [fastNLP](https://github.com/fastnlp/fastNLP) 1.0.0beta

```
pip install transformers==4.21.0
pip install fastNLP==1.0.0beta
```


All code only supports running on Linux.


### Datasets

we have prepared the raw dataset to help you reproduce the results in our paper.  Datasets provided by this repo can  **only**  be used for *Reproduction* and *Research Purposes*.
All files are in `jsonl` format where each line is a `json` sample:

```
{"source": "the input text (for encoder)", "target": "the target text (for decoder)"}}
```

You can Download the jsonl files through these links.

1. Summarization：
   - [XSum](https://drive.google.com/file/d/1t--UZo4Pnv4HjGhAfun5vDz3JCoqIggq/view?usp=sharing)
   - [Multi-News](https://drive.google.com/file/d/16VdfzvLmmOrYsayujA-Hu4d3i_ejHTln/view?usp=sharing)
2. Translation：
   - [WMT'16 Ro-En](https://drive.google.com/file/d/1rGoylmZvIhNvsoPZda7OZP_0nYUfUpoq/view?usp=sharing)
   - [WMT'14 En-De and IWSLT'14 De-En](https://github.com/ChenxinAn-fdu/CoNT)
3. Code comment Generation
   - [java](https://drive.google.com/file/d/1PBdxKvMTvfCzseactMRffTUwuTI7oAGz/view?usp=sharing)
   - [python](https://drive.google.com/file/d/189xlRW3r3UuMTko73zURfJ3I_LXQ026D/view?usp=sharing)
4. Data-to-text generation  
   - [Wikibio](https://drive.google.com/file/d/1i0BZykxifH2hEdCyB_nZFvs2PT4UdUFJ/view?usp=sharing)
   - [ToTTo](https://drive.google.com/file/d/1nOlhGKpTWPCmAwmEI_gdALkAXlMn2Tbk/view?usp=sharing) (Blind test set)
5. Commonsense generation  
   - [CommonGen](https://drive.google.com/file/d/1UvCBenGMzdQyR25ka_1vmaPwGVFQzqvS/view?usp=sharing) (Blind test set)

Before loading the training set, please pre-tokenize these files  with the following command:

```
mkdir jsonl_files
mkdir tokenized_files
mv /download/path/xsum.zip  ./jsonl_files
cd jsonl_files
unzip xsum.zip && cd ..
python preprocess/preprocess.py --model_name  t5-small --dataset xsum
python preprocess/preprocess.py --model_name  bart-lang8-hsk --dataset zh-hsk
```

This command will produce the tokenized files of XSum `tokenized_files/train.t5.jsonl, tokenized_files/val.t5.jsonl` with the tokenizer of t5-small  

### Training

We have provided the training scripts for each dataset we used in this paper, and you can easily start the training process with them:

```
#If there is no warmed-up checkpoint, you should use `--warmup True` to train the generation model with NLLLoss 
python run_xsum.py --mode train --gpus 0,1,2,3 --warmup True --model_name t5-small (or google/pegasus-large)

python run_sighan15.py --mode train --gpus 0,1 --warmup False --model_name bart-base-chinese
python run_zh-hsk.py --mode train --gpus 0,1 --warmup False --model_name bart-lang-chinese

```

the warmed-up checkpoint will be saved to `./pretrained_weigths/xsum/t5(or pegasus)` by default.  
Please notice that huggingface also provides many finetuned checkpoints. So that if the `--model_name`  contains the dataset name  (like `google/pegasus-xsum`), we will skip the warmup.

```
#If you already have a warmed-up checkpoint, you can skip the warm-up stage
python run_xsum.py --mode train --gpus 0,1,2,3 --warmup False
```

After completing the training process,  several best checkpoints will be stored in a folder named after the training start time by default, for example, `checkpoints/xsum/t5/2022-10-05-10-37-24-196200`

### Generation

We suggest first selecting the best checkpoint based on the dev set with `--mode val` and then generating the results on the test set with the best checkpoint. 

You can run the following command to generate the results on test/dev set with all checkpoints in a given folder, e.g., `checkpoints/xsum/t5/2022-10-05-10-37-24-196200/`:

```
python run_xsum.py --mode val (or test) --model_name t5-small --save_path checkpoints/xsum/t5/2022-10-05-10-37-24-196200/ --gpus 0,1,2,3

python run_sighan15.py --mode val --model_name bart-base-chinese --save_path checkpoints/sighan15/bart/2023-11-19-00_36_24_902827 --gpus 0,1

python run_csc.py --mode val --model_name t5-mengzi --save_path checkpoints/csc/t5/2023-11-30-11_58_40_517240/ --gpus 0,1
```

This will produce the generated results in the floder: `results/xsum/t5/2022-10-05-10-37-24-196200/` containing serval system output and ground truth files: `epoch-2_step-8000.val.sys` , `epoch-2_step-8000.val.ref`, `epoch-2_step-10000.val.sys` , `epoch-2_step-10000.val.ref`


To generate the results for test set with  **a specified checkpoint**, you can use the `--ckpt`  parameter and remember to change the mode to `test`:

```
python run_xsum.py --mode test --model_name t5-small --save_path checkpoints/xsum/t5/2022-10-05-10-37-24-196200/ \
--ckpt epoch-2_step-8000.pt --gpus 0,1,2,3
```

This will produce the generated results in the floder `results/xsum/t5/2022-10-05-10-37-24-196200/`  containing `epoch-2_step-8000.test.sys` , `epoch-2_step-8000.test.ref`

### Evaluation

We have proveded the evaluation scripts for each dataset: `evaluation/$dataset/eval.py` with which you can easily get the evaluation results.

This is an example to evaluate all the generated results for `xsum` in the folder `results/xsum/t5/2022-10-05-10-37-24-196200/`:

```
python evaluation/xsum/eval.py --sys_path results/xsum/t5/2022-10-05-10-37-24-196200/
python evaluation/correct/eval.py --sys_path results/sighan15/bart/2023-11-20-09_24_14_968443/
```

If you only want to evaluate a specified file：

```
python evaluation/xsum/eval.py --sys_file results/xsum/t5/2022-10-05-10-37-24-196200/epoch-2_step-8000.sys
```

### Another Example: ToTTo

The first step is downing `totto_meta.zip` via [this link](https://drive.google.com/file/d/1nOlhGKpTWPCmAwmEI_gdALkAXlMn2Tbk/view?usp=sharing) and moving the unziped files to `jsonl_files`. 

```
# preprocess
python preprocess/preprocess.py --model_name  t5-base --dataset totto_meta

# training with warm-up
python run_totto.py --mode train --gpus 0,1,2,3 --warmup True --model_name t5-base  

# generation
python run_totto.py --mode val --model_name t5-base --save_path checkpoints/totto_meta/t5/training_time/ --gpus 0,1,2,3

# evaluation
pip install absl-py (if you do not have the `absl` library)
python evaluation/totto/eval.py --sys_path results/totto_meta/t5/training_time/
```

Results of CoNT-T5-base on *DEV set* of ToTTo are usually `BLEU=48.8~49.2` and `PARENT=58.2~58.7`.  Results of T5-base are `BLEU=47.7` and `PARENT=57.1`. Performance of other scales of T5 can be found in this [paper](https://arxiv.org/pdf/2005.10433.pdf).   



### Details about evaluation scripts and preprocess

- Summarization：[pyrouge library](https://github.com/bheinzerling/pyrouge) which is the most widely-used library to evaluate summarization systems. Instructions for Installation can be found in this [repo](https://github.com/ChenxinAn-fdu/CGSum)
- WMT16_Ro-En: we follow the evaluation scripts of [previous work CLAPS](https://github.com/seanie12/CLAPS)
- Java and Python: the evaluation codes are taken from [CodeT5](https://github.com/salesforce/CodeT5).
- WikiBio: we use evaluation scripts provided by this work.  We preprocess the dataset following the instruction in this [repo](https://github.com/tyliupku/wiki2bio).
  Example:

```
{"source": "caption[jens in 1962], name[salome jens], birth date[8 may 1935], birth place[milwaukee , wisconsin , u.s.], occupation[actress], years active[1956 -- present], article title[salome jens]", 
"target": "salome jens -lrb- born may 8 , 1935 -rrb- is an american stage , film and television actress ."}
```

- ToTTo: the evaluation codes are taken from their [official repo](https://github.com/google-research-datasets/ToTTo). We preprocess the dataset following the instruction in this [repo](https://github.com/google-research/language/tree/master/language/totto).
  Example: 

```
{"source": "<page_title> List of Speakers of the Minnesota House of Representatives </page_title> <section_title> State </section_title> <table> <cell> Ralph J. Parker <col_header> Speaker </col_header> </cell> </table>",
 "target": "Ralph J. Parker was a Minnesota politician and a Speaker of the Minnesota House of Representatives. Ralph J. Parker was a Speaker of the Minnesota House of Representatives. Ralph J. Parke, was a Minnesota politician, and a Speaker of the Minnesota House of Representatives."}
```

To get the results on the test set please submit your results [here](https://docs.google.com/forms/d/e/1FAIpQLScjGJr9z6_DljrYN8ySi1-zdHk8DL4udEmBHU6IsfoLvuDBZA/viewform?usp=send_form).

- CommonGen: we only evaluate ROUGE score on dev set and we get the reported results in our paper with the help of the [authors of CommonGen](https://inklab.usc.edu/CommonGen/).

**Thanks for their work**.