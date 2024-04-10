import argparse
import torch
from transformers import AutoTokenizer
from train import str2bool
import warnings

warnings.filterwarnings("ignore")

def purify(input_text):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default="checkpoints/hybrid/t5/2023-10-11-01_13_21_199591", type=str)
    parser.add_argument('--ckpt', default="")
    parser.add_argument('--gpus', default="0,1")
    parser.add_argument('--model_name', default="t5-robust")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--max_src_len', default=128, type=int)
    parser.add_argument('--mode', choices=["test", "val"], default="test")
    parser.add_argument('--PTM', default="t5", choices=["t5", "pegasus", "codet5", "bart"])
    parser.add_argument('--beam_size', default=8, type=int)
    parser.add_argument('--early_stop', default=True, type=str2bool)
    parser.add_argument('--no_repeat_ngram', default=4, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--min_length', default=5, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--diversity_pen', default=0.0, type=float)
    parser.add_argument('--length_pen', default=1.0, type=float)

    args = parser.parse_args()

    ckpt_path = "./checkpoints/sighan15/bart/2023-11-25-00_04_33_572389/epoch-0_step-3000.pt"

    # 加载预训练的BERT模型和tokenizer
    model_name = 'bart-base-chinese'  # 选择适用于你的任务的预训练模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载训练好的模型参数
    model = torch.load(ckpt_path, map_location=torch.device('cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    # 模型推理
    output = inference(model, tokenizer, input_text, device, args)

    # 输出推理结果
    print("output:", output)

# 推理函数
def inference(model, tokenizer, text, device, args):
    dct = tokenizer.batch_encode_plus(text, max_length=args.max_src_len, return_tensors="pt", truncation=True,
                                      padding=True)
    text_id = dct["input_ids"]
    for p in model.parameters():
        p.requires_grad = False
    cand_id = model.generate(
        input_ids=text_id.to(device),
        attention_mask=dct["attention_mask"].to(device),
        args=args
    )
    dec = tokenizer.decode(cand_id[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return dec

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default="checkpoints/hybrid/t5/2023-10-11-01_13_21_199591", type=str)
    parser.add_argument('--ckpt', default="")
    parser.add_argument('--gpus', default="0,1")
    parser.add_argument('--model_name', default="t5-robust")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--max_src_len', default=128, type=int)
    parser.add_argument('--mode', choices=["test", "val"], default="test")
    parser.add_argument('--PTM', default="t5", choices=["t5", "pegasus", "codet5", "bart"])
    parser.add_argument('--beam_size', default=8, type=int)
    parser.add_argument('--early_stop', default=True, type=str2bool)
    parser.add_argument('--no_repeat_ngram', default=4, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--min_length', default=5, type=int)
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--diversity_pen', default=0.0, type=float)
    parser.add_argument('--length_pen', default=1.0, type=float)

    args = parser.parse_args()

    # 替换为你要进行推理的文本

    input_text = ["Your input text goes here."]
    ckpt_path = "./checkpoints/sighan15/bart/2023-11-25-00_04_33_572389/epoch-0_step-3000.pt"

    # 加载预训练的BERT模型和tokenizer
    model_name = 'bart-base-chinese'  # 选择适用于你的任务的预训练模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载训练好的模型参数
    model = torch.load(ckpt_path, map_location=torch.device('cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    # 模型推理
    output = inference(model, tokenizer, input_text, device)

    # 输出推理结果
    print("output:", output)



