import json
from tqdm import tqdm


def trans(path, output):
    with open(path,'r',encoding='utf-8')as f:                       #打开txt文件
        for line in tqdm(f):                                        
            d = {}
            d['source'] = line.split('\t')[0].strip()               #line表示txt文件中的一行，将每行后面的换行符去掉，并将其作为字典d的content键的值
            d['target'] = line.split('\t')[1].strip()
            with open(output,'a',encoding='utf-8')as file:          #创建一个jsonl文件，mode设置为'a'
                json.dump(d,file,ensure_ascii=False)                #将字典d写入json文件中，并设置ensure_ascii = False,主要是因为汉字是ascii字符码,若不指定该参数，那么文字格式就会是ascii码
                file.write('\n')


if __name__ == '__main__':
    files = ["train", "test"]
    for file in files:
        # path = '/media/HD0/CoNT/RobustT5_data/MCTC_abb/'+ file + '_pinyin_abb.txt'
        # output = '/media/HD0/CoNT/jsonl_files/abb/'+ file + '.jsonl'
        path = '/media/HD0/CoNT/data/roc/'+ file + '.txt'
        output = '/media/HD0/CoNT/jsonl_files/roc/'+ file + '.jsonl'
        trans(path, output)