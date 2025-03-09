import onnxruntime
import numpy as np
from pytorch_pretrained import BertTokenizer

session = onnxruntime.InferenceSession("../workspace/classifier.onnx",
                                       providers=['CPUExecutionProvider'])

tokenizer = BertTokenizer.from_pretrained("bert_pretrain/vocab.txt")

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
def padding(content, pad_size=32):
    token = tokenizer.tokenize(content)
    token = [CLS] + token
    # seq_len = len(token)
    mask = []
    token_ids = tokenizer.convert_tokens_to_ids(token)

    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            # seq_len = pad_size
    return token_ids, mask

class_label = [
    "金融",   # finance
    "房地产", # realty
    "股票",   # stocks
    "教育",   # education
    "科学",   # science
    "社会",   # society
    "政治",   # politics
    "体育",   # sports
    "游戏",   # game
    "娱乐"    # entertainment
]

if __name__ == "__main__":
    text = "世界足球杯在巴西正常举行。"
    input_ids, attention_mask = [np.array(item, dtype=np.int64).reshape((1,-1)) for item in padding(text)]
    pred = session.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})[0]
    print("label: ", class_label[np.argmax(pred)], ", prob: ", pred[0][np.argmax(pred)])