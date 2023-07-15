import os
import  pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer, AutoProcessor
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class Image_Text_Dataset(Dataset):
    def __init__(self, df, image_path, clipimageprocessor,tokeizer):

        self.clip_processor = clipimageprocessor
        self.tokeizer = tokeizer
        self.image_path = image_path
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.image_path, self.df.iloc[idx]['filename'])
        image = Image.open(path)
        text = self.df.iloc[idx]['text']
        image_input = {'pixel_values': None}
        try:
            image_input = self.clip_processor(images=image, return_tensors="pt")
            if image_input['pixel_values'] is None:
                print(path)
                with open('error_1.txt', 'a') as f:
                    f.write(path+'\n')
        except:
            with open('error_1.txt', 'a') as f:
                f.write(path + '\n')
            print(path)
        text_input = self.tokeizer.encode_plus(text, return_tensors="pt")



        return {
            "image": image_input['pixel_values'],
            "input_ids": text_input["input_ids"],
            "attention_mask": text_input["attention_mask"],
        }

def collate_fn(batch):
    # batch是一个列表，其中每个元素都是数据集返回的字典
    # 首先，我们将input_ids、attention_mask和图片分开
    input_ids = [item['input_ids'][0] for item in batch]
    attention_mask = [item['attention_mask'][0] for item in batch]
    images = [item['image'] for item in batch]

    # 对input_ids和attention_mask进行填充操作，使得它们的长度一致
    # 假设input_ids和attention_mask已经被转换为了张量
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # 对图片进行堆叠操作，使得它们可以形成一个批次
    # 假设图片已经被转换为了张量，并且所有图片的形状都是一样的
    try:
        images = torch.stack(images, dim=0)
        images = torch.squeeze(images, dim=1)
    except Exception  as  e:
        print(e)


    # 返回一个新的字典，其中包含了处理后的input_ids、attention_mask和图片
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'image': images}


def get_dataloader(batch_size=32):
    df = pd.read_csv('../CLIP-Chinese-master/data/train-filter.csv')
    path = '../CLIP-Chinese-master/data/images'

    processor = AutoProcessor.from_pretrained("../vit_clip_data",image_mean=[0.485, 0.456, 0.406],  image_std=[0.229, 0.224, 0.225])
    tokenizer = AutoTokenizer.from_pretrained("../roberta_data")

    dataset = Image_Text_Dataset(df, path, processor, tokenizer)  # 你的数据集
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,shuffle=True, num_workers=8)
    return dataloader
from tqdm import tqdm

if __name__ == '__main__':

    df = pd.read_csv('../CLIP-Chinese-master/data/train-filter.csv')
    # 删除df的train-4462.jpg的图片
    df = df[df['filename'] != 'train-112940.jpg']
    with open('./error.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            print(line)
            name = line.split('/')[-1]
            print(name)
            # 删除df的train-4462.jpg 这一行



            df = df[df['filename'] != name]


    df.to_csv('../CLIP-Chinese-master/data/train-filter.csv', index=False)
    path = '../CLIP-Chinese-master/data/images'
    clip_processor = CLIPImageProcessor.from_pretrained("../vit_clip_data",image_mean=[0.485, 0.456, 0.406],  # 这是在ImageNet上预训练的模型通常使用的值
    image_std=[0.229, 0.224, 0.225])
    tokenizer = AutoTokenizer.from_pretrained("../roberta_data")

    dataset = Image_Text_Dataset(df,path,clip_processor,tokenizer)  # 你的数据集
    dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        print()
