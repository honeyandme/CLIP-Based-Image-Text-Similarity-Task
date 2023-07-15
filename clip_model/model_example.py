import torch
from PIL import Image
import  torch.functional as F
from Clip import CLIPModel
from transformers import AutoTokenizer, CLIPProcessor, AutoProcessor


def init_model_processer(model_pre_vit,model_pre_bert,model_weight,images_processer_path,text_processer_path,device):
    model = CLIPModel(model_pre_vit,model_pre_bert)
    model = model.to(device)
    model.load_state_dict(torch.load(model_weight, map_location=device))
    images_processer = CLIPProcessor.from_pretrained(images_processer_path)
    text_processer = AutoTokenizer.from_pretrained(text_processer_path)
    return model,images_processer,text_processer

def process_data(texts, image_files, images_processer, text_processer,device):
    # 如果存在需要对图片进行预处理，则读取文件
    if image_files is not None:
        images = [Image.open(x).convert('RGB') for x in image_files]
    else:
        images = None
    image_input = images_processer(images=images, return_tensors="pt")
    text_input = text_processer(texts, return_tensors="pt", truncation=True,padding=True)

    image_input = image_input.to(device)
    text_input = text_input.to(device)
    return image_input,text_input



def cal_image_text_sim(model,processor,tokenizer,device):
    """
    计算图片和所有候选文本之间的相似度
    """
    print('-------------- 计算图文相似度 --------------')
    texts = ['斗罗大陆','唐三','比比东','药水哥','虎牙抽象直播','虎牙','简自豪','男人','打架','胖子','一群裸男在宿舍打架',
         '可爱的小鸡', '一个人在唱歌','一个人在洗澡','一只狗狗在吃东西'
    ]
    image_files = [
            './images/test/斗罗大陆.jpg', './images/test/药水哥.jpg','./images/test/uzi.jpg',
    ]
    # 特征处理

    image_input,text_input = process_data(texts,image_files,processor,tokenizer,device)


    with torch.no_grad():
        logits_per_image =  model.get_image_features(image_input['pixel_values'])
        logits_per_text  = model.get_text_features(text_input['input_ids'],text_input['attention_mask'] )


    logits_per_img2text = logits_per_image@logits_per_text.t()



    # 对分数做softmax
    logits_per_img2text = torch.softmax(logits_per_img2text, dim=-1)
    # 对于每张图片，将其与所有文本的相似度，进行降序排序
    logits_per_img2text = logits_per_img2text.cpu().numpy().tolist()
    for scores, file in zip(logits_per_img2text, image_files):
        result = sorted([(text, score) for text, score in zip(texts, scores)], key=lambda x: x[1], reverse=True)
        print('图片和所有候选文本的相似度：{}'.format(file))
        print(result)
        print()


if __name__ == '__main__':
    model_pre_vit = '../vit_clip_data'
    model_pre_bert = '../roberta_data'
    model_weight = './model_weight/best_model.pth'
    images_processer_path = '../vit_clip_data'
    text_processer_path = '../roberta_data'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model,images_processer,text_processer = init_model_processer(model_pre_vit,model_pre_bert,model_weight,images_processer_path,text_processer_path,device)
    cal_image_text_sim(model,images_processer,text_processer,device)
