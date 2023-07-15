import tensorboardX
import torch
from torch import nn
from transformers import CLIPVisionModel, AutoModel



# 定义clip模型
class CLIPModel(nn.Module):
    def __init__(self, vision_model_name, text_model_name):
        super(CLIPModel, self).__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(vision_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.logit_scale = nn.Parameter(torch.ones([]))

        # 冻结视觉模型的权重
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_features = self.vision_model(image).pooler_output

        return text_features, image_features
    def get_image_features(self, image):
        return self.vision_model(image).pooler_output
    def get_text_features(self, input_ids, attention_mask):
        return self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
    def compute_loss(self, image_features, text_features):
        # 进行归一化
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # 计算余弦相似度
        logit_scale = self.logit_scale.exp()
        logits_per_image =  logit_scale*image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        N = image_features.shape[0]  # 假设image_features的第一个维度是批次大小
        labels = torch.arange(N).to(image_features.device)  # 创建标签

        # loss_image = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_text = nn.CrossEntropyLoss()(logits_per_text, labels)
        # loss = (loss_image + loss_text) / 2

        loss = loss_text
        return loss
