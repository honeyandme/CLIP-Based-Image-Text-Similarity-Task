import math

import torch
import torch.utils.checkpoint
from torch import nn



class Vit_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = ViTEmbeddings()
        self.encoder = ViTEncoder()
        self.mlp_head = ViTPooler()
        self.cls = nn.Linear(768, 2)
    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        poolout,hidden_states = self.mlp_head(hidden_states)
        logits = self.cls(poolout)
        return logits

class ViTEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        # 负责把图片转换成patch embeddings [batch_size, 196, 768]
        self.patch_embeddings = ViTPatchEmbeddings()
        num_patches = self.patch_embeddings.num_patches
        # 位置编码
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, 768))
        self.dropout = nn.Dropout(0.1)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        # [batch_size, 197, 768]
        return embeddings

class ViTPatchEmbeddings(nn.Module):


    def __init__(self):
        super().__init__()
        image_size, patch_size = 224, 16
        num_channels, hidden_size = 3, 768

        image_size =  (image_size, image_size)
        patch_size =  (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        # [batch_size, 3, 224, 224] -> [batch_size, 768, 14, 14]
        x = self.projection(pixel_values).flatten(2).transpose(-1, -2)
        # [batch_size, 768, 14, 14] -> [batch_size, 196, 768]
        return x


class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer() for _ in range(6)])

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        return hidden_states

class ViTLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = ViTSelfAttention()
        self.layer_norm1 = nn.LayerNorm(768, eps=1e-6)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm2 = nn.LayerNorm(768, eps=1e-6)
        self.mlp_block = ViTMLPBlock()



    def forward(self, hidden_states):
        hidden_states_in_attention = self.layer_norm1(hidden_states)
        # [batch_size, 197, 768]
        attention_output = self.attention(hidden_states_in_attention)
        # [batch_size, 197, 768]
        hidden_states = hidden_states + self.dropout(attention_output)

        hidden_states_in_mlp = self.layer_norm2(hidden_states)
        # [batch_size, 197, 768]
        mlp_output = self.mlp_block(hidden_states_in_mlp)
        # [batch_size, 197, 768]
        hidden_states = hidden_states + self.dropout(mlp_output)
        return hidden_states



class ViTSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 4
        self.attention_head_size = 768 // 4
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(768, 768)
        self.key = nn.Linear(768, 768)
        self.value = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(768, 768)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # [batch_size, 197, 768] -> [batch_size, 197, 4, 192]
        x = x.view(*new_x_shape)
        # [batch_size, 197, 4, 192] -> [batch_size, 4, 197, 192]
        return x.permute(0, 2, 1, 3)
    def forward(self,hidden_states):
        # [batch_size, 197, 768]
        mixed_query_layer = self.query(hidden_states)
        # [batch_size, 197, 768]
        mixed_key_layer = self.key(hidden_states)
        # [batch_size, 197, 768]
        mixed_value_layer = self.value(hidden_states)
        # [batch_size, 4, 197, 192]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # [batch_size, 4, 197, 192]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # [batch_size, 4, 197, 192]
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # # [10, 4, 197, 192] @[10, 4, 192, 197] -> [10, 4, 197, 197]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # [batch_size, 4, 197, 197]
        attention_probs = self.softmax(attention_scores)
        # [batch_size, 4, 197, 197] [batch_size, 4, 197, 192]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [batch_size, 197, 768]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # [batch_size, 197, 768]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [batch_size, 197, 768]
        context_layer = context_layer.view(*new_context_layer_shape)
        # [batch_size, 197, 768]
        attention_output = self.fc(context_layer)
        # [batch_size, 197, 768]
        attention_output = self.dropout(attention_output)
        # [batch_size, 197, 768]
        return  attention_output

class ViTMLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(3072, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        # [batch_size, 197, 768]
        x = self.fc1(hidden_states)
        # [batch_size, 197, 3072]
        x = self.gelu(x)
        x = self.dropout(x)
        # [batch_size, 197, 3072]
        x = self.fc2(x)
        # [batch_size, 197, 768]
        x = self.dropout(x)
        # [batch_size, 197, 768]
        return x

class ViTPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()
        self.output = nn.Linear(768, 768)

    def forward(self, hidden_states):
        # [batch_size, 197, 768]
        first_token_tensor = hidden_states[:, 0]
        # [batch_size, 768]
        pooled_output = self.dense(first_token_tensor)
        # [batch_size, 768]
        pooled_output = self.activation(pooled_output)
        # [batch_size, 768]
        pooled_output = self.output(pooled_output)
        # [batch_size, 768]
        # 对hideen_states进行pooling

        return pooled_output,hidden_states

if __name__ == '__main__':
    ten = torch.randn(10, 3,224, 224)
    vit = Vit_model()
    out = vit(ten)
    print(out.shape)