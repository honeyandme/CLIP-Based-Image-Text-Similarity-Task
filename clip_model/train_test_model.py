import numpy as np
import torch

from Clip import CLIPModel
from  get_dataloader import get_dataloader
from transformers import AdamW
from tensorboardX import SummaryWriter
from tqdm import tqdm
def init_model():
    model = CLIPModel("../vit_clip_data", "../roberta_data")
    model = model.cuda()
    return model



from tensorboardX import SummaryWriter
from tqdm import tqdm

from tensorboardX import SummaryWriter
from tqdm import tqdm

def train(model, dataloader, optimizer, num_epochs):
    model.train()  # 将模型设置为训练模式
    writer = SummaryWriter(log_dir='log')  # 指定 log_dir 为 'log' 文件夹

    global_step = 0
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        best_loss = np.Inf
        pbar = tqdm(dataloader, desc="Training")
        for i, batch in enumerate(pbar):
            # 将数据移动到GPU
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            image = batch['image'].cuda()

            # 前向传播
            text_features, image_features = model(input_ids, attention_mask, image)

            # 计算损失
            loss = model.compute_loss(image_features, text_features)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_description(f"Training epoch {epoch+1} (loss: {epoch_loss/(i+1):.4f})")

            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "./model_weight/best_model.pth")
            print("New best model saved!")
    print("Training complete.")
    writer.close()



if __name__ == '__main__':
    model = init_model()
    dataloader = get_dataloader(64)  # 假设你已经定义了这个函数
    optimizer = AdamW(model.parameters(), lr=2e-5)

    train(model, dataloader, optimizer, num_epochs=40)


