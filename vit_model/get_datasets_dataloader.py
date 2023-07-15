
from  vit import Vit_model
from torchvision.datasets import ImageFolder
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # 变为224x224大小的图像，这通常是一个良好的选择
    transforms.ToTensor(),           # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 对图像进行归一化
])
print(torch.cuda.is_available())
# 创建数据集
train_dir = '../data/dogvscat/training_set'
test_dir = '../data/dogvscat/test_set'

train_dataset = ImageFolder(root=train_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

# 创建数据加载器
batch_size = 128   # 按照你的硬件和需求进行调整

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义模型，损失函数和优化器
model = Vit_model()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练模型
from tqdm import tqdm

# 训练模型
n_epochs = 10
best_acc = 0.0  # 初始的最佳准确率为0

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{n_epochs}")

    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 后向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'Loss': running_loss / (i + 1)})

    # 在每个epoch结束后在测试集上进行验证，并保存最佳模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # 取得分最高的那个类 (outputs.data的索引号)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print('Epoch {}: Accuracy on the test set: {}'.format(epoch + 1, acc))

    # 如果当前模型的准确率比之前的最佳准确率高，就保存当前模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_vit_model.pth')

print('Finished Training')
