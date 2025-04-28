import torch  # 导入PyTorch
from torch import nn  # 导入PyTorch的神经网络模块
from torch import optim  # 导入PyTorch的优化器模块
from model import Rock  # 导入自定义的卷积神经网络模型类
from torchvision import transforms  # 导入torchvision的图像变换模块
from torchvision import datasets  # 导入torchvision的数据集模块
from torch.utils.data import DataLoader  # 导入PyTorch的数据加载器模块
from config import ConfigModel


import os
import time
import csv


# 超参数设置
batch_size = 16
max_epochs = 2000
test_epochs = 1
image_size = 640
learning_rate = 3e-4

# 定义图像变换：调整大小为 640x640，然后转换为张量
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # 调整图像大小为 640x640
    transforms.ToTensor()  # 将图像转换为张量
    ])



# 数据集
train_path = r'E:\Deep_learning\model_pic\dataset\data\train'
#val_path = r'./dataset/data/val'
test_path = r'E:\Deep_learning\model_pic\dataset\data\test'

train_dataset = datasets.ImageFolder(root = train_path, transform=transform)
test_dataset = datasets.ImageFolder(root = test_path, transform=transform)
#val_dataset = datasets.ImageFolder(root = test_path, transform=transform)
out_path = r'./dataset/out'

def count_outnum(directory_path):
    # 获取目录下的所有条目
    entries = os.listdir(directory_path)
        
    # 过滤出子文件夹
    subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(directory_path, entry))]
        
    # 返回子文件夹的数量
    return len(subdirectories)


# 数据加载
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# GPU
device = 'cuda'

# 优化器配置
weight_decay = 1e-1  # 权重衰减
beta1 = 0.9  # adamw 优化器的 beta1 参数
beta2 = 0.95  # adamw 优化器的 beta2 参数
grad_clip = 1.0  # 梯度裁剪的值，0 表示禁用

# 模型
out_num = count_outnum(train_path) if count_outnum(train_path) == count_outnum(test_path) else None
config = ConfigModel(
    out_num=out_num,
    base_channels=16,
    use_cbam=True,
    dropout=0.5,
    fc_hidden_dim=64,
    use_ghost=True,
    c2f_blocks={
        32: 3,
        64: 1,
        128: 6,
        256: 3
    }
)
model = Rock(config).to(device)

# 优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
    )

# 优化器
optimizerSGD = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)



# 交叉熵损失
criterion = nn.CrossEntropyLoss()

# 计算损失
@torch.no_grad()
def estimate_loss():
    # model.eval()  
    losses = []  
    correct = 0  
    total = 0  

    for epoch in range(test_epochs):  
        for batch_idx, (data, label) in enumerate(test_loader):  
            data, label = data.to(device), label.to(device)  

            output = model(data)  
            loss = criterion(output, label)  
            losses.append(loss.item())  

            # 计算准确率
            _, predicted = torch.max(output, 1)  # 找到每行的最大值作为预测类别
            correct += (predicted == label).sum().item()  # 统计预测正确的样本数
            total += label.size(0)  # 累加样本总数

            print("正在评估...")
            print(f"Epoch {epoch + 1}/{test_epochs} "
                  f"| 批次 {batch_idx}/{len(test_loader)} "
                  f"| 损失: {loss.item():.4f} "
                  f"| 准确率: {correct / total:.4f}"
            )

    model.train()  
    avg_loss = sum(losses) / len(losses)  
    accuracy = correct / total  # 总体准确率
    
    return avg_loss, accuracy





if __name__ == '__main__':
    # 训练
    t0 = time.time()

    # 为保存记录文件设置文件路径
    log_file_path = os.path.join(out_path, 'training_log_conv.csv')

    # 如果文件不存在，则创建文件并写入标题行
    if not os.path.isfile(log_file_path):
        with open(log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Accuracy"])

    for epoch in range(max_epochs):
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)

            # 前向传播：计算模型输出
            output = model(data)
            # 计算损失
            loss = criterion(output, label)
            # 反向传播：计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 清零梯度以便进行下一次迭代
            optimizer.zero_grad()

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            print(f"Epoch {epoch + 1}/{max_epochs} "
                  f"| 批次 {batch_idx}/{len(train_loader)} "
                  f"| 损失: {loss.item():.4f}"
                  f"| 时间: {dt:.4f}")

            
        # 每100个批次打印一次损失
        if (epoch) % 1 == 0:
            test_loss, test_accuracy = estimate_loss()
            model_path = os.path.join(out_path, f'rock_{epoch}_{test_accuracy:.4f}.pth')
            torch.save(model.state_dict(), model_path)
            print("正在评估...")
            print(f"Epoch {epoch + 1}/{max_epochs} "
                  f"| 损失: {test_loss:.4f}"
                  f"|总体准确率: {test_accuracy:.4f}"
                  f"| 模型保存到 {model_path}")
            # 将每一个轮的测试数据写入 CSV 文件
            with open(log_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, loss.item(), test_accuracy])
            

    


    