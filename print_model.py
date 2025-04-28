import torch
from model import Rock
from train import count_outnum
from config import ConfigModel

# 数据集
train_path = r'E:\Deep_learning\model_pic\dataset\data\train'
#val_path = r'D:\vscode_code\Deep_learning\study_CNN\MNIST_data\val'
test_path = r'E:\Deep_learning\model_pic\dataset\data\test'


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
model = Rock(config)



# 打印整个模型的结构
print(model)
