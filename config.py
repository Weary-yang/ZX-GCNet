# 模型配置文件

class ConfigModel:
    def __init__(
        self,
        out_num: int,
        base_channels: int = 16,
        use_cbam: bool = True,
        dropout: float = 0.5,
        c2f_blocks: dict = None,
        fc_hidden_dim: int = 64,
        use_ghost: bool = True
    ):
        """
        模型配置类，用于构建网络结构时的参数设置
        """
        self.out_num = out_num
        self.base_channels = base_channels
        self.use_cbam = use_cbam
        self.dropout = dropout
        self.fc_hidden_dim = fc_hidden_dim
        self.use_ghost = use_ghost
        self.c2f_blocks = c2f_blocks or {
            32: 3,
            64: 1,
            128: 6,
            256: 3
        }