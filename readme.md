## 三层神经网络分类器

### 文件结构

```python
Three Layer Net (按推荐阅读顺序[从上往下])
├──requirements.txt         # 依赖库信息
├──__init__.py              # 导入依赖库
├──activation_func.py       # 激活函数
├──linear_component.py      # 线性（+激活）层
├──loss.py                  # 损失函数
├──full_connect_network.py  # N层神经网络分类器
├──optimization.py          # 优化器
├──solver.py                # 求解器，包括训练、预测、保存模型、导入模型
├──utils.py                 # 载入数据，可视化神经网络的辅助函数
├──main.py                  # 主函数，定义parser，用于在终端运行代码
├──main.ipynb               # 包含训练、导入、测试的notebook文件
├──main.html                # notebook文件的html形式
├──data                     # 数据集
    ├──train-images-idx3-ubyte.gz   # 训练图片
    ├──train-labels-idx1-ubyte.gz   # 训练标签
    ├──t10k-images-idx3-ubyte.gz    # 测试图片
    └──t10k-labels-idx1-ubyte.gz    # 测试标签
├──model                    # 存放模型文件
    └──fcnn.npz                     # 训练好的模型参数
└──readme.md                # readme
```

### 数据集

| 名称  | 描述 | 样本数量 | 保存位置|
| --- | --- |--- | --- |
| `train-images-idx3-ubyte.gz`  | 训练集的图像  | 60,000| `data['X_train']` |
| `train-labels-idx1-ubyte.gz`  | 训练集的类别标签  |60,000|`data['y_train']` |
| `t10k-images-idx3-ubyte.gz`  | 测试集的图像  | 10,000|`data['X_val']` |
| `t10k-labels-idx1-ubyte.gz`  | 测试集的类别标签  | 10,000| `data['y_val']` |


### 使用说明

#### 将github上的文件下载到本地

```bash
git clone https://github.com/mango7789/Computer-Vision-HW1.git
cd Computer-Vision-HW1 
```

#### 下载模型权重文件

- [下载地址](https://drive.google.com/file/d/1fHbpA-FtWAH-j2v-awv-D9p3sIjMfqLW/view?usp=drive_link)
- 将下载好的权重文件放入`./model`文件夹(相对路径应为`./model/<file_name>`)

#### 运行代码（提供两种方案）

1. 在`main.ipynb`中运行
     - 根据notebook中的说明运行相应代码块 
 
2. 在终端中运行
     - 安装依赖库
       ```bash
       pip install -r requirements.txt
       ``` 
     - 训练
       ```bash
       # use help to check the provided args
       python main.py train -h

       options:
       -h, --help            show this help message and exit
       --hidden_dims HIDDEN_DIMS [HIDDEN_DIMS ...]
                               Sizes of the hidden layers, default is [128, 64].
       --activation ACTIVATION [ACTIVATION ...]
                               Activation functions, the length should be 1 or equal to the the hidden dims, can choose from ['relu', 'tanh', 'sigmoid'].
       --reg REG             Regularization strength of the l_2 penalty, default is 0.01
       --weight_scale WEIGHT_SCALE
                               Weight scale of the initial weigh matrix, default is 0.01.
       --epochs EPOCHS       Number of training epochs, default is 10.
       --iters ITERS         Number of training iterations, defalut is 6000.
       --update_rule UPDATE_RULE
                               Update rule in training, default is 'sgd', can choose from ['sgd', 'sgd_momentum', 'adam', 'rmsprop'].
       --learning_rate LEARNING_RATE
                               Learning rate of the training, defalut is 1e-3.
       --lr_decay LR_DECAY   Learning rate decay, default is 0.9.
       --batch_size BATCH_SIZE
                               Batch size, default is 64.
       --save SAVE           The trained model should be saved or not, default is False.
       ```
       ```bash
       # example use
       python main.py train --hidden_dims 64 32 --activation relu tanh --reg 0.1
       ``` 
     - 测试
       ```bash
       # use help to check the provided args
       python main.py test -h 

       options:
       -h, --help   show this help message and exit
       --path PATH  The path of the trained model in dir ./model.
       ```
       ```bash
       # example use
       python main.py test --path fcnn.npz
       ``` 
