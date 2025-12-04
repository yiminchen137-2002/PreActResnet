# PreActResNet: 高效的图像分类深度学习模型
Created by [yimin chen],[wenhan lai]。
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-370/)

本项目实现了基于Pre-Activation ResNet的高效图像分类模型，专为CIFAR-10数据集优化。通过先进的训练技术和数据增强方法，我们的模型在CIFAR-10上达到了优异的性能表现。

<!-- ## 🔥News
- **2025-12-04** 项目发布，包含完整的训练、测试和优化流程 -->

## 项目特点

| 特性 | 描述 |
| :---: | :--- |
| 网络架构 | Pre-Activation ResNet & ResNeXt |
| 数据集 | CIFAR-10 |
| 准确率 | 95%+ (具体数值根据实际训练结果填写) |
| 优化技术 | CutMix, MixUp, 标签平滑, 学习率调度等 |
| 框架 | PyTorch >= 2.5.1 |

## 安装要求

### 环境要求

- Python == 3.9
- PyTorch >= 2.5.1
- CUDA >= 10.2
- GCC >= 4.9 

### 依赖安装

```bash
pip install -r requirements.txt
```

项目所需的核心依赖包括：
- torch>=2.5.1
- torchvision>=0.20.1
- numpy>=2.0.2
- matplotlib>=3.9.4
- Pillow>=11.3.0
- timm>=1.0.21
- tqdm>=4.67.1
- huggingface-hub>=0.36.0
- safetensors>=0.6.2

## 项目结构

```
.
├── models/                 # 网络结构定义
│   ├── __init__.py
│   └── resnext.py          # ResNeXt网络实现
├── utils/                  # 辅助功能模块
│   ├── __init__.py
│   └── data_loader.py      # 数据加载器
├── fig/                    # 图表和可视化文件
├── checkpoints/            # 模型检查点
├── runs/                   # TensorBoard日志
├── data/                   # 数据集存储目录
├── optimize_system.py      # 系统优化脚本
├── requirements.txt        # 项目依赖
├── train.py                # 模型训练主程序
├── test.py                 # 模型测试程序
└── test_with_checkpoints.py # 检查点测试程序
```

## 模型
你可以通过以下链接来下载我们训练好的模型
https://pan.baidu.com/s/18yTiOg3UUjm1_eM-nNqFGA 提取码: z33f

### 数据准备

数据集需要下载到[data/](data/)目录中。使用CIFAR-10数据集进行训练和测试。

### 系统优化

在开始训练之前，建议先运行系统优化脚本来提升训练性能：

```bash
python optimize_system.py
```

该脚本会进行以下优化：
- 设置CUDA环境变量
- 启用PyTorch性能优化选项
- 检查GPU状态和系统资源
- 设置进程优先级

### 模型训练

训练模型，请运行：

```bash
python train.py
```

训练过程中采用了多种先进优化技术：
- **数据增强**: RandomCrop, RandomHorizontalFlip, ColorJitter, RandomErasing等
- **正则化**: 标签平滑交叉熵损失
- **优化器**: Adagrad优化器（可切换为SGD with Nesterov或AdamW）
- **学习率调度**: 预热+余弦退火策略
- **混合精度训练**: 自动混合精度以加速训练
- **数据增强策略**: CutMix和MixUp随机应用

训练完成后，最佳模型将保存在[checkpoints/](checkpoints/)目录中。训练结果将会保存在[runs/](runs/)目录中

### 模型测试

要测试训练好的模型，请运行：

```bash
python test.py
```

测试脚本将输出：
- 整体准确率
- 每个类别的准确率

## 网络架构

本项目实现了两种网络架构：

### Pre-Activation ResNet
- 基于预激活残差块构建
- 在每个残差块中先进行BatchNorm和ReLU激活，再进行卷积操作
- 更易于训练深层网络

## 配置说明

### 训练超参数

[train.py](train.py)中的关键超参数：

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| batch_size | 256 | 批处理大小 |
| epochs | 200 | 训练轮数 |
| lr | 0.05 | 初始学习率 |
| optimizer | Adagrad | 优化器类型 |
| :--- | :--- | :--- |
| batch_size | 512 | 批处理大小 |
| epochs | 200 | 训练轮数 |
| lr | 0.001| 初始学习率 |
| optimizer | Adamw | 优化器类型 |
| :--- | :--- | :--- |
| batch_size | 512 | 批处理大小 |
| epochs | 200 | 训练轮数 |
| lr | 0.003| 初始学习率 |
| optimizer | Adam | 优化器类型 |
| :--- | :--- | :--- |
| batch_size | 128 | 批处理大小 |
| epochs | 200 | 训练轮数 |
| lr | 0.0001| 初始学习率 |
| optimizer | RMSprop | 优化器类型 |
| :--- | :--- | :--- |
| batch_size | 512 | 批处理大小 |
| epochs | 200 | 训练轮数 |
| lr | 0.2| 初始学习率 |
| optimizer | SGD_M | 优化器类型 |
| :--- | :--- | :--- |
| batch_size | 512 | 批处理大小 |
| epochs | 200 | 训练轮数 |
| lr | 0.1| 初始学习率 |
| optimizer | SGD_N | 优化器类型 |

### 数据增强策略

[utils/data_loader.py](utils/data_loader.py)中实现了标准的数据增强：

- RandomCrop: 随机裁剪
- RandomHorizontalFlip: 随机水平翻转
- ColorJitter: 颜色抖动
- RandomErasing: 随机擦除

[train.py](train.py)中还实现了高级数据增强：

- MixUp: 图像和标签的线性插值
- CutMix: 将一张图片的部分区域替换为另一张图片

## 结果展示

你可以在[Results](Results)目录下查看我们的训练日志和结果
在CIFAR-10测试集上的性能表现：

| 模型 | 准确率 | 备注 |
| :--- | :--- | :--- |
| PreActResNet | ~95% | 使用全部优化技术 |

## 可视化

训练过程中的指标可以通过TensorBoard进行可视化：

```bash
tensorboard --logdir=runs/
```

可以监控的指标包括：
- 训练损失和准确率
- 测试准确率
- 学习率变化







