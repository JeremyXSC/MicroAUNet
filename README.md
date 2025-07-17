# MicroAUNet: 基于知识蒸馏的轻量化医学图像分割网络

## 项目简介

MicroAUNet是一个专为医学图像分割任务设计的轻量化深度学习框架。该项目实现了基于知识蒸馏的师生网络架构，其中MALUNet作为教师模型，MicroAUNet（MAUNet）作为学生模型，通过课程学习和对比学习机制实现高效的医学图像分割。

## 核心特性

### 🏗️ 轻量化网络架构
- **DWDConv（深度可分离空洞卷积）**: 减少参数量的同时保持感受野
- **LDGA（轻量化深度引导注意力）**: 高效的特征提取和注意力机制
- **LSCAB（轻量化跨阶段注意力桥接）**: 多尺度特征融合

### 🎓 知识蒸馏框架
- **师生网络**: MALUNet（教师）→ MicroAUNet（学生）
- **课程学习**: 分阶段训练策略（模仿学习 → 偏好蒸馏）
- **多损失函数**: 分割损失 + 模仿损失 + 对比损失

### 📊 支持数据集
- **Kvasir-SEG**: 息肉分割数据集
- **CVC-ClinicDB**: 结肠镜图像分割数据集

## 项目结构

```
MALUNet/
├── configs/                    # 配置文件
│   └── config_setting.py      # 训练配置
├── data/                       # 数据集目录
│   ├── CVC/                   # CVC-ClinicDB数据集
│   └── Kvasir-SEG/           # Kvasir-SEG数据集
├── dataset/                    # 数据加载器
│   ├── cvc_datasets.py       # CVC数据集加载器
│   └── npy_datasets.py       # NPY格式数据集加载器
├── models/                     # 模型定义
│   ├── malunet.py            # MALUNet教师模型
│   └── maunet.py             # MicroAUNet学生模型
├── results/                    # 训练结果
├── engine.py                   # 训练/验证引擎
├── train.py                    # 主训练脚本
├── utils.py                    # 工具函数
└── requirements.txt            # 依赖包
```

## 安装与环境配置

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+

### 安装依赖
```bash
pip install torch torchvision
pip install -r requirements.txt
```

## 数据集准备

### Kvasir-SEG数据集 [下载地址](https://datasets.simula.no/downloads/kvasir-seg.zip)
```
data/Kvasir-SEG/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

### CVC-ClinicDB数据集 [下载地址](https://www.dropbox.com/scl/fi/ky766dwcxt9meq3aklkip/CVC-ClinicDB.rar?rlkey=61xclnrraadf1niqdvldlds93&e=1&dl=0)
```
data/CVC/
└── PNG/
    ├── Original/          # 原始图像
    └── Ground Truth/      # 标注掩膜
```

## 模型架构

### MicroAUNet（学生模型）
```python
class MAUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, 
                 c_list=[8,16,24,32,48,64], bridge=True):
        # 编码器: DWDConv + LDGA
        # 桥接: LSCAB轻量化跨阶段注意力
        # 解码器: LDGA + DWDConv
```

**核心组件:**
- **DepthwiseSeparableDilatedConv**: 深度可分离空洞卷积
- **LightweightDGA**: 轻量化深度引导注意力
- **LSCAB**: 轻量化跨阶段注意力桥接
- **DWDBlock**: 深度可分离空洞卷积块

### MALUNet（教师模型）
- 更复杂的网络结构，用于指导学生模型学习
- 提供丰富的特征表示和分割掩膜

## 训练配置

### 基本配置
```python
# configs/config_setting.py
class setting_config:
    network = 'malunet'
    datasets = 'Kvasir-SEG'  # 或 'CVC-ClinicDB'
    
    # 模型配置
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'c_list': [8, 16, 24, 32, 48, 64],
        'bridge': True,
    }
    
    # 训练参数
    batch_size = 8
    epochs = 2
    input_size_h = 256
    input_size_w = 256
    
    # 优化器
    opt = 'AdamW'
    lr = 0.001
    weight_decay = 1e-2
    
    # 学习率调度
    sch = 'CosineAnnealingLR'
    T_max = 50
```

## 使用方法

### 1. 训练模型
```bash
python train.py
```

### 2. 自定义配置
修改 `configs/config_setting.py` 中的参数：
- 数据集路径
- 模型参数
- 训练超参数

### 3. 学习策略
训练过程分为两个阶段：

**阶段1: 模仿学习（Imitation Learning）**
- 时间: 前60%的训练轮次
- 目标: 学习教师模型的特征表示和概率分布
- 损失: 分割损失 + KL散度损失 + 特征对齐损失

**阶段2: 偏好蒸馏（Preference Distillation）**
- 时间: 后40%的训练轮次
- 目标: 基于对比学习优化分割性能
- 损失: 分割损失 + 对比损失 + L2正则化

## 损失函数

### 1. 分割损失
```python
criterion = BceDiceLoss()  # BCE + Dice Loss
```

### 2. 模仿损失
```python
class ImitationLoss(nn.Module):
    def forward(self, student_pred, teacher_pred, 
                student_features, teacher_features):
        L_KL = F.kl_div(F.log_softmax(student_pred, dim=1), 
                        F.softmax(teacher_pred, dim=1))
        L_mimic = feature_alignment_loss(student_features, teacher_features)
        return (1 - omega_KL) * L_mimic + omega_KL * L_KL
```

### 3. 对比损失
```python
class ContrastiveLoss(nn.Module):
    def forward(self, student_pred, teacher_pred, targets):
        # InfoNCE对比学习损失
        return infonce_loss(positive_samples, negative_samples)
```

## 评估指标

- **mIoU**: 平均交并比
- **Dice/F1**: Dice系数
- **Accuracy**: 准确率
- **Sensitivity**: 敏感性（召回率）
- **Specificity**: 特异性

## 文件说明

### 核心文件
- `train.py`: 主训练脚本，实现课程学习和知识蒸馏
- `models/maunet.py`: MicroAUNet学生模型定义
- `models/malunet.py`: MALUNet教师模型定义
- `engine.py`: 训练和验证引擎
- `utils.py`: 工具函数（优化器、调度器、日志等）

### 配置文件
- `configs/config_setting.py`: 训练配置参数

### 数据处理
- `dataset/npy_datasets.py`: Kvasir-SEG数据集加载器
- `dataset/cvc_datasets.py`: CVC-ClinicDB数据集加载器