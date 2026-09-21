# 高光谱数据分类系统

一个完整的高光谱数据处理、分析和分类系统，支持机器学习和深度学习算法。

## 项目概述

本项目提供了从原始高光谱数据提取、预处理到模型训练、验证和可视化的完整工作流程。支持多种分类算法（逻辑回归、随机森林、LDA、XGBoost、神经网络）和交叉验证评估。

**作者**: Maggie Zhang  
**版本**: v2.0  
**最后更新**: 2025年12月16日

---

## 目录

- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
- [配置说明](#配置说明)
- [输出说明](#输出说明)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 功能特性

### 数据处理
- ✅ 高光谱HDR/数据文件解析
- ✅ 支持BSQ、BIL、BIP三种数据组织格式
- ✅ 自动波长信息提取
- ✅ 波段平均反射率计算

### 机器学习分类
- ✅ 逻辑回归（带PCA降维）
- ✅ 随机森林
- ✅ 线性判别分析（LDA）
- ✅ XGBoost梯度提升
- ✅ 神经网络（PyTorch实现）

### 训练与评估
- ✅ 分层K折交叉验证（默认5折）
- ✅ 自动类别平衡处理
- ✅ 早停机制和学习率调度
- ✅ GPU/CPU自适应
- ✅ 模型保存与加载

### 可视化
- ✅ 反射率曲线图
- ✅ 混淆矩阵（支持中文标签）
- ✅ 训练损失曲线
- ✅ 特征重要性图（随机森林）
- ✅ 训练指标监控

---

## 项目结构

```
Code_FHY_V1/
├── README.md                           # 项目说明文档
├── requirements.txt                     # Python依赖包列表
│
├── Data/                               # 数据目录
│   ├── Exp1/                           # 第一批数据
│   └── Exp2/                           # 第二批数据
│
├── models/                             # 模型保存目录
│   ├── Spectral_Classification_Exp1/   # 第一批数据训练结果
│   │   ├── LogisticRegression/
│   │   ├── Random_Forest/
│   │   ├── LDA/
│   │   ├── XGBoost/
│   │   └── Neural_Network/
│   │   └── training_log_*.log          # 训练日志
│   │
│   ├── Spectral_Classification_Exp2/   # 第二批数据训练结果
│   │   ├── LogisticRegression/
│   │   ├── Random_Forest/
│   │   ├── LDA/
│   │   ├── XGBoost/
│   │   └── Neural_Network/
│   │   └── training_log_*.log          # 训练日志
│   │
│   └── three_classes_model             # 三分类加载的模型
│   
│
├── plot/                               # 图表输出目录
│   ├── Spectral_Classification_Exp1/
│   │   ├── LogisticRegression/
│   │   ├── Random_Forest/
│   │   └── ...
│   └── reflectance/                   # 反射率曲线图
│
├── configs/                            # 配置文件目录
│   └── SpectralNN_config.json         # 神经网络配置
│
├── utils/                              # 工具函数模块
│   ├── __init__.py
│   ├── config_utils.py                # 配置加载
│   ├── data_utils.py                  # 数据处理
│   ├── plot_setting.py                # 绘图设置
│   ├── confusion_matrix_plot_utils.py # 混淆矩阵绘制
│   ├── loss_plot_utils.py             # 损失曲线绘制
│   ├── metrics.py                     # 评估指标
│   └── seed.py                        # 随机种子设置
│
├── 数据提取脚本
│   ├── extract_average_reflectivity.py # 提取波段平均反射率
│   ├── extract_leaf_*.py              # 各种叶片数据提取脚本
│   └── view_*.py                      # 数据查看工具
│
├── 训练脚本
│   ├── spectral_ML_classifier.py      # 机器学习模型训练
│   └── SpectralNN_classifier.py       # 神经网络模型训练
│
├── 验证脚本
│   ├── validate_spectral_ML.py        # ML模型验证
│   └── validate_SpectralNN.py         # NN模型验证
│
└── 可视化脚本
    ├── plot_reflectivity.py           # 绘制反射率曲线
    └── plot_setting.py                # 绘图样式设置
```

---

## 环境要求

### 系统要求
- **操作系统**: Windows
- **Python版本**: 3.7+（推荐3.9或3.10）
- **内存**: 8GB以上（推荐16GB）
- **显卡**: 可选，支持CUDA的NVIDIA显卡可加速训练

### Python依赖包
详见 `requirements.txt`

主要依赖：
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- torch >= 2.0.0
- matplotlib >= 3.5.0
- xgboost >= 1.5.0

---


## 快速开始

### 步骤1: 数据提取

从高光谱HDR文件中提取平均反射率：

```bash
python extract_average_reflectivity.py
```

默认配置（在脚本的`main()`函数中修改）：
- 输入目录: `Data/Leaf`
- 输出文件: `Data/leaf.csv`

### 步骤2: 训练模型

#### 训练机器学习模型

```bash
python spectral_ML_classifier.py --experiment Spectral_Classification_Exp2/day14-NIR --folds 5
```

#### 训练神经网络模型

```bash
python SpectralNN_classifier.py --csv Data/Exp2/day14-NIR.csv --experiment Spectral_Classification_Exp2/day14-NIR --folds 5
```

### 步骤3: 验证模型

#### 验证所有fold

```bash
python validate_spectral_ML.py --csv Data/Exp2/day14-NIR.csv --experiment Spectral_Classification_Exp2/day14-NIR --model LDA --folds 5
```

#### 验证特定fold

```bash
python validate_spectral_ML.py --csv Data/Exp2/day14-NIR.csv --experiment Spectral_Classification_Exp2/day14-NIR --model LDA --folds 5 --fold 3
```

### 步骤4: 可视化结果

```bash
python plot_reflectivity.py
```

---

## 详细使用说明

### 数据提取

#### extract_average_reflectivity.py

**功能**: 从高光谱HDR/数据文件中提取每个波段的平均反射率

**使用方法**:
```python
# 编辑脚本中的main()函数
data_directory = "Data/YourFolder"  # 数据目录
output_file = "Data/output.csv"     # 输出文件

# 运行脚本
python extract_average_reflectivity.py
```

**输入格式**:
- HDR文件 (.hdr): 包含元数据
- 数据文件 (.dat/.img): 二进制光谱数据

**输出格式**:
```csv
label,filename,900.00,901.50,903.00,...
CEES,sample1,0.234,0.256,0.278,...
Control,sample2,0.189,0.201,0.215,...
```

### 模型训练

#### spectral_ML_classifier.py

**支持的算法**:
- LogisticRegression: 逻辑回归（带PCA降维）
- Random_Forest: 随机森林
- LDA: 线性判别分析
- XGBoost: 梯度提升树

**命令行参数**:
```bash
--experiment    # 实验名称（必需）
--folds         # 交叉验证折数（默认5）
--retrain       # 强制重新训练
--val-size      # 验证集比例（默认0.15）
--test-size     # 测试集比例（默认0.15）
```

**示例**:
```bash
python spectral_ML_classifier.py --experiment Spectral_Classification_Exp2/day14-NIR --folds 5
```

#### SpectralNN_classifier.py

**特点**:
- 多层神经网络（可配置）
- 支持Dropout和BatchNorm
- 早停机制
- 学习率调度（Cosine/Plateau）
- 类别平衡

**命令行参数**:
```bash
--csv           # CSV数据路径（必需）
--experiment    # 实验名称（必需）
--folds         # 交叉验证折数（默认5）
--config        # JSON配置文件路径（默认configs/SpectralNN_config.json）
```

**配置文件** (configs/SpectralNN_config.json):
```json
{
  "hidden_dims": [1024, 512, 256],
  "dropout_rate": 0.3,
  "batch_size": 128,
  "max_epochs": 500,
  "patience": 30,
  "optimizer": {
    "lr": 0.0005,
    "weight_decay": 0.00001
  },
  "scheduler": {
    "type": "cosine",
    "T_0": 30,
    "eta_min": 0.00001
  }
}
```

### 模型验证

#### validate_spectral_ML.py

**重要说明**:
训练时保存的是交叉验证中表现最好的那一折的模型。

**验证模式**:
1. **验证所有fold** (默认):
   - 最佳fold会复现训练结果
   - 其他fold提供泛化性能参考

2. **验证特定fold** (--fold N):
   - 只验证指定的fold
   - 用于精确复现训练时的最佳结果

**示例**:
```bash
# 验证所有fold
python validate_spectral_ML.py --csv Data/Exp2/day14-NIR.csv --experiment Spectral_Classification_Exp2/day14-NIR --model LDA --folds 5

# 只验证第3个fold（假设训练时最佳fold是3）
python validate_spectral_ML.py --csv Data/Exp2/day14-NIR.csv --experiment Spectral_Classification_Exp2/day14-NIR --model LDA --folds 5 --fold 3
```

**输出内容**:
- 每个fold的准确率
- 详细的分类报告
- 每类别的precision、recall、f1-score、准确率
- 混淆矩阵（PDF）
- 平均准确率和标准差

### 可视化

#### plot_reflectivity.py

**功能**: 绘制所有样本的反射率曲线图

**使用方法**:
```python
# 编辑脚本中的main()函数
csv_file = "Data/band_average_reflectivity_results_3.csv"

# 运行脚本
python plot_reflectivity.py
```

**输出**: `plot/Reflectance_all_samples_plot_3.pdf`

**特点**:
- 自动识别波长信息
- 按类别着色
- 智能刻度设置
- 水平排列图例
- 透明背景PDF矢量图

---

## 配置说明

### 神经网络配置 (configs/SpectralNN_config.json)

```json
{
  "hidden_dims": [1024, 512, 256],      // 隐藏层维度
  "dropout_rate": 0.3,                   // Dropout比例
  "batch_size": 128,                     // 批次大小
  "val_ratio": 0.2,                      // 验证集比例
  "use_class_weight": true,              // 是否使用类别权重
  "noise_std": 0.0,                      // 训练噪声强度
  "patience": 30,                        // 早停耐心值
  "max_epochs": 500,                     // 最大训练轮数
  "drop_last": true,                     // 是否丢弃最后不完整批次
  
  "optimizer": {
    "lr": 0.0005,                        // 学习率
    "weight_decay": 0.00001              // 权重衰减
  },
  
  "scheduler": {
    "type": "cosine",                    // 调度器类型: cosine/plateau
    "T_0": 30,                           // Cosine重启周期
    "T_mult": 1,                         // 周期倍增因子
    "eta_min": 0.00001,                  // 最小学习率
    "per_batch": true                    // 是否按批次更新
  }
}
```

---

## 输出说明

### 目录结构

```
models/[experiment_name]/
├── training_log_[timestamp].log        # 训练日志
├── validation_log_[model]_[timestamp].log  # 验证日志
├── LogisticRegression/
│   └── LogisticRegression.pth          # 模型文件
├── Random_Forest/
│   └── Random_Forest.pth
├── LDA/
│   └── LDA.pth
├── XGBoost/
│   └── XGBoost.pth
└── Neural_Network/
    ├── nn_model.pth                    # 模型参数
    ├── nn_scaler.pth                   # 数据标准化器
    ├── nn_label_encoder.pth            # 标签编码器
    └── model.pth                       # 完整模型bundle

plot/[experiment_name]/
├── LogisticRegression/
│   └── confusion_matrix_*.pdf          # 混淆矩阵
├── Random_Forest/
│   ├── confusion_matrix_*.pdf
│   └── feature_importance_*.pdf        # 特征重要性
└── Neural_Network/
    ├── confusion_matrix_*.pdf
    ├── loss_fold_*.pdf                 # 损失曲线
    └── fold_*.pdf                      # 训练指标
```

### 日志文件

**训练日志**包含:
- 系统信息
- 数据加载信息
- 每个fold的训练/验证/测试结果
- 分类报告（precision、recall、f1-score）
- 每类别准确率
- 最佳模型信息

**验证日志**包含:
- 模型加载信息
- 数据划分信息
- 每个fold的测试结果
- 详细的类别统计
- 混淆矩阵保存路径

---

## 常见问题

### Q1: 训练时内存不足怎么办？

**A**: 
1. 减少批次大小（修改config中的`batch_size`）
2. 减少神经网络层数或隐藏层维度
3. 使用数据采样（在代码中会自动处理）

### Q2: 为什么验证结果和训练结果不一致？

**A**: 
训练时保存的是交叉验证中表现最好的那一折的模型。验证所有fold时，只有最佳fold会复现训练结果，其他fold是用最佳模型在不同数据分割上的性能。

如果需要精确复现训练结果：
1. 查看训练日志，找到最佳fold编号
2. 使用 `--fold N` 参数只验证该fold

### Q3: 如何修改数据路径？

**A**: 
编辑对应脚本的`main()`函数：
```python
# spectral_ML_classifier.py
csv_file = "Data/Exp2/day14-NIR.csv"
experiment_name = "Spectral_Classification_Exp2/day14-NIR"

# extract_average_reflectivity.py
data_directory = "Data/Leaf"
output_file = "Data/leaf.csv"
```

### Q4: 支持GPU加速吗？

**A**: 
是的，神经网络训练自动检测GPU：
- 有GPU时自动使用CUDA加速
- 无GPU时使用CPU训练
- 可以在日志中查看使用的设备

安装GPU版PyTorch：
```bash
# 访问 https://pytorch.org/ 选择对应的CUDA版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Q5: 如何处理中文标签乱码？

**A**: 
系统已自动配置中文字体（Microsoft YaHei/SimHei）。如果仍有问题：
1. 确保系统安装了中文字体
2. 检查CSV文件编码为UTF-8
3. 查看日志中的字体配置信息

### Q6: 模型训练需要多长时间？

**A**: 
取决于多个因素：
- **机器学习模型**: 通常几分钟到十几分钟
  - LogisticRegression: 最快
  - Random Forest: 中等
  - XGBoost: 较慢但效果好
  
- **神经网络**: 取决于数据量和配置
  - 小数据集（<1000样本）: 10-30分钟
  - 中等数据集（1000-5000样本）: 30-60分钟
  - 使用GPU可大幅加速（3-10倍）

### Q7: 如何选择最佳模型？

**A**: 
训练会自动选择并保存交叉验证中准确率最高的模型。查看训练日志：
```
最佳模型: Random_Forest (准确率: 0.9500)
Random Forest 最佳模型来自第 3 折，准确率: 0.9500
```

### Q8: 可以使用自己的数据吗？

**A**: 
可以，需要准备以下格式：

**CSV文件格式**:
```csv
label,filename,波长1,波长2,波长3,...
类别A,样本1,0.234,0.256,0.278,...
类别B,样本2,0.189,0.201,0.215,...
```

要求：
- 第1列：类别标签
- 第2列：样本文件名
- 第3列起：各波段反射率值
- 列名应为波长值（如"900.00"）

**最后更新**: 2025年12月16日  
**文档版本**: v1.0
