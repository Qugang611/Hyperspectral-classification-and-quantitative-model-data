"""
高光谱数据神经网络模型验证脚本

功能描述：
    加载已训练好的神经网络模型，在相同的数据集上进行验证，复现训练时的测试集结果。
    数据处理方式与训练代码完全一致，包括：
    1. 相同的数据加载和预处理流程
    2. 相同的交叉验证划分策略（使用相同的random_state=42）
    3. 使用训练时保存的scaler和label_encoder
    4. 生成详细的分类报告和混淆矩阵

验证模式：
    模式1：所有fold使用同一个最佳模型（model.pth）
        - 默认模式，所有fold均用最佳模型
        - 适用于只保存了最佳模型的情况
    模式2：只验证特定的fold
        - 使用 --fold N 参数指定（N为1-based）
        - 可与模式1组合使用

使用方法：
    # 验证所有fold（均使用最佳模型）
    python validate_SpectralNN.py --csv Data/Exp2/day14-NIR.csv `
                                   --experiment Spectral_Classification_Exp2/day14-NIR `
                                   --folds 5
    
    # 只验证第1个fold（使用最佳模型）
    python validate_SpectralNN.py --csv Data/Exp2/day14-NIR.csv `
                                   --experiment Spectral_Classification_Exp2/day14-NIR `
                                   --folds 5 --fold 1

输出内容：
    - 验证日志：models/[experiment_name]/validation_log_[timestamp].log
    - 每个fold的测试集准确率、分类报告、每类别准确率
    - 混淆矩阵可视化
    - 所有fold的平均准确率和标准差

注意事项：
    1. 确保使用与训练时完全相同的数据文件
    2. 交叉验证折数必须与训练时一致
    3. 只支持加载最佳模型（model.pth）

作者：Maggie Zhang
版本：v2.1
最后修改：2025年12月16日
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import gc
import argparse
import logging
import traceback
import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

import matplotlib
import matplotlib.pyplot as plt

from utils import (
    load_and_preprocess_data,
    set_global_seed,
    plot_confusion_matrix,
)

# 配置matplotlib中文字体支持
def configure_matplotlib_fonts():
    """配置matplotlib支持中文字体显示"""
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial', 'SimHei', 'DejaVu Sans', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
        
        return True
    except Exception as e:
        print(f"配置matplotlib中文字体时出错: {e}")
        return False

configure_matplotlib_fonts()


class SpectralNN(nn.Module):
    """神经网络模型定义，与训练代码完全一致"""
    def __init__(self, input_dim, num_classes, hidden_dims=None, dropout_rate=0.3):
        super(SpectralNN, self).__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout_rate))
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def load_trained_model(experiment_name='default_exp', fold_idx=None):
    """
    加载训练好的模型bundle
    
    Args:
        experiment_name: 实验名称，用于定位模型文件
        fold_idx: 折索引（0-based），如果为None则加载最佳模型
        
    Returns:
        model_bundle: 包含model、scaler、label_encoder、device的字典
    """
    model_dir = f'models/{experiment_name}/Neural_Network'
    
    # 根据fold_idx确定模型文件路径
    if fold_idx is not None:
        # 加载特定fold的模型（如果存在）
        model_path = f'{model_dir}/model_fold{fold_idx+1}.pth'
        if not os.path.exists(model_path):
            logging.warning(f"Fold {fold_idx+1} 的模型文件不存在: {model_path}")
            logging.warning("将使用最佳模型 model.pth")
            model_path = f'{model_dir}/model.pth'
    else:
        # 加载最佳模型
        model_path = f'{model_dir}/model.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    logging.info(f"从 {model_path} 加载模型...")
    
    # 加载完整的model bundle
    # 注意：PyTorch 2.6+ 默认 weights_only=True，需要设置为 False 以加载完整的模型对象
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_bundle = torch.load(model_path, map_location=device, weights_only=False)
    
    # 确保模型在正确的设备上
    model_bundle['model'].to(device)
    model_bundle['device'] = device
    
    logging.info("模型加载成功")
    logging.info(f"设备: {device}")
    logging.info(f"模型类型: {type(model_bundle['model']).__name__}")
    logging.info(f"标签编码器类别: {model_bundle['label_encoder'].classes_}")
    
    return model_bundle


def validate_on_folds(X, y, original_label_encoder, n_splits=5, experiment_name='default_exp', 
                      specific_fold=None):
    """
    在交叉验证的测试集上验证模型，所有fold均使用同一个最佳模型
    
    Args:
        X: 特征数据
        y: 标签数据
        original_label_encoder: 原始标签编码器
        n_splits: 交叉验证折数
        experiment_name: 实验名称
        specific_fold: 只验证特定的fold（1-based），None表示验证所有fold
    Returns:
        fold_scores: 各折的准确率列表
    """
    logging.info(f"使用分层{n_splits}折交叉验证进行模型验证...")
    logging.info("注意：所有fold均使用同一个最佳模型 (model.pth)")
    logging.info("random_state=42确保数据划分一致")
    
    if specific_fold is not None:
        logging.info(f"仅验证 Fold {specific_fold}")
    else:
        logging.info(f"验证所有fold")
    
    # 使用与训练时完全相同的参数进行数据划分
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    
    # 只加载一次最佳模型
    try:
        model_bundle = load_trained_model(experiment_name=experiment_name, fold_idx=None)
    except FileNotFoundError as e:
        logging.error(f"无法加载最佳模型: {e}")
        return []
    
    device = model_bundle['device']
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    label_encoder = model_bundle['label_encoder']
    model.eval()
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # 如果指定了特定fold，跳过其他fold
        if specific_fold is not None and fold_idx + 1 != specific_fold:
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"验证 Fold {fold_idx+1}/{n_splits}")
        logging.info(f"{'='*60}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 保持与训练时一致的验证集划分
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        X_test_s = scaler.transform(X_test)
        test_ds = TensorDataset(torch.FloatTensor(X_test_s), torch.LongTensor(y_test))
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        
        y_pred_list = []
        y_true_list = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                _, pred = torch.max(out, 1)
                y_pred_list.append(pred.cpu().numpy())
                y_true_list.append(yb.cpu().numpy())
        y_pred = np.concatenate(y_pred_list)
        y_true = np.concatenate(y_true_list)
        test_acc = float((y_pred == y_true).mean())
        fold_scores.append(test_acc)
        logging.info(f"\nFold {fold_idx+1} 测试集评估结果:")
        logging.info(f"测试集样本数: {len(y_test)}")
        logging.info(f"测试集准确率: {test_acc:.4f}")
        logging.info(f"\n测试集分类报告:")
        logging.info(classification_report(y_true, y_pred, zero_division=0))
        unique_classes = np.unique(y_true)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=unique_classes, zero_division=0
        )
        logging.info("\n类别详细统计:")
        logging.info("类别\tprecision\trecall\tf1-score\tsupport\t正确数\t准确率")
        for idx, cls in enumerate(unique_classes):
            total = support[idx]
            correct = np.sum((y_true == cls) & (y_pred == cls))
            acc_cls = correct / total if total > 0 else 0.0
            if original_label_encoder is not None and cls < len(original_label_encoder.classes_):
                cls_name = original_label_encoder.classes_[cls]
            else:
                cls_name = str(cls)
            logging.info(f"{cls_name}\t{precision[idx]:.2f}\t{recall[idx]:.2f}\t{f1[idx]:.2f}\t{total}\t{correct}\t{acc_cls:.4f}")
        overall_acc = np.mean(y_true == y_pred)
        logging.info(f"\n总体准确率: {overall_acc:.4f}")
        try:
            plot_confusion_matrix(
                model_bundle, X_test, y_test, 
                f"Neural_Network_Validation_Fold{fold_idx+1}", 
                f"{experiment_name}/Neural_Network"
            )
            logging.info(f"混淆矩阵已保存")
        except Exception as e:
            logging.error(f"绘制混淆矩阵失败: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if fold_scores:
        avg_acc = float(np.mean(fold_scores))
        std_acc = float(np.std(fold_scores))
        logging.info(f"\n{'='*60}")
        logging.info(f"验证总结")
        logging.info(f"{'='*60}")
        logging.info(f"验证的fold数: {len(fold_scores)}")
        logging.info(f"所有fold准确率: {fold_scores}")
        logging.info(f"平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")
        logging.info(f"最高准确率: {max(fold_scores):.4f}")
        logging.info(f"最低准确率: {min(fold_scores):.4f}")
    else:
        logging.warning("没有成功验证任何fold")
    return fold_scores


def setup_logging(log_file=None, experiment_name='default_exp'):
    """设置日志记录"""
    if log_file is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"models/{experiment_name}/validation_log_{timestamp}.log"
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    ch = logging.StreamHandler(sys.stdout)
    
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # 降低字体相关警告
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    
    return logger, log_file


def main():
    parser = argparse.ArgumentParser(description='神经网络模型验证脚本')
    parser.add_argument('--csv', type=str, default='Data/Exp2/day14-NIR.csv', 
                        help='CSV数据路径（必须与训练时使用的数据相同）')
    parser.add_argument('--experiment', type=str, default='Spectral_Classification_Exp2/day14-NIR', 
                        help='实验名称（用于定位模型文件）')
    parser.add_argument('--folds', type=int, default=5, 
                        help='交叉验证折数（必须与训练时相同）')
    parser.add_argument('--fold', type=int, default=None, 
                        help='只验证特定的fold（1-based），不指定则验证所有fold')
    args = parser.parse_args()
    
    # 设置随机种子，与训练时保持一致
    set_global_seed(42)
    
    # 设置日志
    logger, log_file = setup_logging(experiment_name=args.experiment)
    font_configured = configure_matplotlib_fonts()
    
    logging.info("="*60)
    logging.info("神经网络模型验证脚本")
    logging.info("="*60)
    logging.info(f"Python版本: {sys.version}")
    logging.info(f"工作目录: {os.getcwd()}")
    logging.info(f"matplotlib字体配置: {'成功' if font_configured else '失败'}")
    logging.info(f"数据文件: {args.csv}")
    logging.info(f"实验名称: {args.experiment}")
    logging.info(f"交叉验证折数: {args.folds}")
    if args.fold:
        logging.info(f"验证fold: {args.fold}")
    else:
        logging.info(f"验证fold: 所有fold")
    logging.info(f"日志文件: {log_file}")
    
    try:
        # 加载数据（与训练时相同的方式）
        logging.info(f"\n从 {args.csv} 加载数据...")
        X, y, label_encoder, filenames = load_and_preprocess_data(args.csv)
        logging.info(f"数据加载完成: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
        logging.info(f"类别数: {len(np.unique(y))}")
        logging.info(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # 在交叉验证的测试集上进行验证
        fold_scores = validate_on_folds(
            X, y, label_encoder, 
            n_splits=args.folds, 
            experiment_name=args.experiment,
            specific_fold=args.fold
        )
        
        logging.info(f"\n验证完成！")
        logging.info(f"结果已保存到: {log_file}")
        
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}")
        logging.error("请确保已经训练过模型，并且实验名称正确")
    except Exception as e:
        logging.error(f"验证失败: {e}")
        traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
