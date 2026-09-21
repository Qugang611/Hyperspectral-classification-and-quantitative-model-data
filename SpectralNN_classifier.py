"""
高光谱数据神经网络分类器

功能描述：
    高光谱数据的神经网络分类，支持完整的训练、评估和可视化流程：
    1. 自动加载和预处理高光谱反射率数据（CSV格式）
    2. 神经网络模型（SpectralNN）支持多层结构和Dropout
    3. 分层5折交叉验证训练与评估
    4. 自动处理类别不平衡（类别权重）
    5. 支持早停、学习率调度、噪声增强等高级训练技巧
    6. 生成详细的分类报告、每类别准确率、混淆矩阵和损失曲线
    7. 支持模型保存与加载，避免重复训练
    8. 日志记录与错误处理机制完善

主要流程：
    - 数据加载与预处理
    - 神经网络模型定义与训练
    - 交叉验证评估（训练集、验证集、测试集均输出分类报告和每类别准确率）
    - 混淆矩阵和损失曲线绘制
    - 模型保存

输入数据格式：
    - CSV文件，UTF-8编码
    - 第1列：类别标签（支持文本和数字）
    - 第2列：文件名标识
    - 第11列及以后：各波段反射率值（列名为波长值）

输出内容：
    - 训练日志：models/[experiment_name]/training_log_[timestamp].log
    - 神经网络模型文件：models/[experiment_name]/Neural_Network/ 下的.pth文件
    - 可视化图表：plot/[experiment_name]/Neural_Network/ 下的PDF文件
      * 混淆矩阵：confusion_matrix_Neural_Network_FoldX.pdf
      * 损失曲线、训练指标
    - 分类报告：包含精确率、召回率、F1分数、support、正确数、准确率等

依赖库：
    - numpy, pandas: 数据处理
    - scikit-learn: 数据划分、评估
    - torch: 神经网络训练
    - matplotlib, seaborn: 可视化
    - utils: 项目自定义工具函数

作者：Maggie Zhang
版本：v2.0
最后修改：2025年9月
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import gc
import time
import argparse
import logging
import traceback
import datetime
import random
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm

from utils import (
    load_config,
    load_and_preprocess_data,
    print_class_accuracy,
    set_global_seed,
    plot_confusion_matrix,
)
from utils.loss_plot_utils import plot_loss_curves, plot_training_metrics
# 配置matplotlib中文字体支持
def configure_matplotlib_fonts():
    """配置matplotlib支持中文字体显示"""
    try:
        # 设置中文字体优先使用微软雅黑，英文使用Arial
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial', 'SimHei', 'DejaVu Sans', 'Liberation Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        
        # 设置默认字体编码
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['pdf.fonttype'] = 42  # 使用TrueType字体嵌入PDF
        matplotlib.rcParams['ps.fonttype'] = 42   # 使用TrueType字体嵌入PS
        
        # 抑制matplotlib的字体警告
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
        
        return True
    except Exception as e:
        print(f"配置matplotlib中文字体时出错: {e}")
        return False

# 配置字体
configure_matplotlib_fonts()

class SpectralNN(nn.Module):
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

def train_and_evaluate_nn(X, y, original_label_encoder=None, n_splits=1, experiment_name='default_exp', config: dict = None):
    logging.info(f"使用分层{n_splits}折交叉验证训练神经网络...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(np.unique(y))

    # 从配置读取超参
    config = config or {}
    hidden_dims = config.get('hidden_dims', [1024, 512, 256])
    dropout_rate = float(config.get('dropout_rate', 0.3))
    batch_size = int(config.get('batch_size', 128))
    val_ratio = float(config.get('val_ratio', 0.2))
    use_class_weight = bool(config.get('use_class_weight', True))
    noise_std = float(config.get('noise_std', 0.0))
    patience = int(config.get('patience', 30))
    max_epochs = int(config.get('max_epochs', 500))
    drop_last = bool(config.get('drop_last', True))
    opt_cfg = config.get('optimizer', {})
    lr = float(opt_cfg.get('lr', 5e-4))
    weight_decay = float(opt_cfg.get('weight_decay', 1e-5))
    sched_cfg = config.get('scheduler', {})
    sched_type = (sched_cfg.get('type', 'cosine') or 'cosine').lower()
    T_0 = int(sched_cfg.get('T_0', 30))
    T_mult = int(sched_cfg.get('T_mult', 1))
    eta_min = float(sched_cfg.get('eta_min', 1e-5))
    per_batch = bool(sched_cfg.get('per_batch', True))

    # 标签编码器
    if np.issubdtype(np.array(y).dtype, np.number):
        label_encoder = original_label_encoder
        if label_encoder is None:
            label_encoder = LabelEncoder()
            label_encoder.fit(np.unique(y))
    else:
        label_encoder = original_label_encoder or LabelEncoder().fit(np.unique(y))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    nn_fold_scores = []
    best_model_bundle = None
    best_score = -1
    all_fold_losses = []  # 存储所有折的损失历史

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logging.info(f"Fold {fold_idx+1}/{n_splits}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 再划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_ratio, random_state=42, stratify=y_train
        )

        # 标准化
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        # 张量
        train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val_s), torch.LongTensor(y_val))
        test_ds = TensorDataset(torch.FloatTensor(X_test_s), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size)

        model = SpectralNN(X.shape[1], num_classes, hidden_dims=hidden_dims, dropout_rate=dropout_rate).to(device)

        # 损失函数
        if use_class_weight:
            try:
                unique_classes = np.unique(y_train)
                class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
                criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
            except Exception as e:
                logging.error(f"类别权重计算失败: {e}")
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if sched_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        elif sched_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=max(5, patience // 2))
        else:
            scheduler = None

        best_val_loss = float('inf')
        best_state = None
        patience_cnt = 0
        
        # 记录损失历史
        train_losses = []  # 存储带噪声的训练损失（用于反向传播）
        clean_train_losses = []  # 存储纯净的训练损失（用于绘图）
        val_losses = []
        learning_rates = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(max_epochs):
            model.train()
            running = 0.0
            for i, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)
                # 训练扰动
                if noise_std > 0:
                    xb = xb + torch.randn_like(xb) * noise_std
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                # 按batch推进（仅cosine且per_batch时）
                if scheduler is not None and sched_type == 'cosine' and per_batch:
                    t_epoch = epoch + (i + 1) / max(1, len(train_loader))
                    scheduler.step(t_epoch)
                running += loss.item() * xb.size(0)
            train_loss = running / len(train_ds)

            # 计算训练集准确率（不加噪声）
            model.eval()
            train_correct = 0
            train_total = 0
            clean_train_loss = 0.0
            with torch.no_grad():
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    clean_train_loss += loss.item() * xb.size(0)
                    _, pred = torch.max(out, 1)
                    train_total += yb.size(0)
                    train_correct += (pred == yb).sum().item()
            train_acc = train_correct / train_total
            clean_train_loss /= len(train_ds)

            # 计算验证集损失和准确率
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    loss = criterion(out, yb)
                    val_loss += loss.item() * xb.size(0)
                    _, pred = torch.max(out, 1)
                    val_total += yb.size(0)
                    val_correct += (pred == yb).sum().item()
            val_loss /= len(val_ds)
            val_acc = val_correct / val_total
            
            # 记录指标
            train_losses.append(train_loss)  # 带噪声的训练损失
            clean_train_losses.append(clean_train_loss)  # 纯净的训练损失
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            learning_rates.append(optimizer.param_groups[0]['lr'])

            # 非per-batch或plateau的调度
            if scheduler is not None:
                if sched_type == 'cosine' and not per_batch:
                    scheduler.step(epoch)
                elif sched_type == 'plateau':
                    scheduler.step(val_loss)

            if epoch % 10 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                # 添加训练/验证损失比较的诊断信息
                clean_val_ratio = val_loss / clean_train_loss if clean_train_loss > 0 else 1.0
                noisy_val_ratio = val_loss / train_loss if train_loss > 0 else 1.0
                
                logging.info(f"Fold {fold_idx+1} Epoch {epoch+1}/{max_epochs}")
                logging.info(f"  损失: 训练(带噪) {train_loss:.4f}, 训练(纯净) {clean_train_loss:.4f}, 验证 {val_loss:.4f}")
                logging.info(f"  准确率: 训练 {train_acc:.3f}, 验证 {val_acc:.3f}, LR {cur_lr:.6f}")
                if noise_std > 0:
                    logging.info(f"  验证/纯净训练损失比: {clean_val_ratio:.3f} (正常应>1.0)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    logging.info(f"早停: 第 {epoch+1} 轮")
                    break

        # 加载最佳参数
        if best_state is not None:
            model.cpu(); model.load_state_dict(best_state); model.to(device)

        # 分析训练过程
        val_lower_count = sum(1 for t, v in zip(train_losses, val_losses) if v < t)
        total_epochs = len(train_losses)
        val_lower_percentage = (val_lower_count / total_epochs) * 100
        
        final_train_loss = train_losses[-1] if train_losses else 0
        final_val_loss = val_losses[-1] if val_losses else 0
        min_val_loss = min(val_losses) if val_losses else 0
        min_train_loss = min(train_losses) if train_losses else 0
        
        logging.info(f"Fold {fold_idx+1} 训练分析:")
        logging.info(f"  验证损失更低的轮次: {val_lower_count}/{total_epochs} ({val_lower_percentage:.1f}%)")
        logging.info(f"  最终损失 - 训练: {final_train_loss:.4f}, 验证: {final_val_loss:.4f}")
        logging.info(f"  最小损失 - 训练: {min_train_loss:.4f}, 验证: {min_val_loss:.4f}")
        logging.info(f"  噪声强度: {noise_std}")
        logging.info(f"  Dropout率: {dropout_rate}")
        logging.info(f"  最佳验证轮次: {len(train_losses) - patience_cnt if patience_cnt < patience else len(train_losses)}")
        
        # 统一评估函数
        def _eval_split(tag, data_loader):
            model.eval()
            _preds, _true = [], []
            with torch.no_grad():
                for xb, yb in data_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    _, pred = torch.max(out, 1)
                    _preds.append(pred.cpu().numpy())
                    _true.append(yb.cpu().numpy())
            y_pred = np.concatenate(_preds) if _preds else np.array([])
            y_true = np.concatenate(_true) if _true else np.array([])
            if y_true.size > 0:
                acc = float((y_pred == y_true).mean())
                logging.info(f"{tag} 准确率: {acc:.4f}")
                logging.info(f"{tag} 分类报告:\n" + classification_report(y_true, y_pred, zero_division=0))
                # 输出每类别准确率，格式与主脚本一致
                from sklearn.metrics import precision_recall_fscore_support
                unique_classes = np.unique(y_true)
                precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=unique_classes, zero_division=0)
                logging.info("类别\tprecision\trecall\tf1-score\tsupport\t正确数\t准确率")
                for idx, cls in enumerate(unique_classes):
                    mask = (y_true == cls)
                    total = support[idx]
                    correct = np.sum((y_true == cls) & (y_pred == cls))
                    acc_cls = correct / total if total > 0 else 0.0
                    if original_label_encoder is not None and cls < len(original_label_encoder.classes_):
                        cls_name = original_label_encoder.classes_[cls]
                    else:
                        cls_name = str(cls)
                    logging.info(f"{cls_name}\t{precision[idx]:.2f}\t{recall[idx]:.2f}\t{f1[idx]:.2f}\t{total}\t{correct}\t{acc_cls:.4f}")
                overall_acc = np.mean(y_true == y_pred)
                logging.info(f"总体准确率: {overall_acc:.4f}")
            else:
                acc = 0.0
                logging.warning(f"{tag} 数据为空，跳过评估")
            return acc, y_true, y_pred

        # 分别评估训练集、验证集与测试集
        _eval_split('训练集', train_loader)
        _eval_split('验证集', val_loader)
        test_acc, y_true, y_pred = _eval_split('测试集', test_loader)

        # 记录折分数（基于测试集）
        nn_fold_scores.append(test_acc)
        logging.info(f"Fold {fold_idx+1} 准确率: {test_acc:.4f}")

        # 保存当前折的损失历史（使用纯净训练损失）
        all_fold_losses.append((clean_train_losses, val_losses))
        
        # 绘制当前折的损失曲线（使用纯净训练损失）
        # 图表保存在 {experiment_name}/Neural_Network/ 目录下
        plot_loss_curves(clean_train_losses, val_losses, fold_idx+1, f"{experiment_name}/Neural_Network")
        
        # 绘制当前折的训练指标
        metrics_history = {
            '损失': {'训练损失': clean_train_losses, '验证损失': val_losses},
            '准确率': {'训练准确率': train_accuracies, '验证准确率': val_accuracies},
            '学习率': learning_rates
        }
        try:
            plot_training_metrics(metrics_history, f"{experiment_name}/Neural_Network/fold_{fold_idx+1}.pdf")
        except Exception as e:
            logging.error(f"绘制 Fold {fold_idx+1} 训练指标失败: {e}")

        if test_acc > best_score:
            best_score = test_acc
            best_model_bundle = {
                'model': model,
                'device': device,
                'scaler': scaler,
                'label_encoder': label_encoder
            }

        # 清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 保存每折混淆矩阵到 {experiment_name}/Neural_Network/ 目录
        try:
            plot_confusion_matrix(best_model_bundle, X_test, y_test, f"Neural_Network_Fold{fold_idx+1}", f"{experiment_name}/Neural_Network")
        except Exception as e:
            logging.error(f"绘制第{fold_idx+1}折混淆矩阵失败: {e}")

    avg = float(np.mean(nn_fold_scores)) if nn_fold_scores else 0.0
    std = float(np.std(nn_fold_scores)) if nn_fold_scores else 0.0
    logging.info(f"NN {n_splits}折平均准确率: {avg:.4f} ± {std:.4f}")

    # # 绘制所有折的损失汇总图
    # if all_fold_losses:
    #     plot_overall_loss_summary(all_fold_losses, experiment_name)

    return best_model_bundle, best_score


def save_nn_model(bundle, experiment_name='default_exp'):
    """保存神经网络模型到独立文件夹，与spectral_classifier.py保持一致"""
    # 为神经网络创建独立子目录
    model_dir = f'models/{experiment_name}/Neural_Network'
    os.makedirs(model_dir, exist_ok=True)
    
    model = bundle['model']
    scaler = bundle['scaler']
    label_encoder = bundle.get('label_encoder')

    torch.save(model.state_dict(), f'{model_dir}/nn_model.pth')
    torch.save(scaler, f'{model_dir}/nn_scaler.pth')
    torch.save(label_encoder, f'{model_dir}/nn_label_encoder.pth')
    info = {
        'input_dim': model.model[0].in_features if hasattr(model, 'model') else None,
        'num_classes': model.model[-1].out_features if hasattr(model, 'model') else None
    }
    torch.save(info, f'{model_dir}/nn_model_info.pth')
    logging.info(f"神经网络模型已保存到 '{model_dir}/'")
    
    # 额外保存完整模型以便加载（与spectral_classifier.py一致）
    torch.save(bundle, f'{model_dir}/model.pth')
    logging.info(f"完整模型bundle已保存到 '{model_dir}/model.pth'")


def setup_logging(log_file=None, experiment_name='default_exp'):
    if log_file is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"models/{experiment_name}/training_log_{timestamp}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    # 降低字体子集化相关的所有logger级别
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
    logging.getLogger('fontTools.merge').setLevel(logging.ERROR)
    logging.getLogger('fontTools.ttLib').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.backends._backend_pdf').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    return logger, log_file


def main():
    parser = argparse.ArgumentParser(description='仅神经网络的光谱分类器')
    parser.add_argument('--csv', type=str, default='Data/Exp2/day14-NIR.csv', help='CSV数据路径')
    parser.add_argument('--experiment', type=str, default='Spectral_Classification_Exp2/day14-NIR', help='实验名称')
    parser.add_argument('--folds', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--config', type=str, default='configs/SpectralNN_config.json', help='JSON配置文件路径')
    args = parser.parse_args()

    set_global_seed(42)

    logger, log_file = setup_logging(experiment_name=args.experiment)
    font_configured = configure_matplotlib_fonts()
    logging.info("=========================================")
    logging.info("开始执行光谱分类任务 - 神经网络")
    logging.info(f"Python版本: {sys.version}")
    logging.info(f"工作目录: {os.getcwd()}")
    logging.info(f"matplotlib字体配置: {'成功' if font_configured else '失败'}")
    logging.info(f"模型保存路径: models/{args.experiment}/Neural_Network/")
    logging.info(f"图表保存路径: plot/{args.experiment}/Neural_Network/")

    try:
        cfg = load_config(args.config)
        X, y, label_encoder, filenames = load_and_preprocess_data(args.csv)
        best_bundle, best_score = train_and_evaluate_nn(X, y, label_encoder, n_splits=args.folds, experiment_name=args.experiment, config=cfg)
        if best_bundle is not None:
            # 全量数据混淆矩阵，保存在 {experiment_name}/Neural_Network/ 下
            try:
                plot_confusion_matrix(best_bundle, X, y, 'Neural_Network', f"{args.experiment}/Neural_Network")
            except Exception as e:
                logging.error(f"绘制最终混淆矩阵失败: {e}")
            save_nn_model(best_bundle, experiment_name=args.experiment)
            logging.info(f"完成。最佳折准确率: {best_score:.4f}")
        else:
            logging.error("训练失败，未获得可用模型")
    except Exception as e:
        logging.error(f"执行失败: {e}")
        traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
