"""
高光谱数据机器学习模型验证脚本

功能描述：
    加载已训练好的机器学习模型（LogisticRegression、Random Forest、LDA、XGBoost），
    在相同的数据集和数据处理方式下进行验证，复现训练时的测试集结果。
    
特性：
    1. 相同的数据加载和预处理流程
    2. 相同的交叉验证划分策略（random_state=42）
    3. 默认验证所有fold并报告详细结果
    4. 可选只验证特定fold
    5. 生成详细的分类报告和混淆矩阵
    
重要说明：
    训练时保存的模型是在交叉验证中表现最好的那一折的模型。
    验证所有fold时：
    - 最佳fold（如第2折）：应该完全复现训练时的结果
    - 其他fold：使用最佳模型评估，提供模型泛化性能的参考
    如果只想验证最佳fold的结果，请使用 --fold N 指定（从训练日志查看最佳fold编号）

用法示例：
    # 验证所有fold（最佳fold会复现训练结果，其他fold提供泛化性能参考）
    python validate_spectral_ML.py --csv Data/Exp2/day14-NIR.csv `
                                   --experiment Spectral_Classification_Exp2/day14-NIR `
                                   --model LDA --folds 5
    
    # 只验证第3个fold（假设训练时最佳fold是3，可从训练日志查看）
    python validate_spectral_ML.py --csv Data/Exp2/day14-NIR.csv `
                                   --experiment Spectral_Classification_Exp2/day14-NIR `
                                   --model LDA --folds 5 --fold 3

输出内容：
    - 验证日志：models/[experiment_name]/validation_log_{model_name}_{timestamp}.log
    - 每个fold的测试集准确率、分类报告、每类别统计（含准确率）
    - 混淆矩阵可视化（PDF）
    - 所有fold的平均准确率和标准差（如果验证多个fold）

注意事项：
    1. 确保使用与训练时完全相同的数据文件
    2. 交叉验证折数必须与训练时一致
    3. 模型保存路径：models/{experiment_name}/{model_name}/{model_name}.pth
    4. 每个类别的"准确率"列 = 该类正确数/该类总数（即召回率）

作者：Maggie Zhang
版本：v3.0
最后修改：2025年12月16日
"""
import os
import sys
import gc
import argparse
import logging
import traceback
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import torch
from utils.plot_setting import setfig

def configure_matplotlib_fonts():
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

def setup_logging(log_file=None, experiment_name='default_exp', model_name='model'):
    if log_file is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"models/{experiment_name}/validation_log_{model_name}_{timestamp}.log"
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
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    return logger, log_file

def load_and_preprocess_data(csv_file):
    df = pd.read_csv(csv_file, encoding='utf-8')
    label_column = df.columns[0]
    filename_column = df.columns[1]
    label_encoder = None
    if not np.issubdtype(df[label_column].dtype, np.number):
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[label_column])
    else:
        y = df[label_column].values
    feature_columns = df.columns[10:]
    try:
        wavelengths = [float(col) for col in feature_columns]
    except ValueError:
        wavelengths = list(range(len(feature_columns)))
    X = df[feature_columns].values
    filenames = df[filename_column].values if filename_column in df.columns else None
    return X, y, wavelengths, label_encoder, filenames

def plot_confusion_matrix(model, X, y, model_name, label_encoder=None, experiment_name=None):
    base_model_name = model_name.split('_Fold')[0] if '_Fold' in model_name else model_name
    plot_dir = f'plot/{experiment_name}/{base_model_name.replace(" ", "_")}'
    os.makedirs(plot_dir, exist_ok=True)
    y_pred = model.predict(X)
    y_pred_original = y_pred
    if label_encoder is not None:
        if np.issubdtype(y.dtype, np.number) and min(y) == 0 and max(y) < len(label_encoder.classes_):
            y_original = label_encoder.inverse_transform(y)
        else:
            y_original = y
    else:
        y_original = y
    if isinstance(y_original[0], (int, np.integer)):
        if label_encoder is not None:
            y_original = [str(label_encoder.classes_[i]) if i < len(label_encoder.classes_) else f"Unknown_{i}" for i in y_original]
        else:
            y_original = [str(i) for i in y_original]
    if isinstance(y_pred_original[0], (int, np.integer)):
        if label_encoder is not None:
            y_pred_original = [str(label_encoder.classes_[i]) if i < len(label_encoder.classes_) else f"Unknown_{i}" for i in y_pred_original]
        else:
            y_pred_original = [str(i) for i in y_pred_original]
    cm = confusion_matrix(y_original, y_pred_original, labels=np.unique(y_original))
    labels = np.unique(y_original)
    fig = setfig(column=3, x=6, y=5.5)
    import seaborn as sns
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=labels, 
        yticklabels=labels,
        annot_kws={"size": 8, "weight": "bold", "family": "sans-serif", "color": "black"}
    )
    plt.xticks(rotation=30, fontfamily='Microsoft YaHei', weight='bold')
    plt.yticks(fontfamily='Microsoft YaHei', weight='bold')
    plt.xlabel('预测标签', fontfamily='Microsoft YaHei', weight='bold')
    plt.ylabel('真实标签', fontfamily='Microsoft YaHei', weight='bold')
    plt.title(f'{model_name} confusion matrix', fontfamily='Microsoft YaHei', weight='bold')
    plt.savefig(f'{plot_dir}/confusion_matrix_{model_name}_Validation.pdf', 
                   format='PDF', 
                   transparent=False, 
                   bbox_inches='tight',
                   dpi=300,
                   facecolor='white', 
                   edgecolor='none',
                   metadata={'Creator': 'Spectral Classifier'})
    plt.close()
    logging.info(f"已保存 {model_name} 的混淆矩阵图到 '{plot_dir}'")

def load_trained_model(experiment_name, model_name):
    """加载训练好的模型"""
    model_dir = f'models/{experiment_name}/{model_name}'
    model_path = f'{model_dir}/{model_name}.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    logging.info(f"从 {model_path} 加载模型...")
    model = torch.load(model_path, weights_only=False)
    logging.info("模型加载成功")
    return model

def validate_on_folds(X, y, label_encoder, n_splits, experiment_name, model_name, specific_fold=None):
    """
    在交叉验证的测试集上验证模型
    
    Args:
        X: 特征数据
        y: 标签数据
        label_encoder: 标签编码器
        n_splits: 交叉验证折数
        experiment_name: 实验名称
        model_name: 模型名称
        specific_fold: 如果指定（1-based），只验证该fold；否则验证所有fold
        
    Returns:
        fold_scores: 各折的准确率列表
    """
    logging.info(f"开始模型验证...")
    logging.info(f"使用分层{n_splits}折交叉验证（random_state=42，与训练时一致）")
    
    logging.info(f"\n重要说明：")
    logging.info(f"训练时保存的模型是在{n_splits}折交叉验证中表现最好的那一折的模型。")
    logging.info(f"验证所有fold时：")
    logging.info(f"  - 最佳fold：应该完全复现训练时的结果")
    logging.info(f"  - 其他fold：使用最佳模型评估，提供模型泛化性能的参考")
    logging.info(f"如果只想验证最佳fold，请使用 --fold N 指定（可从训练日志中查看最佳fold编号）\n")
    
    if specific_fold is not None:
        logging.info(f"验证模式：只验证 Fold {specific_fold}")
    else:
        logging.info(f"验证模式：验证所有 {n_splits} 个fold（使用训练时的最佳模型）")
    
    # 加载模型（所有fold共用同一个最佳模型）
    try:
        model = load_trained_model(experiment_name, model_name)
    except FileNotFoundError as e:
        logging.error(str(e))
        return []
    
    # 使用与训练时完全相同的参数进行数据划分
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        fold_num = fold_idx + 1
        
        # 如果指定了特定fold，跳过其他fold
        if specific_fold is not None and fold_num != specific_fold:
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"验证 Fold {fold_num}/{n_splits}")
        logging.info(f"{'='*60}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 预测（注意：我们直接在测试集上评估，不需要再划分验证集）
        y_pred = model.predict(X_test)
        test_acc = float((y_pred == y_test).mean())
        fold_scores.append(test_acc)
        
        # 输出基本信息
        logging.info(f"测试集样本数: {len(y_test)}")
        logging.info(f"测试集准确率: {test_acc:.4f}")
        
        # 分类报告
        logging.info(f"\n测试集分类报告:")
        logging.info(classification_report(y_test, y_pred, zero_division=0))
        
        # 每类别详细统计
        unique_classes = np.unique(y_test)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=unique_classes, zero_division=0
        )
        
        logging.info("\n类别详细统计:")
        logging.info("类别\tprecision\trecall\tf1-score\tsupport\t正确数\t准确率")
        total_correct = 0
        total_samples = 0
        
        for idx, cls in enumerate(unique_classes):
            total = support[idx]
            correct = np.sum((y_test == cls) & (y_pred == cls))
            accuracy = correct / total if total > 0 else 0.0
            total_correct += correct
            total_samples += total
            
            if label_encoder is not None and cls < len(label_encoder.classes_):
                cls_name = label_encoder.classes_[cls]
            else:
                cls_name = str(cls)
            
            logging.info(f"{cls_name}\t{precision[idx]:.2f}\t{recall[idx]:.2f}\t{f1[idx]:.2f}\t{total}\t{correct}\t{accuracy:.4f}")
        
        logging.info(f"\n验证：所有类别正确数之和/总样本数 = {total_correct}/{total_samples} = {total_correct/total_samples:.4f}")
        
        # 绘制混淆矩阵
        try:
            plot_confusion_matrix(model, X_test, y_test, 
                                f"{model_name}_Fold{fold_num}", 
                                label_encoder, experiment_name)
        except Exception as e:
            logging.error(f"绘制混淆矩阵失败: {e}")
        
        gc.collect()
    
    # 总结
    if len(fold_scores) == 0:
        logging.warning("没有成功验证任何fold")
    elif len(fold_scores) == 1:
        logging.info(f"\n{'='*60}")
        logging.info(f"验证完成")
        logging.info(f"{'='*60}")
        logging.info(f"Fold {specific_fold} 准确率: {fold_scores[0]:.4f}")
    else:
        avg_acc = float(np.mean(fold_scores))
        std_acc = float(np.std(fold_scores))
        
        logging.info(f"\n{'='*60}")
        logging.info(f"所有fold验证总结")
        logging.info(f"{'='*60}")
        logging.info(f"验证的fold数: {len(fold_scores)}")
        logging.info(f"各fold准确率: {[f'{score:.4f}' for score in fold_scores]}")
        logging.info(f"平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")
        logging.info(f"最高准确率: {max(fold_scores):.4f} (Fold {fold_scores.index(max(fold_scores))+1})")
        logging.info(f"最低准确率: {min(fold_scores):.4f} (Fold {fold_scores.index(min(fold_scores))+1})")
    
    return fold_scores

def main():
    parser = argparse.ArgumentParser(description='机器学习模型验证脚本')
    parser.add_argument('--csv', type=str, required=True, help='CSV数据路径（必须与训练时使用的数据相同）')
    parser.add_argument('--experiment', type=str, required=True, help='实验名称（用于定位模型文件）')
    parser.add_argument('--model', type=str, required=True, choices=['LogisticRegression', 'Random_Forest', 'LDA', 'XGBoost'], help='模型名称')
    parser.add_argument('--folds', type=int, default=5, help='交叉验证折数（必须与训练时相同）')
    parser.add_argument('--fold', type=int, default=None, help='验证特定的fold（1-based）。如果不指定，则验证所有fold')
    args = parser.parse_args()
    set_global_seed = lambda x: np.random.seed(x)
    set_global_seed(42)
    logger, log_file = setup_logging(experiment_name=args.experiment, model_name=args.model)
    font_configured = configure_matplotlib_fonts()
    logging.info("="*60)
    logging.info("机器学习模型验证脚本 v3.0")
    logging.info("="*60)
    logging.info(f"Python版本: {sys.version}")
    logging.info(f"工作目录: {os.getcwd()}")
    logging.info(f"matplotlib字体配置: {'成功' if font_configured else '失败'}")
    logging.info(f"数据文件: {args.csv}")
    logging.info(f"实验名称: {args.experiment}")
    logging.info(f"模型名称: {args.model}")
    logging.info(f"交叉验证折数: {args.folds}")
    if args.fold:
        logging.info(f"验证模式: 只验证Fold {args.fold}")
    else:
        logging.info(f"验证模式: 验证所有fold")
    logging.info(f"日志文件: {log_file}")
    try:
        logging.info(f"\n从 {args.csv} 加载数据...")
        X, y, wavelengths, label_encoder, filenames = load_and_preprocess_data(args.csv)
        logging.info(f"数据加载完成: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
        logging.info(f"类别数: {len(np.unique(y))}")
        logging.info(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")
        fold_scores = validate_on_folds(
            X, y, label_encoder, 
            n_splits=args.folds, 
            experiment_name=args.experiment,
            model_name=args.model,
            specific_fold=args.fold
        )
        logging.info(f"\n验证完成！")
        logging.info(f"结果已保存到: {log_file}")
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}")
        logging.error("请确保已经训练过模型，并且实验名称和模型名称正确")
    except Exception as e:
        logging.error(f"验证失败: {e}")
        traceback.print_exc()
    finally:
        gc.collect()

if __name__ == '__main__':
    main()
