"""
高光谱数据机器学习分类器

功能描述：
    本脚本是一个全面的高光谱数据分类工具，集成了4种机器学习算法：
    1. 自动加载和预处理高光谱反射率数据（CSV格式）
    2. 支持多种分类算法：逻辑回归、随机森林、线性判别分析、XGBoost
    3. 使用分层5折交叉验证进行模型评估和选择
    4. 自动处理类别不平衡问题（计算类别权重）
    5. 智能资源管理和GPU/CPU自适应配置
    6. 生成详细的分类报告、混淆矩阵和特征重要性图
    7. 支持模型保存和加载，避免重复训练
    8. 提供完整的日志记录和错误处理机制

作者：Maggie Zhang
创建日期：2025年9月
版本：v2.0 通过utils提高复用性
最后修改：2025年9月16日

支持的分类算法：
    - 逻辑回归：带PCA降维和L2正则化
    - 随机森林：集成学习，自动特征选择
    - 线性判别分析(LDA)：经典线性分类器
    - XGBoost：梯度提升树算法
    - 神经网络：PyTorch实现的深度学习模型

技术特性：
    - 分层5折交叉验证：确保每折中类别分布一致，避免类别不平衡问题
    - 智能数据采样：所有数据划分均使用分层抽样，保证类别比例一致性
    - 自动超参数优化：使用网格搜索找到的最佳参数
    - 智能资源管理：CPU/内存使用限制，防止系统过热
    - GPU自适应：自动检测和配置GPU，支持CUDA加速
    - 断点续传：支持训练中断后从检查点恢复
    - 批处理预测：处理大型数据集时分批进行预测
    - 中文字体支持：自动配置matplotlib中文字体，确保图表正确显示

使用方法：
    1. 准备数据：确保高光谱数据已处理为CSV格式
    2. 配置参数：修改main()函数中的csv_file路径
    3. 运行脚本：python spectral_classifier.py
    4. 可选参数：
       --retrain: 强制重新训练模型
       --val-size: 验证集比例（默认0.15）
       --test-size: 测试集比例（默认0.15）
       --folds: 交叉验证折数（默认5）
       --experiment: 实验名称，用于区分不同数据集（默认day1_VIS_raw）
    
    使用示例：
       python spectral_classifier.py --experiment day1_VIS_processed --folds 10 --retrain
       python spectral_classifier.py --experiment day2_NIR_raw --val-size 0.2

输入数据格式：
    - CSV文件，UTF-8编码
    - 第1列：类别标签（支持文本和数字）
    - 第2列：文件名标识
    - 第11列及以后：各波段反射率值（列名为波长值）
    
输出内容：
    - 训练日志：models/[experiment_name]/training_log_[timestamp].log
    - 模型文件：models/[experiment_name]/目录下的.pth文件
    - 可视化图表：plot/[experiment_name]/目录下的PDF文件
      * 混淆矩阵：confusion_matrix_[model_name].pdf
      * 特征重要性：feature_importance_Random_Forest.pdf
    - 分类报告：包含精确率、召回率、F1分数等指标

依赖库：
    核心库：
    - pandas, numpy: 数据处理和数值计算
    - scikit-learn: 传统机器学习算法
    - torch: PyTorch深度学习框架
    - matplotlib, seaborn: 数据可视化
    
    可选库：
    - xgboost: 梯度提升算法（可选）
    - psutil: 系统资源监控
    - plot_setting: 自定义绘图设置

系统要求：
    - Python 3.7+
    - 建议内存：8GB以上
    - GPU：NVIDIA显卡（可选，支持CUDA加速）
    - 存储：足够空间保存模型和日志文件
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import gc
import logging
import datetime
import traceback
import time
import sys
from contextlib import contextmanager
import psutil
import multiprocessing
import argparse
from utils.plot_setting import setfig
# 计算类别权重来处理不平衡数据
from sklearn.utils.class_weight import compute_class_weight
# 导入深拷贝，用于保存每个fold的模型
from copy import deepcopy

# 导入XGBoost，如果不可用则记录错误
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

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

# 设置安全加载的全局对象（仅在PyTorch 2.0+且支持该功能时使用）
try:
    import torch.serialization
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            Pipeline, 
            LogisticRegression, 
            RandomForestClassifier, 
            LDA, 
            StandardScaler, 
            PCA,
        ])
        print("已配置torch安全加载全局对象")
except (ImportError, AttributeError) as e:
    # PyTorch版本不支持此功能，使用传统方式加载
    print(f"torch.serialization不可用，将使用weights_only=False加载模型")

# 设置日志记录
def setup_logging(log_file=None, experiment_name="day1_VIS_raw"):
    """设置日志记录系统"""
    if log_file is None:
        # 使用时间戳创建唯一的日志文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"models/{experiment_name}/training_log_{timestamp}.log"
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 抑制fontTools的详细日志输出
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    logging.getLogger('fontTools.subset').setLevel(logging.WARNING)
    
    # 抑制matplotlib字体子集化相关警告
    logging.getLogger('matplotlib.backends._backend_pdf').setLevel(logging.ERROR)
    
    return logger, log_file

@contextmanager
def log_time(task_name):
    """计时上下文管理器，记录任务执行时间"""
    logging.info(f"开始任务: {task_name}")
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        logging.info(f"完成任务: {task_name}, 耗时: {end_time - start_time:.2f}秒")

def limit_cpu_usage():
    """限制CPU使用率，防止系统过热导致关机"""
    try:
        # 获取CPU核心数
        cpu_count = multiprocessing.cpu_count()
        logging.info(f"系统CPU核心数: {cpu_count}")
        
        # 获取当前进程
        current_process = psutil.Process(os.getpid())
        
        # 设置进程的CPU亲和性，限制使用的CPU核心
        # 使用总核心数的70%（至少保留一个核心给系统）
        cores_to_use = max(1, int(cpu_count * 0.7))
        
        # 在Windows上
        if sys.platform == 'win32':
            # 设置进程优先级为低于正常
            current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            logging.info(f"Windows系统: 已设置进程优先级为低于正常")
            
            # 设置CPU亲和性
            mask = sum(1 << i for i in range(cores_to_use))
            current_process.cpu_affinity([i for i in range(cores_to_use)])
            logging.info(f"已限制进程使用 {cores_to_use}/{cpu_count} 个CPU核心")
        # 在Linux/macOS上
        else:
            # 设置进程nice值(优先级)
            os.nice(10)  # 值越大，优先级越低
            logging.info(f"已设置进程nice值为10(优先级降低)")
        
        return True
    except Exception as e:
        logging.error(f"限制CPU使用率时出错: {str(e)}")
        return False

def monitor_resources(threshold_cpu=90, threshold_memory=90):
    """监控系统资源使用情况，如果超过阈值则主动降低资源使用"""
    try:
        import psutil
        
        # 获取CPU使用情况
        cpu_percent = psutil.cpu_percent(interval=0.5)
        
        # 获取内存使用情况
        virtual_memory = psutil.virtual_memory()
        memory_percent = virtual_memory.percent
        available_memory_gb = virtual_memory.available / (1024**3)
        total_memory_gb = virtual_memory.total / (1024**3)
        
        # 如果是Windows系统，获取页面文件使用情况
        if sys.platform == 'win32':
            swap = psutil.swap_memory()
            swap_percent = swap.percent
            swap_used_gb = swap.used / (1024**3)
            swap_total_gb = swap.total / (1024**3)
            swap_info = f"页面文件: {swap_used_gb:.2f}GB/{swap_total_gb:.2f}GB ({swap_percent}%)"
        else:
            swap_info = ""
        
        # 如果有GPU，获取GPU使用情况
        gpu_info = ""
        if 'torch' in sys.modules:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    if hasattr(torch.cuda, 'memory_allocated'):
                        allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
                        gpu_info += f"GPU #{i}: 已分配 {allocated:.2f}GB, 峰值 {max_allocated:.2f}GB; "
        
        # 记录资源使用情况
        logging.info(f"资源监控 - CPU: {cpu_percent}%, 内存: {virtual_memory.used/(1024**3):.2f}GB/{total_memory_gb:.2f}GB ({memory_percent}%), 可用: {available_memory_gb:.2f}GB, {swap_info} {gpu_info}")
        
        # 主动干预：如果CPU使用率超过阈值，强制进行垃圾回收并短暂休眠
        if cpu_percent > threshold_cpu:
            logging.warning(f"CPU使用率过高 ({cpu_percent}%)! 执行主动干预措施。")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # 暂停一小段时间，让CPU降温
            time.sleep(2)
            
            # 如果CPU持续高使用率，降低训练进程优先级
            if psutil.cpu_percent(interval=0.1) > threshold_cpu:
                try:
                    current_process = psutil.Process(os.getpid())
                    if sys.platform == 'win32':
                        current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    else:
                        os.nice(10)
                    logging.warning("已降低进程优先级以减少CPU使用率")
                except:
                    pass
            
        # 如果内存使用率过高，主动清理内存
        if memory_percent > threshold_memory:
            logging.warning(f"内存使用率过高 ({memory_percent}%)! 执行内存释放。")
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'available_memory_gb': available_memory_gb
        }
    except ImportError:
        logging.warning("无法导入psutil库，跳过资源监控")
        return None
    except Exception as e:
        logging.error(f"监控资源时出错: {str(e)}")
        return None

def configure_gpu():
    """配置GPU使用并防止内存溢出"""
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logging.info(f"找到 {device_count} 个 GPU:")
            for i in range(device_count):
                logging.info(f" - {torch.cuda.get_device_name(i)}")
                
            # 自动清理GPU缓存，减少内存占用
            torch.cuda.empty_cache()
            
            # 设置较小的初始内存使用量
            for i in range(device_count):
                # 将最大内存使用量设置为可用内存的90%
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(0.9, i)
            
            logging.info("已配置GPU，将在训练中自动管理内存使用")
            return True
        else:
            logging.info("未检测到GPU，将使用CPU进行计算")
            return False
    except Exception as e:
        logging.error(f"配置GPU时出错: {e}")
        logging.info("将使用CPU进行计算")
        return False

def load_and_preprocess_data(csv_file):
    """
    加载并预处理CSV数据
    
    Args:
        csv_file: CSV文件路径
    
    Returns:
        X: 特征数据
        y: 标签
        wavelengths: 波长列表
        label_encoder: 标签编码器，用于后续转换
        filenames: 文件名列表，用于结果导出
    """
    logging.info(f"加载数据文件: {csv_file}")
    
    df = pd.read_csv(csv_file, encoding='utf-8')
    logging.info(f"成功加载数据，共 {len(df)} 个样本")
    
    label_column = df.columns[0]
    filename_column = df.columns[1]
    
    # 检查标签类型并进行编码
    unique_labels = df[label_column].unique()
    logging.info(f"数据集中共有 {len(unique_labels)} 个类别: {', '.join(str(label) for label in unique_labels)}")
    
    # 检查标签是否为文本，是否需要编码
    label_encoder = None
    if not np.issubdtype(df[label_column].dtype, np.number):
        logging.info(f"标签类型为 {df[label_column].dtype}，需要进行编码")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[label_column])
        logging.info(f"标签已编码: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    else:
        logging.info(f"标签类型为 {df[label_column].dtype}，无需额外编码")
        y = df[label_column].values
    
    # 输出详细的类别分布信息
    unique_y, counts_y = np.unique(y, return_counts=True)
    logging.info("数据集类别分布详情:")
    for i, (cls, count) in enumerate(zip(unique_y, counts_y)):
        if label_encoder is not None:
            original_label = label_encoder.classes_[cls] if cls < len(label_encoder.classes_) else f"Unknown_{cls}"
            logging.info(f"  类别 {cls} ({original_label}): {count} 个样本 ({count/len(y)*100:.1f}%)")
        else:
            logging.info(f"  类别 {cls}: {count} 个样本 ({count/len(y)*100:.1f}%)")
    
    # 检查类别是否平衡
    min_samples = min(counts_y)
    max_samples = max(counts_y)
    imbalance_ratio = max_samples / min_samples
    if imbalance_ratio > 2.0:
        logging.warning(f"数据集存在类别不平衡！最大类别/最小类别 = {imbalance_ratio:.2f}")
        logging.info("建议使用分层抽样确保各类别在训练/测试集中的比例一致")
    else:
        logging.info(f"数据集类别相对平衡，不平衡比例: {imbalance_ratio:.2f}")
    
    feature_columns = df.columns[10:]
    logging.info(f"特征数量: {len(feature_columns)}")
    
    try:
        wavelengths = [float(col) for col in feature_columns]
        logging.info(f"波长范围: {min(wavelengths):.2f} - {max(wavelengths):.2f} nm")
    except ValueError:
        wavelengths = list(range(len(feature_columns)))
        logging.info("无法从列名提取波长信息，使用索引作为替代")
    
    X = df[feature_columns].values
    
    missing_values = np.isnan(X).sum()
    if missing_values.sum() > 0:
        logging.info(f"数据中存在 {missing_values.sum()} 个缺失值，使用列平均值填充")
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
    
    # 保存文件名，用于结果导出
    filenames = df[filename_column].values if filename_column in df.columns else None
    
    return X, y, wavelengths, label_encoder, filenames

def plot_confusion_matrix(model, X, y, model_name, label_encoder=None, experiment_name=None):
    """
    绘制混淆矩阵
    
    Args:
        model: 训练好的模型
        X: 特征数据
        y: 标签数据
        model_name: 模型名称
        label_encoder: 标签编码器，用于将数字标签转换回原始标签
        experiment_name: 实验名称，用于确定保存路径
    """
    # 从model_name提取基础模型名称（去除Fold后缀）
    base_model_name = model_name.split('_Fold')[0] if '_Fold' in model_name else model_name
    plot_dir = f'plot/{experiment_name}/{base_model_name.replace(" ", "_")}'
    os.makedirs(plot_dir, exist_ok=True)
    
    y_pred = model.predict(X)
    y_pred_original = y_pred
        
    # 确保原始标签用于混淆矩阵
    if label_encoder is not None:
        # 检查y是否已经是原始标签
        if np.issubdtype(y.dtype, np.number) and min(y) == 0 and max(y) < len(label_encoder.classes_):
            y_original = label_encoder.inverse_transform(y)
        else:
            y_original = y
    else:
        y_original = y
    
    # 确保y_original和y_pred_original都是字符串类型，以便正确显示在图表上
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
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_original, y_pred_original, labels=np.unique(y_original))
    
    # 获取标签名称列表
    labels = np.unique(y_original)
    
    # 绘制混淆矩阵
    fig = setfig(column=3, x=6, y=5.5)
    
    # 创建热力图，调整字体大小，设置注释格式
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=labels, 
        yticklabels=labels,
        annot_kws={"size": 8, "weight": "bold", "family": "sans-serif", "color": "black"}  # 确保注释文本为黑色
    )
    
    # 设置x轴和y轴刻度标签的字体，确保支持中文
    plt.xticks(rotation=30, fontfamily='Microsoft YaHei', weight='bold')  # 横坐标水平显示，微软雅黑粗体
    plt.yticks(fontfamily='Microsoft YaHei', weight='bold')  # 微软雅黑粗体

    # 设置x轴和y轴标签以及标题的字体，使用中文标签
    plt.xlabel('预测标签', fontfamily='Microsoft YaHei', weight='bold')
    plt.ylabel('真实标签', fontfamily='Microsoft YaHei', weight='bold')
    plt.title(f'{model_name} confusion matrix', fontfamily='Microsoft YaHei', weight='bold')
    
    plt.savefig(f'{plot_dir}/confusion_matrix_{model_name}.pdf', 
                   format='PDF', 
                   transparent=False,  # 关闭透明度确保背景为白色
                   bbox_inches='tight',
                   dpi=300,
                   facecolor='white',  # 明确设置白色背景
                   edgecolor='none',
                   metadata={'Creator': 'Spectral Classifier'})
    
    
    plt.close()
    
    logging.info(f"已保存 {model_name} 的混淆矩阵图到 '{plot_dir}'")

def train_and_evaluate_models(X, y, original_label_encoder=None, original_files=None, val_size=0.15, test_size=0.15, n_splits=5, experiment_name=None):
    """
    训练和评估不同的分类模型，使用分层K折交叉验证
    
    Args:
        X: 特征数据
        y: 标签
        original_label_encoder: 加载数据时使用的标签编码器，如果有的话
        original_files: 原始文件名列表，用于结果导出
        val_size: 验证集比例，默认0.15（仅在不使用交叉验证时使用）
        test_size: 测试集比例，默认0.15（仅在不使用交叉验证时使用）
        n_splits: 交叉验证的折数，默认5
        experiment_name: 实验名称，用于确定保存路径
        
    Returns:
        best_model: 性能最佳的模型
        best_model_name: 最佳模型的名称
    """
    logging.info(f"使用分层{n_splits}折交叉验证进行模型评估")
    
    # 初始化分层K折交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 定义分类器，使用经过网格搜索获取的最佳参数
    classifiers = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('lr', LogisticRegression(C=10, penalty='l2', class_weight=None, max_iter=1000, 
                                     multi_class='multinomial', solver='lbfgs', random_state=42))
        ]),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=5, 
                                              random_state=42, n_jobs=-1),
        'LDA': Pipeline([
            ('scaler', StandardScaler()),
            ('lda', LDA(solver='svd')),
        ]),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.2,
            objective='multi:softprob',
            random_state=42,
            use_label_encoder=False,
            verbosity=0,
            max_depth=3,
            subsample=0.8,
            ),
    }

    # # XGBoost可用
    # if XGBOOST_AVAILABLE:
    #     classifiers['XGBoost'] = xgb.XGBClassifier(
    #         n_estimators=200, 
    #         learning_rate=0.2, 
    #         objective='multi:softprob',
    #         random_state=42,
    #         use_label_encoder=False,
    #         verbosity=0,
    #         max_depth=3,
    #         subsample=0.8,
    #     )

    best_models = {}
    best_scores = {}
    cv_results = {}
    
    # 批量预测函数，用于大型数据集
    def batch_predict(model, X, batch_size=1000):
        n_samples = X.shape[0]
        predictions = []
        
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:min(i + batch_size, n_samples)]
            batch_pred = model.predict(batch_X)
            predictions.append(batch_pred)
            
        return np.concatenate(predictions)
    
    # 添加打印每个类别准确率的函数
    def print_class_accuracy(y_true, y_pred, class_names=None):
        """打印每个类别的准确率（与classification_report格式对齐）"""
        unique_classes = np.unique(y_true)
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=unique_classes, zero_division=0)
        logging.info("类别\tprecision\trecall\tf1-score\tsupport\t正确数\t准确率")
        for idx, cls in enumerate(unique_classes):
            mask = (y_true == cls)
            total = support[idx]
            correct = np.sum((y_true == cls) & (y_pred == cls))
            acc = correct / total if total > 0 else 0.0
            if class_names is not None and cls < len(class_names):
                cls_name = class_names[cls]
            else:
                cls_name = str(cls)
            logging.info(f"{cls_name}\t{precision[idx]:.2f}\t{recall[idx]:.2f}\t{f1[idx]:.2f}\t{total}\t{correct}\t{acc:.4f}")
        overall_acc = np.mean(y_true == y_pred)
        logging.info(f"总体准确率: {overall_acc:.4f}")
    
    # 遍历所有分类器进行交叉验证训练和评估
    for name, classifier in classifiers.items():
        logging.info(f"\n正在使用{n_splits}折交叉验证训练 {name} 模型...")
        
        try:
            # 初始化存储每折结果的列表
            fold_scores = []
            fold_models = []
            fold_confusion_matrices = []
            fold_predictions = []
            fold_true_labels = []
            
            # 执行K折交叉验证
            for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
                logging.info(f"训练第 {fold_idx+1}/{n_splits} 折...")
                # 划分当前折的训练集和测试集
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # 验证分层抽样效果
                train_unique, train_counts = np.unique(y_train, return_counts=True)
                test_unique, test_counts = np.unique(y_test, return_counts=True)
                logging.info(f"第 {fold_idx+1} 折训练集类别分布: {dict(zip(train_unique, train_counts))}")
                logging.info(f"第 {fold_idx+1} 折测试集类别分布: {dict(zip(test_unique, test_counts))}")
                if len(train_unique) != len(np.unique(y)) or len(test_unique) != len(np.unique(y)):
                    logging.warning(f"第 {fold_idx+1} 折中存在类别缺失！这可能影响模型性能评估。")
                file_train, file_test = None, None
                if original_files is not None:
                    file_train, file_test = original_files[train_index], original_files[test_index]
                # 训练模型
                if name == 'XGBoost':
                    val_size = 0.1
                    X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
                        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
                    )
                    eval_set = [(X_val_xgb, y_val_xgb)]
                    classifier.fit(X_train_xgb, y_train_xgb, eval_set=eval_set, verbose=False)
                    # 训练集、验证集、测试集预测
                    y_pred_train = classifier.predict(X_train_xgb)
                    y_pred_val = classifier.predict(X_val_xgb)
                    y_pred_test = classifier.predict(X_test)
                    # 训练集日志
                    logging.info(f"\n第 {fold_idx+1} 折训练集分类报告:")
                    logging.info(classification_report(y_train_xgb, y_pred_train, zero_division=0))
                    print_class_accuracy(y_train_xgb, y_pred_train, class_names=original_label_encoder.classes_ if original_label_encoder else None)
                    # 验证集日志
                    logging.info(f"\n第 {fold_idx+1} 折验证集分类报告:")
                    logging.info(classification_report(y_val_xgb, y_pred_val, zero_division=0))
                    print_class_accuracy(y_val_xgb, y_pred_val, class_names=original_label_encoder.classes_ if original_label_encoder else None)
                else:
                    # 其他模型直接拟合
                    classifier.fit(X_train, y_train)
                    # 训练集、验证集、测试集预测
                    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
                        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
                    )
                    y_pred_train = classifier.predict(X_train_sub)
                    y_pred_val = classifier.predict(X_val_sub)
                    y_pred_test = classifier.predict(X_test)
                    # 训练集日志
                    logging.info(f"\n第 {fold_idx+1} 折训练集分类报告:")
                    logging.info(classification_report(y_train_sub, y_pred_train, zero_division=0))
                    print_class_accuracy(y_train_sub, y_pred_train, class_names=original_label_encoder.classes_ if original_label_encoder else None)
                    # 验证集日志
                    logging.info(f"\n第 {fold_idx+1} 折验证集分类报告:")
                    logging.info(classification_report(y_val_sub, y_pred_val, zero_division=0))
                    print_class_accuracy(y_val_sub, y_pred_val, class_names=original_label_encoder.classes_ if original_label_encoder else None)
                # 测试集日志
                test_score = accuracy_score(y_test, y_pred_test)
                fold_scores.append(test_score)
                # 重要：深拷贝模型，确保每个fold保存独立的模型对象
                fold_models.append(deepcopy(classifier))
                fold_confusion_matrices.append(confusion_matrix(y_test, y_pred_test))
                fold_predictions.append(y_pred_test)
                fold_true_labels.append(y_test)
                logging.info(f"第 {fold_idx+1} 折测试集准确率: {test_score:.4f}")
                logging.info(f"\n第 {fold_idx+1} 折测试集分类报告:")
                logging.info(classification_report(y_test, y_pred_test, zero_division=0))
                print_class_accuracy(y_test, y_pred_test, class_names=original_label_encoder.classes_ if original_label_encoder else None)
                # 绘制混淆矩阵
                try:
                    plot_confusion_matrix(classifier, X_test, y_test, f"{name}_Fold{fold_idx+1}", original_label_encoder, experiment_name)
                except Exception as e:
                    logging.error(f"绘制{name}第{fold_idx+1}折混淆矩阵时出错: {e}")
                    traceback.print_exc()
            
            # 计算平均分数和标准差
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            logging.info(f"{name} {n_splits}折交叉验证平均准确率: {avg_score:.4f} ± {std_score:.4f}")
            
            # 选择得分最高的折作为最佳模型
            best_fold_idx = np.argmax(fold_scores)
            best_fold_score = fold_scores[best_fold_idx]
            best_models[name] = fold_models[best_fold_idx]
            best_scores[name] = best_fold_score
            
            logging.info(f"{name} 最佳模型来自第 {best_fold_idx+1} 折，准确率: {best_fold_score:.4f}")
            
            # 保存交叉验证结果
            cv_results[name] = {
                'fold_scores': fold_scores,
                'avg_score': avg_score,
                'std_score': std_score,
                'best_fold_idx': best_fold_idx,
                'best_fold_score': best_fold_score
            }
            
            # 保存当前模型到独立文件夹
            try:
                save_model(best_models[name], name, experiment_name)
            except Exception as e:
                logging.error(f"保存{name}模型时出错: {e}")
                traceback.print_exc()
            
            # 每个模型训练完后进行垃圾回收
            gc.collect()
            monitor_resources()
            time.sleep(1)  # 给系统短暂休息，避免过热
            
        except Exception as e:
            logging.error(f"训练 {name} 时出错: {e}")
            logging.info(f"跳过 {name} 模型")
            traceback.print_exc()
    
    
    # 选择最佳模型
    if best_scores:
        best_model_name = max(best_scores, key=best_scores.get)
        best_model = best_models[best_model_name]
        logging.info(f"\n最佳模型: {best_model_name} (准确率: {best_scores[best_model_name]:.4f})")
        
        # 输出所有模型的交叉验证结果
        logging.info("\n所有模型的交叉验证结果汇总:")
        for name, result in cv_results.items():
            logging.info(f"{name}: {result['avg_score']:.4f} ± {result['std_score']:.4f}")
    else:
        logging.info("\n未能成功训练任何模型")
        return None, "无"
    
    return best_model, best_model_name

def plot_feature_importance(model, wavelengths, model_name, experiment_name=None):
    """
    绘制特征重要性图（适用于随机森林）
    """
    if not model_name == 'Random Forest':
        return
    
    # 为特征重要性图创建独立文件夹
    plot_dir = f'plot/{experiment_name}/{model_name.replace(" ", "_")}'
    os.makedirs(plot_dir, exist_ok=True)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        if hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
            importances = model.steps[-1][1].feature_importances_
        else:
            logging.info("无法提取特征重要性")
            return
    
    indices = np.argsort(importances)[::-1]
    
    top_n = min(20, len(wavelengths))
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_wavelengths = [wavelengths[i] for i in top_indices]
    
    # plt.figure(figsize=(12, 6))
    fig = setfig(column=3, x=5, y=4)
    plt.bar(range(top_n), top_importances, align='center')
    plt.xticks(range(top_n), [f"{w:.2f}" for w in top_wavelengths], rotation=45, fontfamily='sans-serif')
    plt.xlabel('Wavelength (nm)', fontfamily='sans-serif')
    plt.ylabel('Feature Importance', fontfamily='sans-serif')
    plt.title(f'{model_name} Feature Importance - Top {top_n}', fontfamily='sans-serif')
    plt.tight_layout()
    
    try:
        # 保存PDF时指定字体嵌入方式
        plt.savefig(f'{plot_dir}/feature_importance_{model_name}.pdf', 
                   format='PDF', 
                   transparent=True, 
                   bbox_inches='tight',
                   dpi=300,
                   metadata={'Creator': 'Spectral Classifier'})
        logging.info(f"已保存特征重要性图到 '{plot_dir}'")
    except Exception as e:
        # 如果PDF保存失败，尝试保存为PNG
        logging.warning(f"PDF保存失败，改为保存PNG格式: {e}")
        plt.savefig(f'{plot_dir}/feature_importance_{model_name}.png', 
                   format='PNG', 
                   transparent=True, 
                   bbox_inches='tight',
                   dpi=300)
    
    plt.close()

def save_model(model, model_name, experiment_name=None):
    """保存模型，支持PyTorch和传统模型，每个模型保存在独立文件夹"""
    # 为每个模型创建独立子目录
    model_dir = f'models/{experiment_name}/{model_name.replace(" ", "_")}'
    os.makedirs(model_dir, exist_ok=True)

    model_path = f"{model_dir}/{model_name.replace(" ", "_")}.pth"
    torch.save(model, model_path)
    logging.info(f"模型已保存到 '{model_path}'")

def start_resource_monitoring(interval=15):
    import threading
    
    def monitor_thread():
        try:
            while True:
                monitor_resources()
                time.sleep(interval)
        except Exception as e:
            logging.error(f"资源监控线程出错: {e}")
    
    monitor_thread = threading.Thread(target=monitor_thread, daemon=True)
    monitor_thread.start()
    logging.info(f"已启动资源监控线程，监控间隔: {interval}秒")

def main():
    csv_file = "Data/Exp2/day14-NIR.csv"
    experiment_name = "Spectral_Classification_Exp2/day14-NIR"  
    
    logger, log_file = setup_logging(experiment_name=experiment_name)
    
    # 配置matplotlib字体
    font_configured = configure_matplotlib_fonts()
    
    logging.info("=========================================")
    logging.info("开始执行光谱分类任务")
    logging.info(f"Python版本: {sys.version}")
    logging.info(f"工作目录: {os.getcwd()}")
    logging.info(f"matplotlib字体配置: {'成功' if font_configured else '失败'}")
    
    try:
        import platform
        logging.info(f"系统信息: {platform.platform()}")
        
        import numpy as np
        import pandas as pd
        import sklearn
        import torch
        logging.info(f"NumPy版本: {np.__version__}")
        logging.info(f"Pandas版本: {pd.__version__}")
        logging.info(f"Scikit-learn版本: {sklearn.__version__}")
        logging.info(f"PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            logging.info(f"CUDA可用: 版本{torch.version.cuda}")
            logging.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("CUDA不可用，将使用CPU")
    except Exception as e:
        logging.error(f"记录系统信息时出错: {str(e)}")
    
    with log_time("内存配置"):
        import psutil
        try:
            process = psutil.Process(os.getpid())
            total_memory = psutil.virtual_memory().total
            max_memory = int(total_memory * 0.7)
            logging.info(f"系统总内存: {total_memory / (1024**3):.2f} GB")
            logging.info(f"设置最大内存使用量为: {max_memory / (1024**3):.2f} GB")
        except ImportError:
            logging.warning("无法导入psutil库，跳过内存限制设置")
        except Exception as e:
            logging.error(f"设置内存限制时出错: {str(e)}")
            traceback.print_exc()
    
    initial_resources = monitor_resources()
    
    checkpoint_file = f"models/{experiment_name}/training_checkpoint.pth"
    
    with log_time("GPU配置"):
        has_gpu = configure_gpu()
    
    # csv_file = "Data/day1-VIS-raw.csv"
    
    with log_time("检查点处理"):
        try:
            if os.path.exists(checkpoint_file):
                logging.info(f"发现检查点文件 {checkpoint_file}，尝试恢复之前的进度...")
                checkpoint_data = torch.load(checkpoint_file, weights_only=False)
                X = checkpoint_data.get('X')
                y = checkpoint_data.get('y')
                wavelengths = checkpoint_data.get('wavelengths')
                label_encoder = checkpoint_data.get('label_encoder')
                filenames = checkpoint_data.get('filenames')  # 加载文件名
                already_loaded = True
                logging.info(f"成功从检查点恢复数据: X形状={X.shape}, y形状={y.shape}")
                logging.info(f"标签分布: {np.unique(y, return_counts=True)}")
            else:
                already_loaded = False
                logging.info("未发现检查点文件，将从头开始处理数据")
        except Exception as e:
            logging.error(f"加载检查点时出错: {str(e)}")
            traceback.print_exc()
            already_loaded = False
    
    if not already_loaded:
        with log_time("数据加载与预处理"):
            try:
                X, y, wavelengths, label_encoder, filenames = load_and_preprocess_data(csv_file)
                
                os.makedirs("models", exist_ok=True)
                checkpoint_data = {
                    'X': X,
                    'y': y,
                    'wavelengths': wavelengths,
                    'label_encoder': label_encoder,
                    'filenames': filenames  # 保存文件名
                }
                torch.save(checkpoint_data, checkpoint_file)
                logging.info(f"已保存数据加载检查点到 {checkpoint_file}")
                logging.info(f"数据形状: X={X.shape}, y={y.shape}")
            except Exception as e:
                logging.error(f"加载和处理数据时出错: {str(e)}")
                traceback.print_exc()
                return
    
    try:
        start_resource_monitoring(interval=15)
    except Exception as e:
        logging.error(f"启动资源监控时出错: {str(e)}")
    
    with log_time("CPU使用限制"):
        limit_cpu_usage()
    
    with log_time("内存评估"):
        try:
            if hasattr(psutil, 'virtual_memory'):
                available_memory = psutil.virtual_memory().available
                required_memory = X.nbytes * 5
                
                if available_memory < required_memory:
                    logging.warning(f"警告: 可用内存({available_memory/(1024**3):.2f}GB)可能不足以处理数据({required_memory/(1024**3):.2f}GB)")
                    logging.warning("尝试降低模型复杂度或减少数据量")
                    
                    if available_memory < required_memory / 2:
                        sample_size = min(len(X), int(len(X) * (available_memory / (required_memory * 1.5))))
                        logging.info(f"由于内存限制，将随机采样 {sample_size} 个样本(原始: {len(X)})")
                        from sklearn.model_selection import train_test_split
                        X, _, y, _ = train_test_split(X, y, train_size=sample_size, stratify=y, random_state=42)
                        logging.info(f"采样后数据规模: {X.shape}")
        except Exception as e:
            logging.error(f"内存检查出错: {str(e)}")
    
    model_checkpoint_file = f"models/{experiment_name}/model_checkpoint.pth"
    final_model_files = {
        'LogisticRegression': f"models/{experiment_name}/LogisticRegression/LogisticRegression.pth",
        'Random Forest': f"models/{experiment_name}/Random_Forest/Random_Forest.pth",
        'LDA': f"models/{experiment_name}/LDA/LDA.pth",
        'XGBoost': f"models/{experiment_name}/XGBoost/XGBoost.pth",
    }
    
    # 检查是否有保存的最终模型，如果有就直接使用
    best_model = None
    best_model_name = None
    use_saved_model = False
    
    # 首先检查是否明确指定要重新训练
    retrain = False
    val_size = 0.15
    test_size = 0.15
    n_folds = 5
    try:
        parser = argparse.ArgumentParser(description='光谱分类器')
        parser.add_argument('--retrain', action='store_true', help='强制重新训练模型')
        parser.add_argument('--val-size', type=float, default=0.15, help='验证集比例 (默认: 0.15)')
        parser.add_argument('--test-size', type=float, default=0.15, help='测试集比例 (默认: 0.15)')
        parser.add_argument('--folds', type=int, default=5, help='交叉验证折数 (默认: 5)')
        parser.add_argument('--experiment', type=str, default=experiment_name, help='实验名称，用于区分不同数据集 ')
        args, unknown = parser.parse_known_args()
        retrain = args.retrain
        val_size = args.val_size
        test_size = args.test_size
        n_folds = args.folds
        if retrain:
            logging.info("指定了--retrain参数，将重新训练模型")
        logging.info(f"实验名称: {experiment_name}")
        logging.info(f"数据集划分比例: 验证集 {val_size:.2%}, 测试集 {test_size:.2%}, 训练集 {1-val_size-test_size:.2%}")
        logging.info(f"将使用{n_folds}折分层交叉验证进行模型评估")
    except:
        pass
    
    if not retrain:
        # 尝试加载保存的模型
        logging.info("检查是否有保存的模型可以直接使用...")
        
        model_priority = ['Random Forest', 'XGBoost', 'LogisticRegression', 'LDA']
        
        for model_name in model_priority:
            if use_saved_model:
                break
                
            if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                model_path = final_model_files[model_name]
                if os.path.exists(model_path):
                    try:
                        logging.info(f"尝试加载{model_name}模型...")
                        best_model = torch.load(model_path, weights_only=False)
                        best_model_name = model_name
                        use_saved_model = True
                        logging.info(f"成功加载{model_name}模型")
                    except Exception as e:
                        logging.error(f"加载{model_name}模型失败: {e}")
                        use_saved_model = False
            else:
                model_path = final_model_files.get(model_name)
                if model_path and os.path.exists(model_path):
                    try:
                        logging.info(f"尝试加载{model_name}模型...")
                        best_model = torch.load(model_path, weights_only=False)
                        best_model_name = model_name
                        use_saved_model = True
                        logging.info(f"成功加载{model_name}模型")
                    except Exception as e:
                        logging.error(f"加载{model_name}模型失败: {e}")
                        use_saved_model = False
    
    # 如果已经加载了保存的模型，跳过训练步骤
    if not use_saved_model:
        logging.info("未找到可用的保存模型，将重新训练...")
        
        try:
            logging.info("训练前系统散热休息5秒...")
            monitor_resources()
            time.sleep(5)
            monitor_resources()
        except:
            pass
        
        # try:
        with log_time("模型训练与评估"):
            logging.info("开始训练和评估模型...")
            monitor_resources()
            
            # 更新函数调用，添加n_splits参数
            best_model, best_model_name = train_and_evaluate_models(
                X, y, label_encoder, filenames, val_size=val_size, test_size=test_size, n_splits=n_folds, experiment_name=experiment_name
            )
            
            monitor_resources()
        
        if best_model is not None:
            try:
                with log_time("保存模型检查点"):
                    checkpoint_data = {
                        'best_model': best_model,
                        'best_model_name': best_model_name
                    }
                    torch.save(checkpoint_data, model_checkpoint_file)
                    logging.info(f"已保存模型训练检查点到 {model_checkpoint_file}")
            except Exception as e:
                logging.error(f"保存模型检查点时出错: {str(e)}")
                traceback.print_exc()
        
            try:
                with log_time("绘制混淆矩阵"):
                    # 为最佳模型绘制全数据集混淆矩阵
                    plot_confusion_matrix(best_model, X, y, best_model_name, label_encoder)
            except Exception as e:
                logging.error(f"绘制混淆矩阵时出错: {str(e)}")
                traceback.print_exc()
            
            if best_model_name == 'Random Forest':
                try:
                    with log_time("绘制特征重要性图"):
                        plot_feature_importance(best_model, wavelengths, best_model_name, experiment_name)
                except Exception as e:
                    logging.error(f"绘制特征重要性图时出错: {str(e)}")
                    traceback.print_exc()
            
            try:
                with log_time("保存最终模型"):
                    save_model(best_model, best_model_name, experiment_name)
            except Exception as e:
                logging.error(f"保存模型时出错: {str(e)}")
                traceback.print_exc()
            
            logging.info("\n分类模型训练和评估完成！")
            logging.info(f"最佳模型: {best_model_name}")
        else:
            logging.error("\n模型训练失败，请检查数据或调整参数后重试")
            
        logging.info("程序执行结束，日志保存在: " + log_file)
        logging.info("=========================================")
    else:
        logging.info(f"使用已保存的{best_model_name}模型，跳过训练步骤")
    
    # 不管是新训练的模型还是加载的模型，都执行评估和可视化
    if best_model is not None:
        # 确保有必要的数据可用于可视化
        if 'X' not in locals() and 'y' not in locals():
            logging.info("加载原始数据以用于可视化...")
            try:
                with log_time("数据加载"):
                    X, y, wavelengths, label_encoder, filenames = load_and_preprocess_data(csv_file)
            except Exception as e:
                logging.error(f"加载数据失败: {e}")
                X, y, wavelengths, label_encoder, filenames = None, None, None, None, None
        
        if X is not None and y is not None:
            try:
                with log_time("绘制混淆矩阵"):
                    plot_confusion_matrix(best_model, X, y, best_model_name, label_encoder, experiment_name)
            except Exception as e:
                logging.error(f"绘制混淆矩阵时出错: {str(e)}")
                traceback.print_exc()
            
            if best_model_name == 'Random Forest' and wavelengths is not None:
                try:
                    with log_time("绘制特征重要性图"):
                        plot_feature_importance(best_model, wavelengths, best_model_name, experiment_name)
                except Exception as e:
                    logging.error(f"绘制特征重要性图时出错: {str(e)}")
                    traceback.print_exc()
        
        logging.info(f"使用模型: {best_model_name}")
    else:
        logging.error("没有可用的模型")

if __name__ == "__main__":
    main()