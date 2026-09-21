"""
高光谱叶片前景提取脚本（感染叶片版本）
- 伪彩色可视化
- 双波段比值法前景分割
- 使用三分类模型预测
- 只保存感染叶片（infected_leaf）的高光谱数据
"""
import os
import numpy as np
import cv2
import re
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging

# 导入模型
from SpectralNN_classifier import SpectralNN


def read_hdr_metadata(hdr_path):
    metadata = {
        'samples': 0,
        'lines': 0,
        'bands': 0,
        'data_type': 12,
        'interleave': 'bil',
        'wavelengths': [],
    }
    try:
        with open(hdr_path, 'r', encoding='utf-8') as f:
            content = f.read()
        patterns = {
            'samples': r'samples\s*=\s*(\d+)',
            'lines': r'lines\s*=\s*(\d+)',
            'bands': r'bands\s*=\s*(\d+)',
            'data_type': r'data\s+type\s*=\s*(\d+)',
            'interleave': r'interleave\s*=\s*(\w+)',
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                if key in ['samples', 'lines', 'bands', 'data_type']:
                    metadata[key] = int(match.group(1))
                else:
                    metadata[key] = match.group(1).lower().strip()
        wavelength_match = re.search(r'wavelength\s*=\s*\{([^}]+)\}', content, re.IGNORECASE)
        if wavelength_match:
            wave_str = wavelength_match.group(1).strip()
            metadata['wavelengths'] = [float(w.strip()) for w in wave_str.split(',')]
        return metadata
    except Exception as e:
        print(f"读取HDR文件时出错: {str(e)}")
        return None

def read_hyperspectral_data(raw_path, metadata):
    data_type_map = {
        1: np.int8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64,
        12: np.uint16, 13: np.uint32, 14: np.uint8,
    }
    dtype = data_type_map.get(metadata['data_type'], np.uint16)
    samples = metadata['samples']
    lines = metadata['lines']
    bands = metadata['bands']
    interleave = metadata['interleave']
    try:
        data = np.fromfile(raw_path, dtype=dtype)
        if interleave == 'bsq':
            data = data.reshape((bands, lines, samples)).transpose(1,2,0)
        elif interleave == 'bil':
            data = data.reshape((lines, bands, samples)).transpose(0,2,1)
        elif interleave == 'bip':
            data = data.reshape((lines, samples, bands))
        else:
            raise ValueError(f"不支持的interleave格式: {interleave}")
        return data
    except Exception as e:
        print(f"读取RAW文件时出错: {str(e)}")
        return None

def generate_pseudo_color_image(hyper_data, metadata):
    wavelengths = np.array(metadata.get('wavelengths', []))
    h, w, bands = hyper_data.shape
    if len(wavelengths) != bands:
        print("未找到波长信息，使用前三个波段")
        rgb = hyper_data[:,:,:3]
    else:
        wl_min, wl_max = wavelengths.min(), wavelengths.max()
        if wl_min >= 800:
            r_idx = 0
            g_idx = bands // 3
            b_idx = 2 * bands // 3
            print(f"SWIR伪彩色: R={wavelengths[r_idx]:.1f}, G={wavelengths[g_idx]:.1f}, B={wavelengths[b_idx]:.1f}")
        else:
            r_idx = np.argmin(np.abs(wavelengths-850))
            g_idx = np.argmin(np.abs(wavelengths-720))
            b_idx = np.argmin(np.abs(wavelengths-650))
            print(f"VIS-NIR伪彩色: R={wavelengths[r_idx]:.1f}, G={wavelengths[g_idx]:.1f}, B={wavelengths[b_idx]:.1f}")
        rgb = np.stack([hyper_data[:,:,r_idx], hyper_data[:,:,g_idx], hyper_data[:,:,b_idx]], axis=2)
    rgb = rgb.astype(np.float32)
    for i in range(3):
        channel = rgb[:,:,i]
        valid = channel[channel>0]
        if len(valid)>0:
            p_low, p_high = np.percentile(valid, [0.5,99.5])
            if p_high>p_low:
                rgb[:,:,i] = np.clip((channel-p_low)/(p_high-p_low)*255,0,255)
            else:
                rgb[:,:,i]=0
        else:
            rgb[:,:,i]=0
    rgb = rgb.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    for i in range(3):
        rgb[:,:,i] = clahe.apply(rgb[:,:,i])
    return rgb

def extract_foreground_mask(hyper_data, metadata):
    wavelengths = np.array(metadata.get('wavelengths', []))
    h, w, bands = hyper_data.shape
    wl_high = 1102.0
    wl_abs = 1452.3
    tol = 5.0
    def find_idx(target):
        diff = np.abs(wavelengths - target)
        idx = np.argmin(diff)
        if diff[idx]>tol:
            print(f"⚠ 未找到{target}nm, 用{wavelengths[idx]:.1f}nm")
        return idx
    idx_high = find_idx(wl_high)
    idx_abs = find_idx(wl_abs)
    band_high = hyper_data[:,:,idx_high].astype(np.float32)
    band_abs = hyper_data[:,:,idx_abs].astype(np.float32)
    
    # 计算比值
    ratio = band_high/(band_abs+1e-6)
    
    # 直接使用原始值判断
    band_high_refl = band_high  # 1102nm的反射率
    band_abs_refl = band_abs     # 1452.3nm的反射率
    
    # print(f"  band_high (1102nm) 统计: min={band_high_refl.min():.4f}, max={band_high_refl.max():.4f}, mean={band_high_refl.mean():.4f}")
    # print(f"  band_abs (1452.3nm) 统计: min={band_abs_refl.min():.4f}, max={band_abs_refl.max():.4f}, mean={band_abs_refl.mean():.4f}")
    
    # 过滤策略（与原始extract_leaf_foreground.py完全一致）：
    # 条件1: 比值 > 1.0
    # 条件2: R_1102 > 0.5
    # 条件3: R_1452.3 < 0.7
    # 条件4：开运算去噪
    # 条件5: 连通区域分析，过滤小区域
    # 条件6: 外轮廓腐蚀
    ratio_threshold = 1.0
    r1102_min = 0.5  # 1102nm反射率下限
    r1102_max = 0.85  # 1102nm反射率上限
    r1452_max = 0.7  # 1452.3nm反射率上限
    
    # 步骤1: 仅用比值过滤
    mask_step1 = (ratio > ratio_threshold).astype(np.uint8)
    # print(f"\n  步骤1 - 仅比值>{ratio_threshold}: 保留 {mask_step1.sum()} 像素 ({mask_step1.sum()/(h*w)*100:.1f}%)")
    
    # 统计步骤1中两个波段的反射率分布
    if mask_step1.sum() > 0:
        step1_r1102 = band_high_refl[mask_step1 == 1]
        step1_r1452 = band_abs_refl[mask_step1 == 1]
        # print(f"    R_1102范围: [{step1_r1102.min():.4f}, {step1_r1102.max():.4f}], 均值={step1_r1102.mean():.4f}")
        # print(f"    R_1452.3范围: [{step1_r1452.min():.4f}, {step1_r1452.max():.4f}], 均值={step1_r1452.mean():.4f}")
    
    # 步骤2: 应用R_1102的范围限制
    mask_step2 = (mask_step1 & (band_high_refl > r1102_min)).astype(np.uint8)
    removed_step2 = mask_step1.sum() - mask_step2.sum()
    # print(f"\n  步骤2 - 过滤R_1102<{r1102_min}的像素")
    # print(f"    移除: {removed_step2} 个像素")
    # print(f"    保留: {mask_step2.sum()} 像素 ({mask_step2.sum()/(h*w)*100:.1f}%)")
    
    # 步骤3: 应用R_1452.3<0.7的限制
    mask_step3 = (mask_step2 & (band_abs_refl < r1452_max)).astype(np.uint8)
    removed_step3 = mask_step2.sum() - mask_step3.sum()
    # print(f"\n  步骤3 - 过滤R_1452.3>={r1452_max}的像素")
    # print(f"    移除: {removed_step3} 个像素")
    # print(f"    保留: {mask_step3.sum()} 像素 ({mask_step3.sum()/(h*w)*100:.1f}%)")
    
    # # 步骤4: 开运算去除小噪声点
    # print("\n  步骤4 - 开运算去除噪声点...")
    # kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    # mask_step4 = cv2.morphologyEx(mask_step3, cv2.MORPH_OPEN, kernel_open)
    # removed_step4 = mask_step3.sum() - mask_step4.sum()
    # print(f"    开运算移除: {removed_step4} 个像素")
    # print(f"    开运算后: {mask_step4.sum()} 像素 ({mask_step4.sum()/(h*w)*100:.1f}%)")
    
    # 步骤5: 连通区域分析，过滤小区域
    # print("\n  步骤5 - 连通区域分析，过滤小区域...")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_step3, connectivity=8)
    
    if num_labels > 1:  # 0是背景
        # print(f"    检测到 {num_labels-1} 个连通区域")
        
        # 显示各区域面积
        areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过背景
        # for i, area in enumerate(areas, 1):
        #     print(f"      区域{i}: 面积={area}像素")
        
        # 过滤策略：只保留面积大于阈值的区域
        min_region_area = 1000  # 最小区域面积（像素）
        valid_labels = np.where(areas >= min_region_area)[0] + 1  # +1因为跳过了背景
        
        if len(valid_labels) > 0:
            # 使用向量化过滤
            mask_step5 = np.isin(labels, valid_labels).astype(np.uint8)
            removed_regions = (num_labels - 1) - len(valid_labels)
            removed_pixels = mask_step3.sum() - mask_step5.sum()

            # print(f"    移除了 {removed_regions} 个小区域 (共 {removed_pixels} 个像素)")
            # print(f"    保留 {len(valid_labels)} 个区域 (面积 ≥ {min_region_area}像素)")
            print(f"    最终保留: {mask_step5.sum()} 像素 ({mask_step5.sum()/(h*w)*100:.1f}%)")
            
            mask = mask_step5
        else:
            print(f"    ⚠ 没有区域满足面积要求，保留步骤4结果")
            mask = mask_step3
    else:
        print(f"    只有1个连通区域，无需过滤")
        mask = mask_step3
    
    # # 步骤6: 腐蚀外部边缘（删除边缘2个像素，保持内部不变）
    # print("\n  步骤6 - 腐蚀外部边缘，删除边缘2个像素...")
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # if contours:
    #     mask_step6 = np.zeros_like(mask)
    #     for contour in contours:
    #         single_mask = np.zeros_like(mask)
    #         cv2.drawContours(single_mask, [contour], -1, 1, thickness=cv2.FILLED)
    #         kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  
    #         # (5, 5) 椭圆核 ≈ 腐蚀 2个像素
    #         # (7, 7) 椭圆核 ≈ 腐蚀 3个像素
    #         # (9, 9) 椭圆核 ≈ 腐蚀 4个像素
    #         # (11, 11) 椭圆核 ≈ 腐蚀 5个像素
    #         eroded_single = cv2.erode(single_mask, kernel_erode, iterations=1)
    #         mask_step6 = cv2.bitwise_or(mask_step6, eroded_single)
        
    #     removed_edge = mask.sum() - mask_step6.sum()
    #     print(f"    外部边缘腐蚀移除: {removed_edge} 个像素")
    #     print(f"    最终保留: {mask_step6.sum()} 像素 ({mask_step6.sum()/(h*w)*100:.1f}%)")
    #     mask = mask_step6
    # else:
    #     print(f"    未检测到轮廓，跳过边缘腐蚀")
    
    return mask

def save_hyper_data(data, metadata, out_base):
    lines, samples, bands = data.shape
    interleave = metadata['interleave']
    if interleave=='bsq':
        save_data = np.transpose(data, (2,0,1))
    elif interleave=='bil':
        save_data = np.transpose(data, (0,2,1))
    elif interleave=='bip':
        save_data = data
    dtype_map = {1: np.int8,2: np.int16,3: np.int32,4: np.float32,5: np.float64,12: np.uint16,13: np.uint32,14: np.uint8}
    dtype = dtype_map.get(metadata['data_type'], np.uint16)
    save_data = save_data.astype(dtype)
    save_data.tofile(out_base+'.raw')
    with open(out_base+'.hdr','w',encoding='utf-8') as f:
        f.write("ENVI\n")
        f.write(f"samples = {samples}\n")
        f.write(f"lines = {lines}\n")
        f.write(f"bands = {bands}\n")
        f.write(f"header offset = {metadata.get('header_offset',0)}\n")
        f.write("file type = ENVI Standard\n")
        f.write(f"data type = {metadata['data_type']}\n")
        f.write(f"interleave = {metadata['interleave']}\n")
        f.write("sensor type = Unknown\n")
        f.write(f"wavelength units = {metadata.get('wavelength_units','nm')}\n")
        if metadata.get('wavelengths'):
            f.write("wavelength = {\n")
            for i,wl in enumerate(metadata['wavelengths']):
                if i<len(metadata['wavelengths'])-1:
                    f.write(f"{wl:.2f},\n")
                else:
                    f.write(f"{wl:.2f}\n")
            f.write("}\n")
    print(f"已保存: {out_base+'.raw'} 和 {out_base+'.hdr'}")

def load_three_classes_model(model_dir='models/three_classes_model'):
    """
    加载训练好的三分类模型
    
    Returns:
        model, scaler, label_encoder, device
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    
    # 加载模型信息
    info = torch.load(f'{model_dir}/nn_model_info.pth')
    input_dim = info['input_dim']
    num_classes = info['num_classes']
    
    # 加载标准化器和标签编码器
    scaler = torch.load(f'{model_dir}/nn_scaler.pth')
    label_encoder = torch.load(f'{model_dir}/nn_label_encoder.pth')
    
    # 创建模型并加载权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralNN(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/nn_model.pth', map_location=device))
    model.eval()
    
    print(f"✓ 成功加载三分类模型")
    print(f"  类别: {label_encoder.classes_.tolist()}")
    
    return model, scaler, label_encoder, device

def predict_pixels(model, scaler, label_encoder, device, pixels_data):
    """
    预测像素的类别
    
    Args:
        model: 训练好的模型
        scaler: 标准化器
        label_encoder: 标签编码器
        device: 设备
        pixels_data: 像素数据 (n_pixels, bands)
        
    Returns:
        predictions: 预测的类别名称
        is_infected: 是否为感染叶片的布尔数组
    """
    # 标准化
    X_scaled = scaler.transform(pixels_data)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted_indices = torch.max(outputs, 1)
    
    # 转换为类别名称
    predicted_indices_np = predicted_indices.cpu().numpy()
    predictions = label_encoder.inverse_transform(predicted_indices_np)
    
    # 判断是否为感染叶片
    is_infected = (predictions == 'infected_leaf')
    
    return predictions, is_infected

def filter_infected_pixels(hyper_data, mask, model, scaler, label_encoder, device):
    """
    使用三分类模型过滤出感染叶片像素
    
    Args:
        hyper_data: 高光谱数据 (lines, samples, bands)
        mask: 前景mask (lines, samples)
        model, scaler, label_encoder, device: 模型相关参数
        
    Returns:
        infected_mask: 感染叶片的mask
        class_counts: 各类别统计
    """
    h, w, bands = hyper_data.shape
    
    # 提取前景像素的光谱数据
    foreground_indices = np.where(mask == 1)
    foreground_pixels = hyper_data[foreground_indices]  # (n_pixels, bands)
    
    total_foreground = len(foreground_pixels)
    print(f"\n  使用三分类模型预测 {total_foreground} 个前景像素...")
    
    if total_foreground == 0:
        return np.zeros_like(mask), {}
    
    # 预测
    predictions, is_infected = predict_pixels(
        model, scaler, label_encoder, device, foreground_pixels
    )
    
    # 统计各类别
    unique, counts = np.unique(predictions, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    print(f"  预测结果:")
    for class_name in label_encoder.classes_:
        count = class_counts.get(class_name, 0)
        ratio = count / total_foreground if total_foreground > 0 else 0.0
        print(f"    {class_name}: {count} ({ratio:.2%})")
    
    # 创建感染叶片mask
    infected_mask = np.zeros_like(mask)
    infected_mask[foreground_indices[0][is_infected], foreground_indices[1][is_infected]] = 1
    
    infected_count = infected_mask.sum()
    print(f"  最终保留 infected_leaf: {infected_count} 像素 ({infected_count/total_foreground*100:.1f}%)")
    
    return infected_mask, class_counts

def main():
    input_dir = r"F:\Code\Code_FHY_V1\20251118-L1-ref"
    output_dir = r"F:\Code\Code_FHY_V1\20251118-L1-infected"
    os.makedirs(output_dir, exist_ok=True)
    
    # # 只处理指定的文件
    # target_file = "newrawfile20251118131400_ref"
    
    # 日志配置
    log_path = os.path.join(output_dir, "process.log")
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("===== 开始高光谱叶片处理任务 =====")
    
    # 加载三分类模型（只加载一次）
    print("\n加载三分类模型...")
    model_3class, scaler_3class, label_encoder_3class, device = load_three_classes_model()
    
    # 加载17分类模型
    print("\n加载17分类模型...")
    model_17class, scaler_17class, label_encoder_17class, _ = load_three_classes_model(model_dir='models/day1_17classes_model')
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.hdr'):
            base = os.path.splitext(filename)[0]
            
            # # 跳过不是目标文件的其他文件
            # if base != target_file:
            #     continue
            
            hdr_path = os.path.join(input_dir, base+'.hdr')
            raw_path = os.path.join(input_dir, base+'.raw')
            if not os.path.exists(raw_path):
                continue
            # print(f"\n{'='*60}")
            # print(f"处理: {base}")
            # print(f"{'='*60}")
            logging.info("\n" + "="*60)
            logging.info(f"处理: {base}")
            logging.info("="*60)
            
            metadata = read_hdr_metadata(hdr_path)
            hyper_data = read_hyperspectral_data(raw_path, metadata)
            if hyper_data is None:
                continue
            
            # 生成伪彩色
            rgb = generate_pseudo_color_image(hyper_data, metadata)
            cv2.imwrite(os.path.join(output_dir, base+'_pseudo_color.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # 步骤1: 提取前景mask
            print(f"\n步骤1: 提取前景区域...")
            mask = extract_foreground_mask(hyper_data, metadata)
            # 保存背景过滤后的前景bmp
            rgb_foreground = rgb.copy()
            rgb_foreground[mask == 0] = [0, 0, 0]
            bmp_fg_path = os.path.join(output_dir, base+'_foreground_only.bmp')
            cv2.imwrite(bmp_fg_path, cv2.cvtColor(rgb_foreground, cv2.COLOR_RGB2BGR))
            print(f"✓ 已保存背景过滤后前景BMP: {base+'_foreground_only.bmp'}")
            
            # 步骤2: 使用三分类模型过滤感染叶片
            print(f"\n步骤2: 三分类模型预测...")
            infected_mask, class_counts = filter_infected_pixels(
                hyper_data, mask, model_3class, scaler_3class, label_encoder_3class, device
            )
            # 保存三分类过滤后的感染叶片bmp
            rgb_infected_classify = rgb.copy()
            rgb_infected_classify[infected_mask == 0] = [0, 0, 0]
            bmp_classify_path = os.path.join(output_dir, base+'_infected_classify_only.bmp')
            cv2.imwrite(bmp_classify_path, cv2.cvtColor(rgb_infected_classify, cv2.COLOR_RGB2BGR))
            print(f"✓ 已保存三分类过滤后感染叶片BMP: {base+'_infected_classify_only.bmp'}")
            
            # 步骤3: 使用17分类模型预测并保存像素最多的类别
            print(f"\n步骤3: 17分类模型预测...")
            if infected_mask.sum() == 0:
                print(f"⚠ 无感染像素，跳过17分类")
            else:
                # 提取感染像素的光谱数据
                infected_indices = np.where(infected_mask == 1)
                infected_pixels = hyper_data[infected_indices]
                
                # 使用17分类模型预测
                X_scaled = scaler_17class.transform(infected_pixels)
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                with torch.no_grad():
                    outputs = model_17class(X_tensor)
                    _, predicted_indices = torch.max(outputs, 1)
                predictions_17 = label_encoder_17class.inverse_transform(predicted_indices.cpu().numpy())
                
                # 统计各类别
                unique, counts = np.unique(predictions_17, return_counts=True)
                class_counts_17 = dict(zip(unique, counts))
                
                # print(f"  17分类预测结果:")
                logging.info(f"  17分类预测结果:")
                for class_name, count in sorted(class_counts_17.items(), key=lambda x: x[1], reverse=True):
                    ratio = count / len(predictions_17)
                    # print(f"    {class_name}: {count} ({ratio:.2%})")
                    logging.info(f"    {class_name}: {count} ({ratio:.2%})")
                
                # 找出像素数最多的类别
                max_class = max(class_counts_17, key=class_counts_17.get)
                max_count = class_counts_17[max_class]
                # print(f"  最多类别: {max_class} ({max_count} 像素)")
                logging.info(f"  最多类别: {max_class} ({max_count} 像素)")
                
                # 创建最多类别的mask
                is_max_class = (predictions_17 == max_class)
                max_class_mask = np.zeros_like(infected_mask)
                max_class_mask[infected_indices[0][is_max_class], infected_indices[1][is_max_class]] = 1
                
                # 保存最多类别的bmp
                rgb_max_class = rgb.copy()
                rgb_max_class[max_class_mask == 0] = [0, 0, 0]
                bmp_max_class_path = os.path.join(output_dir, base+f'_{max_class}_only.bmp')
                cv2.imwrite(bmp_max_class_path, cv2.cvtColor(rgb_max_class, cv2.COLOR_RGB2BGR))
                print(f"✓ 已保存{max_class}类别BMP: {base}_{max_class}_only.bmp")
                
                # 保存最多类别的高光谱数据
                if max_class_mask.sum() > 0:
                    max_class_data = hyper_data * np.expand_dims(max_class_mask, axis=2)
                    out_base_max_class = os.path.join(output_dir, base+f'_{max_class}')
                    save_hyper_data(max_class_data, metadata, out_base_max_class)
                    print(f"✓ {max_class}类别像素: {max_class_mask.sum()} / {infected_mask.sum()} ({max_class_mask.sum()/infected_mask.sum()*100:.1f}%)")
                else:
                    print(f"⚠ 未检测到{max_class}类别像素，跳过保存")
            
            print(f"{'='*60}\n")
            
            # break  # 只处理一个文件


if __name__ == '__main__':
    main()
