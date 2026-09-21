"""
高光谱叶片前景提取脚本
- 伪彩色可视化
- 双波段比值法前景分割
- 前景高光谱数据保存
"""
import os
import numpy as np
import cv2
import re
from pathlib import Path
from skimage.filters import threshold_otsu


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
    
    # # 计算比值
    # ratio = band_high/(band_abs+1e-6)
    
    # 计算NDI
    band_high_refl = band_high  # 1102nm的反射率
    band_abs_refl = band_abs     # 1452.3nm的反射率
    ndi = (band_high_refl + band_abs_refl) / (band_high_refl - band_abs_refl + 1e-6)
    
    print(f"  band_high (1102nm) 统计: min={band_high_refl.min():.4f}, max={band_high_refl.max():.4f}, mean={band_high_refl.mean():.4f}")
    print(f"  band_abs (1452.3nm) 统计: min={band_abs_refl.min():.4f}, max={band_abs_refl.max():.4f}, mean={band_abs_refl.mean():.4f}")
    print(f"  NDI(1102,1452.3) 统计: min={ndi.min():.4f}, max={ndi.max():.4f}, mean={ndi.mean():.4f}")
    
    # 过滤策略：
    # 条件1: ndi > 0
    # 条件2: R_1102 > 0.5
    # 条件3: R_1452.3 < 0.7
    # 条件4: 连通区域分析，过滤小区域
    # ratio_threshold = 1.0
    r1102_min = 0.5  # 1102nm反射率下限
    r1102_max = 0.85  # 1102nm反射率上限
    r1452_max = 0.7  # 1452.3nm反射率上限
    
    # 步骤1: 用NDI过滤
    mask_step1 = (ndi > 0).astype(np.uint8)
    # mask_step1 = (ndi > threshold_otsu(ndi[~np.isnan(ndi)])).astype(np.uint8)
    print(f"\n  步骤1 - NDI>0: 保留 {mask_step1.sum()} 像素 ({mask_step1.sum()/(h*w)*100:.1f}%)")
    
    # 统计步骤1中两个波段的反射率分布
    if mask_step1.sum() > 0:
        step1_r1102 = band_high_refl[mask_step1 == 1]
        step1_r1452 = band_abs_refl[mask_step1 == 1]
        print(f"    R_1102范围: [{step1_r1102.min():.4f}, {step1_r1102.max():.4f}], 均值={step1_r1102.mean():.4f}")
        print(f"    R_1452.3范围: [{step1_r1452.min():.4f}, {step1_r1452.max():.4f}], 均值={step1_r1452.mean():.4f}")
    
    # 步骤2: 应用R_1102的范围限制
    # mask_step2 = (mask_step1 & (band_high_refl > r1102_min) & (band_high_refl < r1102_max)).astype(np.uint8)
    mask_step2 = (mask_step1 & (band_high_refl > r1102_min) ).astype(np.uint8)
    removed_step2 = mask_step1.sum() - mask_step2.sum()
    print(f"\n  步骤2 - 过滤R_1102不在[{r1102_min}, {r1102_max}]范围内的像素")
    print(f"    移除: {removed_step2} 个像素")
    print(f"    保留: {mask_step2.sum()} 像素 ({mask_step2.sum()/(h*w)*100:.1f}%)")
    
    # 步骤3: 应用R_1452.3<0.7的限制
    mask_step3 = (mask_step2 & (band_abs_refl < r1452_max)).astype(np.uint8)
    removed_step3 = mask_step2.sum() - mask_step3.sum()
    print(f"\n  步骤3 - 过滤R_1452.3>={r1452_max}的像素")
    print(f"    移除: {removed_step3} 个像素")
    print(f"    保留: {mask_step3.sum()} 像素 ({mask_step3.sum()/(h*w)*100:.1f}%)")
    
    
    # 步骤5: 连通区域分析，过滤小区域
    print("\n  步骤5 - 连通区域分析，过滤小区域...")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_step3, connectivity=8)
    
    if num_labels > 1:  # 0是背景
        print(f"    检测到 {num_labels-1} 个连通区域")
        
        # 显示各区域面积
        areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过背景
        for i, area in enumerate(areas, 1):
            print(f"      区域{i}: 面积={area}像素")
        
        # 过滤策略：只保留面积大于阈值的区域
        min_region_area = 1000  # 最小区域面积（像素）
        valid_labels = np.where(areas >= min_region_area)[0] + 1  # +1因为跳过了背景
        
        if len(valid_labels) > 0:
            # 使用向量化过滤
            mask_step5 = np.isin(labels, valid_labels).astype(np.uint8)
            removed_regions = (num_labels - 1) - len(valid_labels)
            removed_pixels = mask_step3.sum() - mask_step5.sum()

            print(f"    移除了 {removed_regions} 个小区域 (共 {removed_pixels} 个像素)")
            print(f"    保留 {len(valid_labels)} 个区域 (面积 ≥ {min_region_area}像素)")
            print(f"    最终保留: {mask_step5.sum()} 像素 ({mask_step5.sum()/(h*w)*100:.1f}%)")
            
            mask = mask_step5
        else:
            print(f"    ⚠ 没有区域满足面积要求，保留开运算结果")
            mask = mask_step3
    else:
        print(f"    只有1个连通区域，无需过滤")
        mask = mask_step3
    
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

def main():
    input_dir = r"F:\Code\Code_FHY_V1\20251118-L1-ref"
    output_dir = r"F:\Code\Code_FHY_V1\20251118-L1-foreground"
    os.makedirs(output_dir, exist_ok=True)
    
    # # 只处理指定的文件
    # target_file = "newrawfile20251118131400_ref"
    
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
            print(f"\n处理: {base}")
            metadata = read_hdr_metadata(hdr_path)
            hyper_data = read_hyperspectral_data(raw_path, metadata)
            if hyper_data is None:
                continue
            # 生成伪彩色
            rgb = generate_pseudo_color_image(hyper_data, metadata)
            cv2.imwrite(os.path.join(output_dir, base+'_pseudo_color.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # 前景mask
            mask = extract_foreground_mask(hyper_data, metadata)
            
            # 保存无背景的BMP文件（前景伪彩色，背景黑色）
            rgb_foreground = rgb.copy()
            rgb_foreground[mask == 0] = [0, 0, 0]  # 背景设为黑色
            bmp_path = os.path.join(output_dir, base+'_foreground_only.bmp')
            cv2.imwrite(bmp_path, cv2.cvtColor(rgb_foreground, cv2.COLOR_RGB2BGR))
            print(f"已保存无背景BMP: {base+'_foreground_only.bmp'}")
            
            # 前景高光谱数据
            mask3d = np.expand_dims(mask,2)
            fg_data = hyper_data * mask3d
            # 保存
            out_base = os.path.join(output_dir, base+'_foreground')
            save_hyper_data(fg_data, metadata, out_base)
            print(f"前景像素: {mask.sum()} / {mask.size} ({mask.sum()/mask.size*100:.1f}%)")
            print(f"伪彩色和前景高光谱已保存到: {output_dir}")
            
            # break  # 只处理一个文件

if __name__ == '__main__':
    main()
