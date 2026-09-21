"""
高光谱数据平均反射率提取脚本

功能描述：
    从高光谱HDR/数据文件对中提取每个波段的平均反射率，并保存为CSV格式。
    支持批量处理指定目录下的所有高光谱文件。

主要功能：
    1. 自动识别HDR头文件及其对应的数据文件（.dat/.img等）
    2. 解析HDR文件元数据（图像尺寸、波段数、数据类型、波长信息等）
    3. 支持三种数据组织格式（BSQ、BIL、BIP）
    4. 计算每个波段的全图平均反射率
    5. 提取波长信息作为CSV表头
    6. 支持追加模式，可多次运行合并不同目录的数据

输入数据格式：
    - HDR头文件（.hdr）：包含元数据信息
      * samples: 图像列数
      * lines: 图像行数
      * bands: 波段数
      * data type: 数据类型（1-14）
      * interleave: 数据组织方式（bsq/bil/bip）
      * byte order: 字节序（0=小端，1=大端）
      * wavelength: 波长列表
      * wavelength units: 波长单位
    
    - 数据文件（.dat/.img或无扩展名）：二进制格式的光谱数据

支持的数据类型：
    - 1: 8位有符号整数
    - 2: 16位有符号整数
    - 3: 32位有符号整数
    - 4: 32位浮点数
    - 5: 64位浮点数
    - 12: 16位无符号整数
    - 13: 32位无符号整数
    - 14: 8位无符号整数

支持的数据组织方式：
    - BSQ (Band Sequential): 按波段顺序存储
    - BIL (Band Interleaved by Line): 按行交叉存储
    - BIP (Band Interleaved by Pixel): 按像素交叉存储

输出格式：
    CSV文件结构：
    - 第1列：label（数据类别，来自目录名）
    - 第2列：filename（样本文件名，不含扩展名）
    - 第3-N列：各波段平均反射率（列名为波长值，如"900.00"）
    
    示例：
    label,filename,900.00,901.50,903.00,...
    Leaf,sample1,0.234,0.256,0.278,...
    Leaf,sample2,0.189,0.201,0.215,...

工作流程：
    1. 扫描指定目录，查找所有.hdr文件
    2. 为每个HDR文件查找对应的数据文件
    3. 解析HDR元数据（尺寸、波段数、数据类型等）
    4. 根据数据组织方式读取二进制数据
    5. 计算每个波段的平均反射率
    6. 提取波长信息作为CSV列名
    7. 追加或创建CSV文件保存结果

使用方法：
    1. 基本用法（修改main函数中的路径）：
       data_directory = "Data/YourFolder"  # 数据目录
       output_file = "Data/output.csv"     # 输出文件
       python extract_average_reflectivity.py
    
    2. 批量处理多个目录：
       多次运行脚本，指定不同的data_directory
       结果会自动追加到同一个CSV文件
    
    3. 结果可视化：
       运行完成后，使用plot_reflectivity.py绘制反射率曲线

注意事项：
    1. HDR文件和数据文件必须同名（扩展名不同）
    2. 数据文件可以是.dat、.img或无扩展名
    3. 确保HDR文件包含完整的元数据信息
    4. CSV输出使用UTF-8编码
    5. 如果输出文件已存在，会追加新数据（不会覆盖）
    6. 波长信息从第一个HDR文件中提取，应用于所有样本

依赖库：
    - os: 文件和目录操作
    - csv: CSV文件读写
    - numpy: 数值计算
    - struct: 二进制数据解包
    - re: 正则表达式解析HDR

作者：Maggie Zhang
版本：v1.0
最后修改：2025年12月16日
"""

import os
import csv
import numpy as np
import struct

def extract_average_reflectivity(directory_path):
    """
    提取指定目录中每个高光谱文件的每个波段的平均反射率
    
    Args:
        directory_path: 包含高光谱数据文件的目录路径
    
    Returns:
        一个字典，键为文件名，值为该文件每个波段的平均反射率列表
    """
    results = {}
    
    # 确保目录存在
    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在")
        return results
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.upper().endswith('.HDR'):
            try:
                # 获取基本文件名（不包括扩展名）
                base_filename = os.path.splitext(filename)[0]
                hdr_path = os.path.join(directory_path, filename)
                
                # 检查是否有对应的数据文件
                data_extensions = ['.dat', '.DAT', '.img', '.IMG', '']
                data_path = None
                
                for ext in data_extensions:
                    possible_path = os.path.join(directory_path, base_filename + ext)
                    if os.path.exists(possible_path) and not possible_path.upper().endswith('.HDR'):
                        data_path = possible_path
                        break
                
                if not data_path:
                    print(f"未找到HDR文件 {filename} 对应的数据文件")
                    continue
                
                print(f"处理文件: {filename} 和 {os.path.basename(data_path)}")
                
                # 从HDR文件读取元数据
                metadata = read_hdr_metadata(hdr_path)
                
                if not metadata:
                    print(f"无法从 {filename} 读取元数据")
                    continue
                
                # 读取数据文件并计算每个波段的平均值
                band_averages = read_and_calculate_band_averages(data_path, metadata)
                
                if band_averages:
                    # 存储结果
                    results[filename] = band_averages
                    print(f"成功处理 {filename}: 共 {len(band_averages)} 个波段")
                else:
                    print(f"无法计算 {filename} 的波段平均值")
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    return results

def read_hdr_metadata(hdr_path):
    """
    从HDR文件读取必要的元数据
    
    Returns:
        包含元数据的字典，如果读取失败则返回None
    """
    metadata = {
        'samples': 0,        # 列数
        'lines': 0,          # 行数
        'bands': 0,          # 波段数
        'data_type': 1,      # 数据类型编码，默认1表示8位有符号整数
        'interleave': 'bsq', # 数据组织方式，默认为BSQ
        'byte_order': 0,     # 字节顺序，0表示小端
        'wavelengths': None, # 波长列表
        'wavelength_units': None, # 波长单位
    }
    
    try:
        with open(hdr_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 使用简单的正则表达式匹配键值对
            import re
            
            # 提取基本参数
            samples_match = re.search(r'samples\s*=\s*(\d+)', content, re.IGNORECASE)
            if samples_match:
                metadata['samples'] = int(samples_match.group(1))
                
            lines_match = re.search(r'lines\s*=\s*(\d+)', content, re.IGNORECASE)
            if lines_match:
                metadata['lines'] = int(lines_match.group(1))
                
            bands_match = re.search(r'bands\s*=\s*(\d+)', content, re.IGNORECASE)
            if bands_match:
                metadata['bands'] = int(bands_match.group(1))
                
            data_type_match = re.search(r'data\s+type\s*=\s*(\d+)', content, re.IGNORECASE)
            if data_type_match:
                metadata['data_type'] = int(data_type_match.group(1))
                
            interleave_match = re.search(r'interleave\s*=\s*(\w+)', content, re.IGNORECASE)
            if interleave_match:
                metadata['interleave'] = interleave_match.group(1).lower().strip()
                
            byte_order_match = re.search(r'byte\s+order\s*=\s*(\d+)', content, re.IGNORECASE)
            if byte_order_match:
                metadata['byte_order'] = int(byte_order_match.group(1))
            
            # 提取波长单位
            wavelength_units_match = re.search(r'wavelength\s+units\s*=\s*(\w+)', content, re.IGNORECASE)
            if wavelength_units_match:
                metadata['wavelength_units'] = wavelength_units_match.group(1).strip()
            
            # 提取波长列表
            wavelength_match = re.search(r'wavelength\s*=\s*\{([^}]+)\}', content, re.IGNORECASE)
            if wavelength_match:
                wave_str = wavelength_match.group(1).strip()
                try:
                    # 分割并转换为浮点数
                    metadata['wavelengths'] = [float(w.strip()) for w in wave_str.split(',')]
                    wavelength_count = len(metadata['wavelengths'])
                    print(f"从HDR文件读取到 {wavelength_count} 个波长值")
                    
                except ValueError:
                    print(f"波长值格式不正确")
                    
    except Exception as e:
        print(f"读取HDR文件时出错: {str(e)}")
        return None
        
    # 验证必要的元数据是否存在
    if metadata['samples'] <= 0 or metadata['lines'] <= 0 or metadata['bands'] <= 0:
        print("无效的图像维度")
        return None
        
    return metadata

def read_and_calculate_band_averages(data_path, metadata):
    """
    读取二进制数据文件并计算每个波段的平均值
    
    Args:
        data_path: 数据文件路径
        metadata: 从HDR文件提取的元数据
        
    Returns:
        每个波段的平均值列表
    """
    # 定义不同数据类型的格式字符和字节数
    data_type_formats = {
        1: ('b', 1),   # 8位有符号整数
        2: ('h', 2),   # 16位有符号整数
        3: ('i', 4),   # 32位有符号整数
        4: ('f', 4),   # 32位浮点数
        5: ('d', 8),   # 64位浮点数
        12: ('H', 2),  # 16位无符号整数
        13: ('I', 4),  # 32位无符号整数
        14: ('B', 1),  # 8位无符号整数
    }
    
    if metadata['data_type'] not in data_type_formats:
        print(f"不支持的数据类型: {metadata['data_type']}")
        return None
    
    format_char, bytes_per_pixel = data_type_formats[metadata['data_type']]
    samples = metadata['samples']
    lines = metadata['lines']
    bands = metadata['bands']
    interleave = metadata['interleave']
    
    # 设置字节顺序
    endian = '<' if metadata['byte_order'] == 0 else '>'
    format_str = endian + format_char
    
    try:
        band_averages = []
        
        with open(data_path, 'rb') as f:
            data_bytes = f.read()
            
            # 根据不同的数据组织方式(interleave)计算每个波段的平均值
            if interleave == 'bsq':  # Band Sequential
                for band in range(bands):
                    band_sum = 0.0
                    pixel_count = 0
                    
                    start_pos = band * samples * lines * bytes_per_pixel
                    end_pos = start_pos + samples * lines * bytes_per_pixel
                    
                    for i in range(start_pos, end_pos, bytes_per_pixel):
                        if i + bytes_per_pixel <= len(data_bytes):
                            value = struct.unpack(format_str, data_bytes[i:i+bytes_per_pixel])[0]
                            band_sum += value
                            pixel_count += 1
                    
                    if pixel_count > 0:
                        band_averages.append(band_sum / pixel_count)
                    else:
                        band_averages.append(0.0)
                        
            elif interleave == 'bil':  # Band Interleaved by Line
                for band in range(bands):
                    band_sum = 0.0
                    pixel_count = 0
                    
                    for line in range(lines):
                        for sample in range(samples):
                            pos = ((line * bands + band) * samples + sample) * bytes_per_pixel
                            if pos + bytes_per_pixel <= len(data_bytes):
                                value = struct.unpack(format_str, data_bytes[pos:pos+bytes_per_pixel])[0]
                                band_sum += value
                                pixel_count += 1
                    
                    if pixel_count > 0:
                        band_averages.append(band_sum / pixel_count)
                    else:
                        band_averages.append(0.0)
                        
            elif interleave == 'bip':  # Band Interleaved by Pixel
                for band in range(bands):
                    band_sum = 0.0
                    pixel_count = 0
                    
                    for line in range(lines):
                        for sample in range(samples):
                            pos = ((line * samples + sample) * bands + band) * bytes_per_pixel
                            if pos + bytes_per_pixel <= len(data_bytes):
                                value = struct.unpack(format_str, data_bytes[pos:pos+bytes_per_pixel])[0]
                                band_sum += value
                                pixel_count += 1
                    
                    if pixel_count > 0:
                        band_averages.append(band_sum / pixel_count)
                    else:
                        band_averages.append(0.0)
                        
            else:
                print(f"不支持的数据组织方式: {interleave}")
                return None
                
        return band_averages
                
    except Exception as e:
        print(f"读取数据文件时出错: {str(e)}")
        return None

def save_to_csv(results, output_file, data_directory, label_name, wavelength_info=None):
    """
    将结果保存到CSV文件
    
    Args:
        results: 包含每个文件每个波段平均反射率的字典
        output_file: 输出CSV文件的路径
        data_directory: 数据目录路径
        label_name: 标签，用于标记数据
        wavelength_info: 包含波长信息的字典，键为文件名，值为(波长列表, 波长单位)
    """
    # 找出最大波段数量
    max_bands = 0
    for band_averages in results.values():
        max_bands = max(max_bands, len(band_averages))
    
    # 准备表头
    header = ['label', 'filename']  # 第一列为数据标签，第二列为文件名
    
    # 如果有波长信息，使用波长作为列标题（从第3列开始）
    if wavelength_info and any(wavelength_info.values()):
        # 选择第一个有波长信息的文件
        first_file = next((f for f in wavelength_info if wavelength_info[f][0]), None)
        
        if first_file:
            wavelengths, units = wavelength_info[first_file]
            
            # 只使用数字作为列标题，避免乱码问题
            for wl in wavelengths:
                header.append(f'{wl:.2f}')
        else:
            # 默认波段编号
            header.extend([f'{i+1}' for i in range(max_bands)])
    else:
        # 默认波段编号
        header.extend([f'{i+1}' for i in range(max_bands)])
    
    # 准备数据行
    rows = []
    for filename, band_averages in results.items():
        # 处理文件名：移除.HDR后缀
        base_name = os.path.splitext(os.path.basename(filename))[0]  # 移除扩展名
        
        # 创建一行数据：第一列为数据集标签，第二列为文件名
        row = [label_name, base_name]
        
        # 从第3列开始添加波段数据
        row.extend(band_averages)
        
        rows.append(row)
    
    # 判断输出文件是否存在
    file_exists = os.path.isfile(output_file)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 写入CSV文件
    mode = 'a' if file_exists else 'w'  # 如果文件存在则追加，否则创建新文件
    with open(output_file, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writerow(header)
            print(f"创建新文件 {output_file} 并写入表头")
        else:
            print(f"文件 {output_file} 已存在，添加新数据")
        
        # 写入数据行
        writer.writerows(rows)
    
    print(f"已将 {len(rows)} 行数据保存到 {output_file}")

def main():
    # 数据目录路径
    data_directory = "Data/Leaf"
    # 提取数据集名称(文件夹名)
    label_name = os.path.basename(data_directory)
    # 输出文件路径 - 固定使用同一个文件名
    # output_file = "Data/band_average_reflectivity_results.csv"
    output_file = "Data/leaf.csv"
    
    # 提取每个波段的平均反射率
    results = extract_average_reflectivity(data_directory)
    
    # 简化波长信息收集：只从第一个HDR文件中读取
    wavelength_info = {}
    if results:
        # 获取第一个HDR文件
        first_hdr_file = next(iter(results.keys()))
        hdr_path = os.path.join(data_directory, first_hdr_file)
        
        # 读取元数据
        metadata = read_hdr_metadata(hdr_path)
        if metadata and metadata['wavelengths']:
            print(f"从文件 {first_hdr_file} 中提取波长信息，用于所有文件")
            # 为所有文件使用相同的波长信息
            wavelengths = metadata['wavelengths']
            units = metadata['wavelength_units']
            for filename in results:
                wavelength_info[filename] = (wavelengths, units)
        else:
            print("无法从第一个文件中提取波长信息")
    
    # 保存结果到CSV文件
    if results:
        save_to_csv(results, output_file, data_directory, label_name, wavelength_info)
        print(f"已处理 {len(results)} 个文件")
        
        # 提示用户可以绘制图表
        print("\n提示: 您可以运行 plot_reflectivity.py 来绘制反射率图")
    else:
        print("未找到任何有效数据")

if __name__ == "__main__":
    main()
