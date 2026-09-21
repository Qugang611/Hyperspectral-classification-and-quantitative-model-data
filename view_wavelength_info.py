"""
view_wavelength_info.py

本脚本用于批量提取指定目录下所有HDR文件的波长信息。

主要功能：
extract_wavelength_info(directory_path):
   - 扫描目录下所有.HDR文件，解析并提取每个文件的波长列表、单位、最小/最大波长、波段数等信息。
   - 支持中文输出，便于在中文环境下查看。

使用方法：
- 修改 main() 中的 data_directory 路径为实际HDR文件所在目录。
- 运行脚本后，将在终端输出每个文件的波长信息。

注意事项：
- 需安装 matplotlib、numpy 等依赖。
- 若目录或文件不存在，将有相应提示。
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置matplotlib支持中文，防止中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def extract_wavelength_info(directory_path):
    """
    提取指定目录中所有HDR文件的波长信息
    
    Args:
        directory_path: 包含HDR文件的目录路径
        
    Returns:
        包含文件名、波长列表和波长单位的字典列表
    """
    results = []
    
    if not os.path.exists(directory_path):
        print(f"目录 {directory_path} 不存在")
        return results
    
    for filename in os.listdir(directory_path):
        if filename.upper().endswith('.HDR'):
            file_path = os.path.join(directory_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    wavelengths = None
                    wavelength_units = None
                    
                    # 尝试查找波长单位
                    unit_match = re.search(r'wavelength\s+units\s*=\s*(\w+)', content, re.IGNORECASE)
                    if unit_match:
                        wavelength_units = unit_match.group(1).strip()
                    
                    # 尝试查找波长列表
                    wave_match = re.search(r'wavelength\s*=\s*\{([^}]+)\}', content, re.IGNORECASE)
                    if wave_match:
                        wave_str = wave_match.group(1).strip()
                        try:
                            # 分割并转换为浮点数
                            wavelengths = [float(w.strip()) for w in wave_str.split(',')]
                            
                            results.append({
                                'filename': filename,
                                'wavelengths': wavelengths,
                                'units': wavelength_units,
                                'min': min(wavelengths),
                                'max': max(wavelengths),
                                'count': len(wavelengths)
                            })
                            
                            print(f"文件 {filename}: {len(wavelengths)} 个波段, 范围 {min(wavelengths)}-{max(wavelengths)} {wavelength_units or ''}")
                        except ValueError:
                            print(f"文件 {filename}: 波长值格式不正确")
                    else:
                        print(f"文件 {filename}: 未找到波长信息")
            
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    return results

def main():
    # 数据目录路径
    data_directory = "Data/Exp1/CEES"
    
    # 提取波长信息
    results = extract_wavelength_info(data_directory)
    
if __name__ == "__main__":
    main()
