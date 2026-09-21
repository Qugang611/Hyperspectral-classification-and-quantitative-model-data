"""
高光谱反射率数据可视化脚本

功能描述：
    读取高光谱反射率CSV数据文件，绘制所有样本的反射率曲线图，
    不同类别使用不同颜色区分，支持波长信息自动识别。

主要功能：
    1. 自动检测CSV文件编码（utf-8, latin1, gbk, gb18030, cp1252）
    2. 从CSV表头提取波长信息（或使用波段索引）
    3. 按类别自动分配颜色（使用matplotlib的tab10配色）
    4. 绘制所有样本的反射率曲线（同一类别使用相同颜色）
    5. 智能设置x轴刻度（根据波长范围自动调整刻度间隔）
    6. 生成水平排列的图例

输入文件格式：
    CSV文件结构（UTF-8编码）：
    - 第1列：label（类别标签，如"CEES"、"Control"等）
    - 第2列：filename（样本文件名）
    - 第3-258列：波段反射率值（列名为波长，如"900.0"、"901.5"等）
    
    示例：
    label,filename,900.0,901.5,903.0,...
    CEES,sample1.hdr,0.234,0.256,0.278,...
    Control,sample2.hdr,0.189,0.201,0.215,...

输出：
    - PDF图表：plot/Reflectance_all_samples_plot_3.pdf
    - 特点：透明背景、紧凑布局、高质量矢量图

技术特性：
    - 自动波长范围检测（850-1710nm）
    - 智能刻度间隔（>1000nm用100nm，500-1000nm用50nm，<500nm用25nm）
    - 类别自动配色（最多支持10个类别，循环使用tab10配色）
    - 水平排列图例，自动适应类别数量

依赖库：
    - pandas: CSV数据读取
    - matplotlib: 绘图
    - numpy: 数值计算
    - utils.plot_setting: 自定义图表样式

作者：Maggie Zhang
版本：v1.0
最后修改：2025年12月16日
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from utils.plot_setting import setfig

def plot_reflectivity_with_wavelength(csv_file):
    # ========== 第1步：文件读取与编码检测 ==========
    if not os.path.exists(csv_file):
        print(f"文件 {csv_file} 不存在")
        return
    
    df = None
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print("成功使用utf-8编码读取CSV文件")
    except UnicodeDecodeError:
        # 备用编码列表（适用于不同来源的数据）
        encodings = ['latin1', 'gbk', 'gb18030', 'cp1252']
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取CSV文件")
                break
            except UnicodeDecodeError:
                continue
    
    # 验证文件读取结果
    if df is None or df.empty:
        print("无法读取CSV文件或文件为空")
        return
    
    # 打印文件基本信息（用于调试）
    print("CSV文件的列名:", df.columns.tolist())
    print(f"CSV文件共有 {len(df.columns)} 列")
    
    # ========== 第2步：数据结构验证与列识别 ==========
    # 验证至少有3列（label, filename, 至少1个波段）
    if len(df.columns) < 3:
        print("CSV文件列数不足，至少需要3列")
        return
        
    # 第一列是label（类别标签，如CEES、Control等）
    label_column = df.columns[0]
    print(f"使用第一列 '{label_column}' 作为标签列")
    
    # 第二列是filename（样本文件名）
    filename_column = df.columns[1]
    print(f"使用第二列 '{filename_column}' 作为文件名列")
    
    # 第三列到第258列是波段数据（最多256个波段）
    band_columns = df.columns[2:256+1] if len(df.columns) >= 256 else df.columns[2:]
    num_bands = len(band_columns)
    
    if num_bands == 0:
        print("CSV文件中没有波段数据")
        return
        
    print(f"检测到 {num_bands} 个波段，从第3列到第{2+num_bands}列")
    
    # ========== 第3步：波长信息提取与X轴设置 ==========
    try:
        x_values = [float(col) for col in band_columns]
        x_label = 'Wavelength (nm)'
        print(f"成功从CSV表头提取波长信息: {min(x_values)}-{max(x_values)} nm")
        min_x = min(x_values)
        max_x = max(x_values)
        range_x = max_x - min_x
        

        if range_x > 1000:
            step = 100  # 大范围（>1000nm）：每100nm一个刻度
        elif range_x > 500:
            step = 50   # 中等范围（500-1000nm）：每50nm一个刻度
        else:
            step = 25   # 小范围（<500nm）：每25nm一个刻度
            
        # 生成对齐到step倍数的刻度位置
        x_ticks = np.arange(np.floor(min_x / step) * step, 
                           np.ceil(max_x / step) * step + step, 
                           step)
        
    except ValueError:
        x_values = np.arange(1, num_bands + 1)
        x_label = 'Band Index'
        print("无法从列名提取波长信息，将使用波段索引")
        
        x_ticks = np.arange(0, num_bands + 1, max(1, num_bands // 20))
    
    # ========== 第4步：类别分组与颜色分配 ==========
    unique_labels = df[label_column].unique()
    print(f"找到 {len(unique_labels)} 个标签: {', '.join(str(label) for label in unique_labels)}")
    
    # ========== 第5步：创建图形并设置颜色映射 ==========
    fig = setfig(column=3, x=7, y=2)

    if hasattr(mpl, 'colormaps') and 'tab10' in mpl.colormaps:
        cmap = mpl.colormaps['tab10']
    else:
        cmap = plt.cm.tab10
    
    # ========== 第6步：绘制所有样本的反射率曲线 ==========
    for i, (_, row) in enumerate(df.iterrows()):

        label_val = row[label_column]
        filename = row[filename_column]

        label_idx = np.where(unique_labels == label_val)[0][0]

        values = [float(row[col]) if pd.notna(row[col]) and row[col] != '' else np.nan 
                 for col in band_columns]

        color = cmap(label_idx % 10)  # tab10有10种颜色，超过10个类别会循环
        plt.plot(x_values, values, color=color, linewidth=1, alpha=0.7)
    
    # ========== 第7步：设置图表属性和样式 ==========
    plt.xlabel(x_label)
    plt.ylabel('Reflectance')
    
    plt.xticks(x_ticks)
    
    plt.xlim(850, 1710)
    
    plt.setp(plt.gca().get_xticklabels(), 
             fontsize=8,            # 字体大小
             fontweight='normal',   # 字体粗细（normal/bold）
             color='black',         # 文本颜色
            #  rotation=45,         # 旋转角度（已注释，保持水平）
             ha='center',           # 水平对齐：居中
             va='top',              # 垂直对齐：顶部
             visible=True,          # 确保标签可见
             alpha=0.8)             # 透明度
    
    # ========== 第8步：创建图例 ==========
    handles = []
    legend_labels = []
    for i, label_val in enumerate(unique_labels):
        color = cmap(i % 10)
        line, = plt.plot([], [], color=color, linewidth=2)
        handles.append(line)
        legend_labels.append(label_val)

    plt.legend(handles, legend_labels, 
               prop={'size': 8, 'family': 'Arial'},  # 字体设置
               loc='upper left',              # 基准位置：左上角
               frameon=False,                 # 无边框
               borderpad=0.1,                 # 图例内边距
               handlelength=0.5,              # 图例符号长度
               labelspacing=0.3,              # 图例项之间的间距
               ncol=len(legend_labels),       # 列数=标签数量（水平排列）
               columnspacing=0.8,             # 列间距
               bbox_to_anchor=(0.02, 1.0),    # 精确位置调整
               handletextpad=0.5)             # 符号和文本间距
    
    # ========== 第9步：保存图形 ==========
    os.makedirs('plot', exist_ok=True)


    plt.savefig('plot/Reflectance_all_samples_plot_3.pdf', 
               format='PDF', 
               transparent=True,      # 透明背景
               bbox_inches='tight')   # 紧凑布局（去除多余空白）
    
    plt.show()

def main():
    # 数据CSV文件路径
    csv_file = "Data/band_average_reflectivity_results_3.csv"
    
    # 绘制反射率图
    plot_reflectivity_with_wavelength(csv_file)


if __name__ == "__main__":
    main()
