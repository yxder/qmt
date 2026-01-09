#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查2025年10月数据文件格式的脚本
"""

import os
import sys
import pickle
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 简单的打印日志函数
logger = print

def main():
    """主函数：检查2025年10月数据文件格式"""
    print("=== 检查2025年10月数据文件格式 ===")
    
    # 数据文件路径
    data_file = "data/raw_data/history/history_20251001_20251031.pkl"
    
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        return 1
    
    print(f"打开数据文件: {data_file}")
    
    # 读取数据
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"数据类型: {type(data)}")
    print(f"数据键: {list(data.keys())}")
    
    if isinstance(data, dict):
        # 检查每个键对应的数据
        for key, value in data.items():
            print(f"\n键 '{key}' 的值类型: {type(value)}")
            if isinstance(value, pd.DataFrame):
                print(f"  数据形状: {value.shape}")
                print(f"  数据列: {list(value.columns)}")
                print(f"  数据索引: {value.index[:5]}")
                print(f"  数据示例:\n{value.head()}")
            elif isinstance(value, dict):
                print(f"  子键: {list(value.keys())[:10]}")
            elif isinstance(value, list):
                print(f"  列表长度: {len(value)}")
    
    print("\n=== 数据格式检查完成 ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
