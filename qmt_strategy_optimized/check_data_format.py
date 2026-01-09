#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os

# 检查历史数据文件的格式
data_file = "data/raw_data/history/history_20250101_20251231.pkl"

if os.path.exists(data_file):
    print(f"打开文件: {data_file}")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"数据类型: {type(data)}")
    print(f"数据键: {list(data.keys())}")
    
    # 如果是字典，检查每个键对应的值
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"\n键 '{key}' 的值类型: {type(value)}")
            if isinstance(value, dict):
                print(f"  子键: {list(value.keys())}")
                # 检查子键是否包含时间和数据
                if 'time' in value:
                    print(f"  时间数据长度: {len(value['time'])}")
                if 'open' in value:
                    print(f"  开盘价数据长度: {len(value['open'])}")
    
    # 检查另一个数据文件
    print("\n\n检查另一个数据文件:")
    data_file2 = "data/raw_data/history/history_20251101_20251231.pkl"
    if os.path.exists(data_file2):
        with open(data_file2, 'rb') as f:
            data2 = pickle.load(f)
        print(f"数据类型: {type(data2)}")
        print(f"数据键: {list(data2.keys())[:10]}")
        
        # 检查第一个键对应的数据
        if isinstance(data2, dict) and data2:
            first_key = list(data2.keys())[0]
            first_data = data2[first_key]
            print(f"\n第一个键 '{first_key}' 的值类型: {type(first_data)}")
            if isinstance(first_data, dict):
                print(f"  子键: {list(first_data.keys())}")
                # 检查子键是否包含时间和数据
                if 'time' in first_data:
                    print(f"  时间数据长度: {len(first_data['time'])}")
                    print(f"  时间数据示例: {first_data['time'][:5]}")
                if 'open' in first_data:
                    print(f"  开盘价数据长度: {len(first_data['open'])}")
                    print(f"  开盘价数据示例: {first_data['open'][:5]}")
else:
    print(f"文件不存在: {data_file}")
