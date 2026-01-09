#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成真实的模拟数据用于回测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from qmt_strategy.utils.simulate_data import generate_stock_list, generate_historical_data, generate_bid_data
from qmt_strategy.utils.logger import setup_logger

logger = setup_logger()

def generate_realistic_backtest_data():
    """生成真实的回测数据"""
    logger.info("=== 生成真实模拟回测数据开始 ===")
    
    # 生成100只股票的模拟数据
    logger.info("生成股票列表")
    stock_codes = generate_stock_list(n=100)
    start_date = "20251101"
    end_date = "20251231"
    
    logger.info(f"生成{len(stock_codes)}只股票的历史数据")
    # 生成历史数据
    historical_data = generate_historical_data(stock_codes, start_date, end_date)
    
    # 保存数据到本地，模拟数据获取模块的输出格式
    import pickle
    os.makedirs("data/raw_data/history", exist_ok=True)
    data_file = f"data/raw_data/history/history_{start_date}_{end_date}.pkl"
    with open(data_file, 'wb') as f:
        pickle.dump(historical_data, f)
    
    logger.info(f"真实模拟数据生成完成，共{len(historical_data)}只股票，保存到{data_file}")
    
    # 验证数据格式
    logger.info("验证数据格式")
    for stock, data in historical_data.items():
        if 'time' in data and len(data['time']) > 0:
            logger.info(f"✓ {stock} 数据格式正确，共{len(data['time'])}行")
            break
    
    logger.info("=== 生成真实模拟回测数据完成 ===")
    
    return historical_data

if __name__ == "__main__":
    generate_realistic_backtest_data()
