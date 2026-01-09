#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块测试脚本
"""

import os
import sys

# 设置Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qmt_strategy.strategy.data_fetch import DataFetcher
from qmt_strategy.strategy.data_process import DataProcessor
from qmt_strategy.utils.logger import setup_logger

# 设置日志
logger = setup_logger()

def test_data_process():
    """测试数据预处理功能"""
    logger.info("开始测试数据预处理功能")
    
    # 1. 初始化数据获取器和预处理器
    fetcher = DataFetcher(use_local_data=True)
    processor = DataProcessor()
    
    # 2. 获取测试数据
    start_date = "20251101"
    end_date = "20251231"
    logger.info(f"获取 {start_date} 到 {end_date} 的测试数据")
    
    # 使用模拟数据，因为真实数据可能无法获取
    history_data = fetcher.get_historical_data(start_date, end_date, "SH")
    logger.info(f"获取到 {len(history_data)} 只股票的历史数据")
    
    # 3. 测试数据预处理
    logger.info("测试数据预处理功能")
    processed_data = processor.process(history_data)
    
    if processed_data is not None:
        logger.info(f"数据预处理成功，结果形状: {processed_data.shape}")
        logger.info(f"数据列: {processed_data.columns.tolist()}")
        logger.info(f"数据行数: {len(processed_data)}")
        logger.info(f"缺失值情况:")
        logger.info(processed_data.isnull().sum())
        
        # 检查是否包含必要列
        required_cols = ['stock_code', 'label', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col in processed_data.columns:
                logger.info(f"✓ 包含必要列: {col}")
            else:
                logger.warning(f"✗ 缺少必要列: {col}")
    else:
        logger.error("数据预处理失败")
    
    # 4. 测试集合竞价数据处理
    logger.info("测试集合竞价数据处理功能")
    bid_data = fetcher.get_bid_data(start_date)
    logger.info(f"获取到 {len(bid_data)} 只股票的集合竞价数据")
    
    # 转换集合竞价数据为DataFrame格式，以便测试处理功能
    all_bid_data = []
    for stock_code, stock_data in bid_data.items():
        if isinstance(stock_data, dict):
            df = pd.DataFrame(stock_data)
        else:
            df = stock_data.copy()
        df['stock_code'] = stock_code
        all_bid_data.append(df)
    
    if all_bid_data:
        combined_bid_data = pd.concat(all_bid_data, ignore_index=True)
        processed_bid_data = processor.process_bid_data(combined_bid_data)
        
        if processed_bid_data is not None:
            logger.info(f"集合竞价数据处理成功，结果形状: {processed_bid_data.shape}")
            logger.info(f"集合竞价数据列: {processed_bid_data.columns.tolist()}")
        else:
            logger.error("集合竞价数据处理失败")
    
    logger.info("数据预处理功能测试完成")

if __name__ == "__main__":
    import pandas as pd
    test_data_process()
