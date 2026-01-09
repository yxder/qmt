#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据获取模块测试脚本
"""

import os
import sys

# 设置Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qmt_strategy.strategy.data_fetch import DataFetcher
from qmt_strategy.utils.logger import setup_logger

# 设置日志
logger = setup_logger()

def test_data_fetch():
    """测试数据获取功能"""
    logger.info("开始测试数据获取功能")
    
    # 初始化数据获取器
    fetcher = DataFetcher(use_local_data=True)
    
    # 1. 测试获取股票列表
    logger.info(f"获取到股票列表，共{len(fetcher.stock_list)}只股票")
    logger.info(f"前10只股票：{fetcher.stock_list[:10]}")
    
    # 2. 测试获取历史数据
    start_date = "20251101"
    end_date = "20251231"
    logger.info(f"测试获取 {start_date} 到 {end_date} 的历史数据")
    
    history_data = fetcher.get_historical_data(start_date, end_date, "SH")
    logger.info(f"获取到 {len(history_data)} 只股票的历史数据")
    
    # 3. 测试获取DataFrame格式的历史数据
    df = fetcher.get_historical_data_dataframe(start_date, end_date, "SH")
    logger.info(f"获取到DataFrame格式数据，形状: {df.shape}")
    if not df.empty:
        logger.info(f"数据列: {df.columns.tolist()}")
        logger.info(f"股票数量: {df['stock_code'].nunique()}")
        logger.info(f"日期范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
    
    # 4. 测试获取集合竞价数据
    test_date = "20251101"
    logger.info(f"测试获取 {test_date} 的集合竞价数据")
    
    bid_data = fetcher.get_bid_data(test_date)
    logger.info(f"获取到 {len(bid_data)} 只股票的集合竞价数据")
    
    if bid_data:
        sample_stock = next(iter(bid_data.keys()))
        sample_df = bid_data[sample_stock]
        logger.info(f"示例股票: {sample_stock}")
        logger.info(f"集合竞价数据形状: {sample_df.shape}")
        logger.info(f"集合竞价数据列: {sample_df.columns.tolist()}")
        logger.info(f"时间范围: {sample_df.index.min()} 到 {sample_df.index.max()}")
    
    # 5. 测试获取市场情绪数据
    logger.info(f"测试获取 {test_date} 的市场情绪数据")
    market_sentiment = fetcher.get_market_sentiment_data(test_date)
    if market_sentiment:
        logger.info("成功获取市场情绪数据")
    
    logger.info("数据获取功能测试完成")

if __name__ == "__main__":
    test_data_fetch()
