#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试pandas_datareader数据获取功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy.data_fetch import DataFetcher
from utils.logger import setup_logger

logger = setup_logger()

def test_xtdata_historical_data():
    """
    测试从xtdata（QMT）获取2025年历史数据
    """
    import pandas as pd
    logger.info("开始测试xtdata历史数据获取功能")
    
    # 创建数据获取器，使用本地数据优先
    fetcher = DataFetcher(use_local_data=True)
    
    # 获取2025年1月1日至2025年12月31日的历史数据
    start_date = "20250101"
    end_date = "20251231"
    
    logger.info(f"从xtdata获取{start_date}至{end_date}的历史数据")
    
    # 使用xtdata获取历史数据
    data = fetcher.get_historical_data(start_date, end_date, market="SH")
    
    if data:
        logger.info(f"成功获取{len(data)}只股票的历史数据")
        
        # 打印前5只股票的数据信息
        count = 0
        for stock_code, stock_data in data.items():
            if count >= 5:  # 只打印5只股票的信息
                break
            logger.info(f"字段：{stock_code}")
            if isinstance(stock_data, dict):
                logger.info(f"数据类型：dict, 包含股票：{list(stock_data.keys())[:3]}...")
            elif isinstance(stock_data, pd.DataFrame):
                logger.info(f"数据形状：{stock_data.shape}")
                logger.info(f"数据列名：{list(stock_data.columns)}")
                logger.info(f"数据前5行：\n{stock_data.head()}")
            count += 1
            
        # 打印整体数据结构
        logger.info(f"数据包含字段：{list(data.keys())}")
        # 获取第一个字段的第一个股票数据
        first_field = list(data.keys())[0]
        if isinstance(data[first_field], dict):
            first_stock = list(data[first_field].keys())[0]
            logger.info(f"{first_field}字段下的第一个股票：{first_stock}")
            logger.info(f"该股票数据类型：{type(data[first_field][first_stock])}")
            logger.info(f"该股票数据：\n{data[first_field][first_stock][:5]}")
        
        logger.info("xtdata数据获取成功，数据已保存到本地")
    else:
        logger.warning("未能获取到任何数据")
    
    logger.info("xtdata历史数据获取测试完成")

if __name__ == "__main__":
    test_xtdata_historical_data()