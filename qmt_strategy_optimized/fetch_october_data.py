#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从QMT获取2025年10月数据的脚本
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy.data_fetch import DataFetcher
from utils.logger import setup_logger
import config

logger = setup_logger()

def main():
    """主函数：从QMT获取2025年10月数据"""
    logger.info("=== 开始从QMT获取2025年10月数据 ===")
    
    # 创建数据获取器
    fetcher = DataFetcher(use_local_data=False)  # 不使用本地数据，强制从QMT获取
    
    # 设置10月的日期范围
    start_date = "20251001"
    end_date = "20251031"
    
    logger.info(f"获取日期范围：{start_date} 到 {end_date}")
    
    # 获取历史数据
    data = fetcher.get_historical_data(
        start_date=start_date,
        end_date=end_date,
        market="ALL",  # 获取所有市场
        batch_size=50
    )
    
    logger.info(f"数据获取完成，共获取 {len(data)} 只股票的数据")
    
    # 打印数据信息
    if data:
        # 打印部分股票的数据信息
        sample_stocks = list(data.keys())[:5]
        for stock_code in sample_stocks:
            stock_data = data[stock_code]
            if isinstance(stock_data, dict):
                logger.info(f"股票 {stock_code} 的数据类型：dict，包含字段：{list(stock_data.keys())}")
            elif isinstance(stock_data, list):
                logger.info(f"股票 {stock_code} 的数据类型：list，长度：{len(stock_data)}")
            else:
                logger.info(f"股票 {stock_code} 的数据类型：{type(stock_data)}")
    
    logger.info("=== 2025年10月数据获取完成 ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
