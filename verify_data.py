#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据验证脚本，用于检查数据获取功能是否正常工作
"""

import os
import pandas as pd
import sys

# 设置Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qmt_strategy.strategy.data_fetch import DataFetcher
from qmt_strategy.utils.logger import setup_logger

# 设置日志
logger = setup_logger()

def verify_data():
    """验证数据获取功能"""
    logger.info("开始验证数据获取功能")
    
    # 1. 检查历史数据文件是否存在
    history_data_dir = "F:\\projects\\qmt\\data\\raw_data\\history"
    if not os.path.exists(history_data_dir):
        logger.error(f"历史数据目录不存在: {history_data_dir}")
        return False
    
    logger.info(f"历史数据目录存在: {history_data_dir}")
    
    # 2. 检查是否有数据文件
    data_files = [f for f in os.listdir(history_data_dir) if f.endswith('.pkl')]
    if not data_files:
        logger.error(f"历史数据目录中没有数据文件: {history_data_dir}")
        return False
    
    logger.info(f"历史数据目录中找到 {len(data_files)} 个数据文件")
    for f in data_files:
        logger.info(f"  - {f}")
    
    # 3. 检查配置的回测数据文件是否存在
    backtest_file = "history_20251101_20251231.pkl"
    if backtest_file in data_files:
        logger.info(f"配置的回测数据文件存在: {backtest_file}")
        
        # 4. 读取并验证数据文件内容
        file_path = os.path.join(history_data_dir, backtest_file)
        try:
            data = pd.read_pickle(file_path)
            logger.info(f"成功读取数据文件: {file_path}")
            logger.info(f"数据文件包含 {len(data)} 只股票的数据")
            
            # 5. 检查数据结构
            if data:
                sample_stock = next(iter(data.keys()))
                sample_data = data[sample_stock]
                logger.info(f"示例股票: {sample_stock}")
                
                # 检查数据类型
                if isinstance(sample_data, dict):
                    logger.info(f"示例数据类型: dict")
                    logger.info(f"示例数据键: {list(sample_data.keys())}")
                    
                    # 检查是否包含数据字段
                    if 'open' in sample_data:
                        logger.info(f"开盘价数据长度: {len(sample_data['open'])}")
                elif isinstance(sample_data, pd.DataFrame):
                    logger.info(f"数据形状: {sample_data.shape}")
                    logger.info(f"数据列: {sample_data.columns.tolist()}")
                    logger.info(f"日期范围: {sample_data.index.min()} 到 {sample_data.index.max()}")
                else:
                    logger.info(f"示例数据类型: {type(sample_data).__name__}")
            
        except Exception as e:
            logger.error(f"读取数据文件失败: {e}")
            return False
    else:
        logger.warning(f"配置的回测数据文件不存在: {backtest_file}")
    
    # 6. 测试DataFetcher类
    logger.info("测试DataFetcher类")
    try:
        fetcher = DataFetcher(use_local_data=True)
        logger.info(f"DataFetcher初始化成功")
        logger.info(f"获取到股票列表，共{len(fetcher.stock_list)}只股票")
        
        # 7. 测试获取指定日期范围的数据
        start_date = "20251101"
        end_date = "20251231"
        logger.info(f"测试获取 {start_date} 到 {end_date} 的数据")
        
        data = fetcher.get_historical_data(start_date, end_date, "SH")
        logger.info(f"获取到 {len(data)} 只股票的数据")
        
        # 8. 测试获取DataFrame格式的数据
        df = fetcher.get_historical_data_dataframe(start_date, end_date, "SH")
        logger.info(f"获取到DataFrame格式数据，形状: {df.shape}")
        
        # 9. 检查数据质量
        if not df.empty:
            logger.info(f"数据列: {df.columns.tolist()}")
            logger.info(f"股票数量: {df['stock_code'].nunique()}")
            logger.info(f"日期范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
            logger.info(f"数据记录数: {len(df)}")
    except Exception as e:
        logger.error(f"测试DataFetcher类失败: {e}")
        return False
    
    logger.info("数据验证完成，数据获取功能正常工作")
    return True

if __name__ == "__main__":
    success = verify_data()
    if success:
        logger.info("数据获取功能验证通过")
        sys.exit(0)
    else:
        logger.error("数据获取功能验证失败")
        sys.exit(1)
