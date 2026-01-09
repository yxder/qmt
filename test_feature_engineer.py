#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征工程模块测试脚本
"""

import os
import sys

# 设置Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qmt_strategy.strategy.data_fetch import DataFetcher
from qmt_strategy.strategy.data_process import DataProcessor
from qmt_strategy.strategy.feature_engineer import FeatureEngineer
from qmt_strategy.utils.logger import setup_logger

# 设置日志
logger = setup_logger()

def test_feature_engineer():
    """测试特征工程功能"""
    logger.info("开始测试特征工程功能")
    
    # 1. 初始化数据获取器、预处理器和特征工程师
    fetcher = DataFetcher(use_local_data=True)
    processor = DataProcessor()
    engineer = FeatureEngineer()
    
    # 2. 生成测试数据
    # 使用模拟数据，因为真实数据可能无法获取
    # 生成一个包含必要列的测试DataFrame
    import pandas as pd
    import numpy as np
    
    # 创建一个测试DataFrame
    test_data = pd.DataFrame()
    test_data['stock_code'] = [f"SH{str(i).zfill(6)}" for i in range(100)] + [f"SZ{str(i).zfill(6)}" for i in range(100)]
    test_data['label'] = np.random.choice([0, 1], size=200, p=[0.9, 0.1])
    test_data['open'] = np.random.uniform(10, 20, 200)
    test_data['high'] = np.random.uniform(10, 20, 200)
    test_data['low'] = np.random.uniform(10, 20, 200)
    test_data['close'] = np.random.uniform(10, 20, 200)
    test_data['volume'] = np.random.randint(1000, 100000, 200)
    test_data['amount'] = test_data['close'] * test_data['volume']
    
    # 模拟集合竞价数据
    test_data['bid_price_915'] = test_data['open'] * np.random.normal(1, 0.01, 200)
    test_data['bid_price_920'] = test_data['open'] * np.random.normal(1, 0.01, 200)
    test_data['bid_price_925'] = test_data['open']
    test_data['bid_volume_915'] = test_data['volume'] * np.random.uniform(0.5, 1.0, 200)
    test_data['bid_volume_920'] = test_data['volume'] * np.random.uniform(0.7, 1.2, 200)
    test_data['bid_volume_925'] = test_data['volume'] * np.random.uniform(0.8, 1.5, 200)
    test_data['buy_order_size_925'] = test_data['volume'] * np.random.uniform(0.1, 0.5, 200)
    test_data['prev_close'] = test_data['close'] * np.random.normal(1, 0.02, 200)
    test_data['circulating_cap'] = np.random.randint(100000000, 1000000000, 200, dtype=np.int64)
    
    # 模拟板块数据
    test_data['sector_change'] = np.random.normal(0, 0.02, 200)
    test_data['sector_money_flow'] = np.random.uniform(-100000000, 100000000, 200)
    test_data['sector_rank'] = np.random.randint(1, 100, 200)
    test_data['sector_limit_up_count'] = np.random.randint(0, 20, 200)
    
    # 模拟市场情绪数据
    test_data['index_change'] = np.random.normal(0, 0.01, 200)
    test_data['up_down_ratio'] = np.random.uniform(0.5, 2.0, 200)
    test_data['profit_effect'] = np.random.normal(0, 0.1, 200)
    test_data['market_volume'] = np.random.uniform(1000000000, 10000000000, 200)
    test_data['market_volume_ratio'] = np.random.uniform(0.5, 2.0, 200)
    
    # 模拟基本面数据
    test_data['circulating_market_cap'] = np.random.uniform(1000000000, 10000000000, 200)
    test_data['pe_ratio'] = np.random.uniform(10, 50, 200)
    test_data['pb_ratio'] = np.random.uniform(1, 5, 200)
    test_data['earnings_announcement'] = np.random.choice([0, 1], size=200, p=[0.95, 0.05])
    test_data['turnover_rate'] = test_data['volume'] / test_data['circulating_market_cap'] * 100
    test_data['volatility_5d'] = np.random.normal(0, 0.05, 200)
    
    # 模拟技术指标
    test_data['ma5'] = test_data['close'].rolling(window=5).mean()
    test_data['ma10'] = test_data['close'].rolling(window=10).mean()
    test_data['macd'] = np.random.normal(0, 0.1, 200)
    test_data['macd_signal'] = np.random.normal(0, 0.1, 200)
    test_data['rsi_14'] = np.random.uniform(30, 70, 200)
    test_data['volume_ratio'] = np.random.uniform(0.5, 2.0, 200)
    test_data['avg_volume_5d'] = test_data['volume'].rolling(window=5).mean()
    
    logger.info(f"生成测试数据，形状: {test_data.shape}")
    logger.info(f"测试数据列: {test_data.columns.tolist()}")
    
    # 3. 测试特征提取
    logger.info("测试特征提取功能")
    features = engineer.extract_features(test_data)
    
    if features is not None:
        logger.info(f"特征提取成功，结果形状: {features.shape}")
        logger.info(f"提取到 {len(features.columns)} 个特征")
        logger.info(f"特征列: {features.columns.tolist()}")
        
        # 检查是否包含label列
        if 'label' in features.columns:
            logger.info("✓ 包含label列")
        else:
            logger.warning("✗ 缺少label列")
        
        # 检查特征数量是否达到要求（至少20个）
        if len(features.columns) >= 20:
            logger.info(f"✓ 特征数量满足要求，共 {len(features.columns)} 个特征")
        else:
            logger.warning(f"✗ 特征数量不足，仅 {len(features.columns)} 个特征")
        
        # 检查缺失值情况
        missing_values = features.isnull().sum()
        if missing_values.sum() == 0:
            logger.info("✓ 无缺失值")
        else:
            logger.warning(f"✗ 存在缺失值: {missing_values.sum()}")
            logger.info(f"缺失值详情: {missing_values[missing_values > 0]}")
    else:
        logger.error("特征提取失败")
    
    logger.info("特征工程功能测试完成")

if __name__ == "__main__":
    test_feature_engineer()
