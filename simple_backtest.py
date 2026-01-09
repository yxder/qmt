#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的回测脚本，直接使用模拟数据进行回测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from qmt_strategy.strategy.data_process import DataProcessor
from qmt_strategy.strategy.feature_engineer import FeatureEngineer
from qmt_strategy.strategy.model_train import ModelTrainer
from qmt_strategy.strategy.backtest import Backtester
from qmt_strategy.strategy.monitor import MonitorPanel
from qmt_strategy.utils.logger import setup_logger
from qmt_strategy.utils.simulate_data import generate_stock_list, generate_historical_data

logger = setup_logger()

def create_simple_backtest():
    """创建简化的回测"""
    logger.info("=== 简化回测开始 ===")
    
    # 1. 读取已生成的模拟数据
    logger.info("读取2025年11-12月的模拟数据")
    start_date = "20251101"
    end_date = "20251231"
    
    # 读取已生成的数据文件
    import pickle
    data_file = f"data/raw_data/history/history_{start_date}_{end_date}.pkl"
    with open(data_file, 'rb') as f:
        historical_data = pickle.load(f)
    logger.info(f"读取{len(historical_data)}只股票的模拟数据")
    
    # 2. 数据预处理
    logger.info("数据预处理")
    data_processor = DataProcessor()
    processed_data = data_processor.process(historical_data)
    if processed_data is None:
        logger.error("数据预处理失败")
        return
    logger.info(f"预处理后的数据形状: {processed_data.shape}")
    
    # 3. 特征工程
    logger.info("特征工程")
    feature_engineer = FeatureEngineer()
    features = feature_engineer.extract_features(processed_data)
    if features is None:
        logger.error("特征工程失败")
        return
    logger.info(f"提取特征后的数据形状: {features.shape}")
    
    # 4. 模型训练
    logger.info("模型训练")
    model_trainer = ModelTrainer()
    model_trainer.train(features)
    
    # 5. 回测
    logger.info("回测")
    backtester = Backtester()
    backtest_results = backtester.run(features)
    logger.info(f"回测结果: {backtest_results}")
    
    # 6. 生成报告
    logger.info("生成绩效报告")
    monitor = MonitorPanel()
    monitor.update_equity_curve(backtester.equity_curve)
    monitor.update_daily_returns(backtester.daily_returns)
    monitor.update_recent_trades(backtester.trade_records)
    report = monitor.generate_performance_report()
    logger.info(f"绩效报告生成完成，包含{len(report)}个字段")
    
    logger.info("=== 简化回测完成 ===")
    
    return backtest_results

if __name__ == "__main__":
    create_simple_backtest()
