#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测验证脚本：使用训练好的模型对2025年11-12月的历史数据进行回测验证
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import pandas as pd
import numpy as np
from strategy.backtest import Backtester
from strategy.feature_engineer import FeatureEngineer
from utils.logger import setup_logger
import config

logger = setup_logger()

def load_historical_data():
    """加载历史数据"""
    logger.info("加载历史数据")
    
    data_file = "data/raw_data/history/history_20251101_20251231.pkl"
    if not os.path.exists(data_file):
        logger.error(f"数据文件不存在：{data_file}")
        return None
    
    with open(data_file, 'rb') as f:
        historical_data = pickle.load(f)
    
    logger.info(f"加载了{len(historical_data)}只股票的数据")
    logger.info(f"历史数据示例：{list(historical_data.keys())[:5]}")
    return historical_data

def prepare_backtest_data(historical_data):
    """准备回测数据"""
    logger.info("准备回测数据")
    
    # 统计涨停股票数量
    limit_up_count = 0
    total_days = 0
    
    # 首先转换所有数据格式，合并成一个大的DataFrame
    all_dfs = []
    logger.info(f"开始处理{len(historical_data)}只股票的数据")
    
    for stock, data in historical_data.items():
        try:
            # 转换数据格式
            df = pd.DataFrame({
                'time': pd.to_datetime(data['time']),
                'open': data['open'],
                'close': data['close'],
                'high': data['high'],
                'low': data['low'],
                'volume': data['volume'],
                'stock_code': stock
            })
            
            total_days += len(df)
            
            # 计算涨跌幅作为标签
            # 先计算前收盘价
            df['prev_close'] = df['close'].shift(1)
            # 计算涨跌幅（相对于前收盘价）
            df['return'] = (df['close'] - df['prev_close']) / df['prev_close']
            # 当日涨幅>=9.95%标记为涨停
            df['label'] = (df['return'] >= 0.0995).astype(int)
            
            # 统计涨停股票
            stock_limit_up = df['label'].sum()
            limit_up_count += stock_limit_up
            
            # 打印更详细的信息
            if stock_limit_up > 0:
                logger.info(f"股票{stock}的涨停天数：{stock_limit_up}，总天数：{len(df)}")
                logger.info(f"涨停日详情：{df[df['label'] == 1][['time', 'return']]}")
            
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"处理股票{stock}时出错：{e}")
            continue
    
    logger.info(f"总共有{limit_up_count}个涨停日，总交易天数：{total_days}")
    logger.info(f"处理了{len(all_dfs)}只股票")
    
    if limit_up_count == 0:
        logger.warning("没有检测到涨停股票，这可能导致模型无法学习到有效的涨停模式")
    
    if not all_dfs:
        logger.error("没有生成任何数据")
        return None
    
    # 合并所有股票的数据
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"合并后的数据总行数：{len(combined_df)}")
    
    # 提取特征（使用简化的特征提取方法，不进行复杂的特征选择）
    logger.info("开始提取特征")
    feature_engineer = FeatureEngineer()
    
    # 为每只股票提取特征
    all_features = []
    for stock, group in combined_df.groupby('stock_code'):
        try:
            # 生成特征（跳过复杂的特征选择和重要性评估）
            features = feature_engineer.extract_features_simple(group)
            if features is not None and not features.empty:
                # 添加股票代码和日期信息
                features['stock_code'] = stock
                features['date'] = group['time'].dt.date.values
                features['label'] = group['label'].values
                # 处理NaN值
                features = features.fillna(0)  # 将NaN值填充为0
                all_features.append(features)
        except Exception as e:
            logger.error(f"为股票{stock}提取特征时出错：{e}")
            continue
    
    if not all_features:
        logger.error("没有生成任何特征数据")
        return None
    
    # 合并所有股票的特征数据
    features_df = pd.concat(all_features, ignore_index=True)
    logger.info(f"生成了{len(features_df)}条回测特征数据")
    logger.info(f"特征数据中涨停标签数量：{features_df['label'].sum()}")
    
    # 再次处理NaN值，确保没有遗漏
    features_df = features_df.fillna(0)
    
    return features_df

def filter_numeric_features(features_df):
    """过滤出数值特征，用于模型预测"""
    logger.info("过滤出数值特征")
    
    # 保存股票代码和日期信息
    stock_codes = features_df['stock_code'] if 'stock_code' in features_df.columns else None
    dates = features_df['date'] if 'date' in features_df.columns else None
    
    # 移除非数值特征，只保留数值特征用于模型预测
    numeric_features = features_df.select_dtypes(include=[np.number])
    logger.info(f"移除非数值特征，保留{len(numeric_features.columns)}个数值特征")
    
    return numeric_features, stock_codes, dates

def run_backtest(features_df):
    """运行回测"""
    logger.info("=== 开始回测验证 ===")
    
    if features_df is None or features_df.empty:
        logger.error("回测数据为空，无法进行回测")
        return None
    
    # 初始化回测器
    backtester = Backtester()
    
    # 运行回测
    results = backtester.run(features_df)
    
    logger.info("=== 回测完成 ===")
    return results

def main():
    """主函数"""
    logger.info("=== 开始回测验证流程 ===")
    
    # 1. 加载历史数据
    historical_data = load_historical_data()
    if historical_data is None:
        return 1
    
    # 处理全量股票池
    logger.info(f"原始数据包含{len(historical_data)}只股票，处理全量股票池")
    
    # 2. 准备回测数据
    features_df = prepare_backtest_data(historical_data)
    if features_df is None:
        return 1
    
    # 3. 运行回测
    results = run_backtest(features_df)
    if results is None:
        return 1
    
    # 4. 输出回测结果
    logger.info("=== 回测结果 ===")
    for key, value in results.items():
        logger.info(f"{key}: {value}")
    
    logger.info("=== 回测验证流程完成 ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
