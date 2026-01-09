#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
极端行情压力测试脚本：测试策略在极端市场环境下的表现
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from qmt_strategy.strategy.feature_engineer import FeatureEngineer
from qmt_strategy.strategy.model_train import ModelTrainer
from qmt_strategy.strategy.strategy_decision import StrategyDecision
from qmt_strategy.strategy.risk_control import RiskController
from qmt_strategy.utils.logger import setup_logger
import qmt_strategy.config as config

logger = setup_logger()

def generate_extreme_market_data():
    """生成极端行情数据"""
    logger.info("生成极端行情数据")
    
    # 生成不同类型的极端行情数据
    market_types = [
        "normal",          # 正常行情
        "market_crash",    # 大盘暴跌
        "market_rally",    # 大盘暴涨
        "stock_crash",     # 个股暴跌
        "stock_rally",     # 个股暴涨
        "high_volatility"   # 高波动行情
    ]
    
    extreme_data = {}
    
    for market_type in market_types:
        logger.info(f"生成{market_type}行情数据")
        
        # 生成10只股票，每只股票42天的数据
        stocks = [f"TEST{i:04d}.SZ" for i in range(10)]
        market_data = {}
        
        for stock in stocks:
            # 生成基础数据
            dates = pd.date_range(start="2025-11-01", periods=42, freq="D")
            n_days = len(dates)
            
            # 基础价格
            base_price = np.random.uniform(10, 50)
            
            # 根据不同市场类型生成价格走势
            if market_type == "normal":
                # 正常行情：小幅波动
                returns = np.random.normal(0, 0.02, n_days)
            elif market_type == "market_crash":
                # 大盘暴跌：连续下跌
                returns = np.random.normal(-0.05, 0.03, n_days)
                # 第一天大幅下跌
                returns[0] = -0.08
            elif market_type == "market_rally":
                # 大盘暴涨：连续上涨
                returns = np.random.normal(0.05, 0.03, n_days)
                # 第一天大幅上涨
                returns[0] = 0.08
            elif market_type == "stock_crash":
                # 个股暴跌：单只股票大幅下跌
                returns = np.random.normal(0, 0.02, n_days)
                # 前5天连续下跌
                returns[:5] = np.random.normal(-0.06, 0.02, 5)
            elif market_type == "stock_rally":
                # 个股暴涨：单只股票连续涨停
                returns = np.random.normal(0, 0.02, n_days)
                # 前5天连续上涨，包含涨停
                returns[:5] = np.random.normal(0.08, 0.01, 5)
                # 确保第二天涨停
                returns[1] = 0.099
            elif market_type == "high_volatility":
                # 高波动行情：大幅上下波动
                returns = np.random.normal(0, 0.05, n_days)
                # 添加几个极端波动日
                returns[[2, 7, 15, 25, 35]] = [0.09, -0.08, 0.07, -0.09, 0.1]
            
            # 计算价格系列
            prices = [base_price]
            for ret in returns[:-1]:
                next_price = prices[-1] * (1 + ret)
                prices.append(next_price)
            
            prices = np.array(prices)
            
            # 生成最高价和最低价
            high = prices * (1 + np.abs(np.random.normal(0, 0.02, n_days)))
            low = prices * (1 - np.abs(np.random.normal(0, 0.02, n_days)))
            
            # 生成成交量（根据行情调整）
            if market_type in ["market_crash", "market_rally", "high_volatility"]:
                volume = np.random.normal(2000000, 500000, n_days).astype(int)
            else:
                volume = np.random.normal(1000000, 300000, n_days).astype(int)
            
            # 确保成交量为正
            volume = np.maximum(volume, 100000)
            
            # 生成集合竞价相关数据
            bid_price_915 = prices * (1 + np.random.normal(0, 0.01, n_days))
            bid_price_920 = prices * (1 + np.random.normal(0, 0.005, n_days))
            bid_price_925 = prices * (1 + np.random.normal(0, 0.002, n_days))
            
            bid_volume_915 = np.random.normal(500000, 200000, n_days).astype(int)
            bid_volume_920 = np.random.normal(700000, 200000, n_days).astype(int)
            bid_volume_925 = np.random.normal(1000000, 300000, n_days).astype(int)
            
            buy_order_size_925 = np.random.normal(50000, 20000, n_days).astype(int)
            
            # 生成板块和市场数据
            sector_change = np.random.normal(0, 0.02, n_days)
            up_down_ratio = np.random.uniform(0.5, 1.5, n_days)
            
            # 组装数据
            stock_data = {
                'time': dates.strftime("%Y-%m-%d").tolist(),
                'open': prices.tolist(),
                'close': (prices * (1 + returns)).tolist(),
                'high': high.tolist(),
                'low': low.tolist(),
                'volume': volume.tolist(),
                'bid_price_915': bid_price_915.tolist(),
                'bid_price_920': bid_price_920.tolist(),
                'bid_price_925': bid_price_925.tolist(),
                'bid_volume_915': bid_volume_915.tolist(),
                'bid_volume_920': bid_volume_920.tolist(),
                'bid_volume_925': bid_volume_925.tolist(),
                'buy_order_size_925': buy_order_size_925.tolist(),
                'sector_change': sector_change.tolist(),
                'up_down_ratio': up_down_ratio.tolist()
            }
            
            market_data[stock] = stock_data
        
        extreme_data[market_type] = market_data
    
    logger.info("极端行情数据生成完成")
    return extreme_data

def test_strategy_on_extreme_data(extreme_data):
    """在极端行情数据上测试策略"""
    logger.info("在极端行情数据上测试策略")
    
    # 初始化各模块
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    model_trainer.load_model()
    
    if model_trainer.model is None:
        logger.error("无法加载模型，测试失败")
        return
    
    strategy_decision = StrategyDecision()
    risk_controller = RiskController()
    
    # 测试结果
    test_results = {}
    
    for market_type, market_data in extreme_data.items():
        logger.info(f"测试{market_type}行情")
        
        # 处理每只股票
        stock_results = {}
        
        for stock, data in market_data.items():
            try:
                # 转换数据格式
                df = pd.DataFrame({
                    'time': pd.to_datetime(data['time']),
                    'open': data['open'],
                    'close': data['close'],
                    'high': data['high'],
                    'low': data['low'],
                    'volume': data['volume'],
                    'bid_price_915': data['bid_price_915'],
                    'bid_price_920': data['bid_price_920'],
                    'bid_price_925': data['bid_price_925'],
                    'bid_volume_915': data['bid_volume_915'],
                    'bid_volume_920': data['bid_volume_920'],
                    'bid_volume_925': data['bid_volume_925'],
                    'buy_order_size_925': data['buy_order_size_925'],
                    'sector_change': data['sector_change'],
                    'up_down_ratio': data['up_down_ratio'],
                    'stock_code': stock
                })
                
                # 计算涨跌幅作为标签
                df['return'] = (df['close'] - df['open']) / df['open']
                df['label'] = (df['return'] >= 0.095).astype(int)  # 当日涨幅>=9.5%标记为涨停
                
                # 生成特征
                features = feature_engineer.extract_features(df)
                if features is not None and not features.empty:
                    # 添加股票代码
                    features['stock_code'] = stock
                    features['return'] = df['return'].values
                    features['label'] = df['label'].values
                    
                    # 只保留数值特征用于预测
                    numeric_features = features.select_dtypes(include=[np.number])
                    if 'label' in numeric_features.columns:
                        numeric_features = numeric_features.drop('label', axis=1)
                    if 'return' in numeric_features.columns:
                        numeric_features = numeric_features.drop('return', axis=1)
                    if 'stock_code' in numeric_features.columns:
                        numeric_features = numeric_features.drop('stock_code', axis=1)
                    
                    # 进行预测
                    predictions = model_trainer.predict(numeric_features)
                    
                    if predictions is not None:
                        # 做出交易决策
                        decisions = strategy_decision.make_decisions(features, predictions)
                        
                        # 风险控制检查
                        if decisions:
                            approved_decisions = risk_controller.check(decisions)
                        else:
                            approved_decisions = []
                        
                        # 记录结果
                        stock_results[stock] = {
                            'predictions': predictions['predictions'].tolist(),
                            'probabilities': predictions['probabilities'].tolist(),
                            'decisions': len(decisions),
                            'approved_decisions': len(approved_decisions),
                            'true_labels': df['label'].tolist(),
                            'actual_returns': df['return'].tolist()
                        }
                    else:
                        stock_results[stock] = {
                            'error': '预测失败',
                            'predictions': [],
                            'probabilities': [],
                            'decisions': 0,
                            'approved_decisions': 0,
                            'true_labels': df['label'].tolist(),
                            'actual_returns': df['return'].tolist()
                        }
                else:
                    stock_results[stock] = {
                        'error': '特征提取失败',
                        'predictions': [],
                        'probabilities': [],
                        'decisions': 0,
                        'approved_decisions': 0,
                        'true_labels': df['label'].tolist(),
                        'actual_returns': df['return'].tolist()
                    }
            except Exception as e:
                logger.error(f"处理股票{stock}时出错：{e}")
                stock_results[stock] = {
                    'error': str(e),
                    'predictions': [],
                    'probabilities': [],
                    'decisions': 0,
                    'approved_decisions': 0,
                    'true_labels': [],
                    'actual_returns': []
                }
        
        test_results[market_type] = stock_results
    
    return test_results

def analyze_test_results(test_results):
    """分析测试结果"""
    logger.info("分析测试结果")
    
    analysis_results = {}
    
    for market_type, stock_results in test_results.items():
        logger.info(f"分析{market_type}行情测试结果")
        
        # 统计数据
        total_stocks = len(stock_results)
        total_decisions = 0
        total_approved = 0
        total_predictions = 0
        correct_predictions = 0
        total_positive_predictions = 0
        total_positive_labels = 0
        avg_prediction_prob = []
        avg_actual_return = []
        
        for stock, result in stock_results.items():
            if 'error' not in result:
                # 统计决策数量
                total_decisions += result['decisions']
                total_approved += result['approved_decisions']
                
                # 统计预测数量
                predictions = result['predictions']
                true_labels = result['true_labels']
                probabilities = result['probabilities']
                actual_returns = result['actual_returns']
                
                total_predictions += len(predictions)
                
                # 计算正确预测数量
                correct_predictions += sum(p == l for p, l in zip(predictions, true_labels))
                
                # 统计正样本预测
                total_positive_predictions += sum(p == 1 for p in predictions)
                
                # 统计实际正样本
                total_positive_labels += sum(true_labels)
                
                # 统计概率和实际收益
                if probabilities:
                    avg_prediction_prob.extend(probabilities)
                if actual_returns:
                    avg_actual_return.extend(actual_returns)
        
        # 计算指标
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
        else:
            accuracy = 0
        
        if avg_prediction_prob:
            avg_prob = np.mean(avg_prediction_prob)
        else:
            avg_prob = 0
        
        if avg_actual_return:
            avg_return = np.mean(avg_actual_return)
            volatility = np.std(avg_actual_return)
        else:
            avg_return = 0
            volatility = 0
        
        # 记录分析结果
        analysis_results[market_type] = {
            'total_stocks': total_stocks,
            'total_decisions': total_decisions,
            'total_approved': total_approved,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'total_positive_predictions': total_positive_predictions,
            'total_positive_labels': total_positive_labels,
            'avg_probability': avg_prob,
            'avg_return': avg_return,
            'volatility': volatility
        }
        
        logger.info(f"{market_type}行情结果：")
        logger.info(f"  股票数量：{total_stocks}")
        logger.info(f"  决策数量：{total_decisions}")
        logger.info(f"  通过风控决策：{total_approved}")
        logger.info(f"  预测准确率：{accuracy:.4f}")
        logger.info(f"  正样本预测：{total_positive_predictions}")
        logger.info(f"  实际正样本：{total_positive_labels}")
        logger.info(f"  平均预测概率：{avg_prob:.4f}")
        logger.info(f"  平均实际收益：{avg_return:.4f}")
        logger.info(f"  收益波动率：{volatility:.4f}")
    
    return analysis_results

def main():
    """主函数"""
    logger.info("=== 开始极端行情压力测试 ===")
    
    # 1. 生成极端行情数据
    extreme_data = generate_extreme_market_data()
    
    # 2. 在极端行情数据上测试策略
    test_results = test_strategy_on_extreme_data(extreme_data)
    
    # 3. 分析测试结果
    analysis_results = analyze_test_results(test_results)
    
    # 4. 保存测试结果
    results_df = pd.DataFrame(analysis_results).T
    results_file = "extreme_market_test_results.csv"
    results_df.to_csv(results_file)
    logger.info(f"极端行情测试结果已保存到{results_file}")
    
    logger.info("=== 极端行情压力测试完成 ===")
    return 0

if __name__ == "__main__":
    main()
