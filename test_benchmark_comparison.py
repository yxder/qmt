#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基准策略对比验证脚本：将ML策略与传统技术分析策略进行对比
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

def load_historical_data():
    """加载历史数据"""
    logger.info("加载历史数据")
    
    data_file = "data/raw_data/history/history_20251101_20251231.pkl"
    if not os.path.exists(data_file):
        logger.error(f"数据文件不存在：{data_file}")
        return None
    
    import pickle
    with open(data_file, 'rb') as f:
        historical_data = pickle.load(f)
    
    logger.info(f"加载了{len(historical_data)}只股票的数据")
    return historical_data

def run_ml_strategy(historical_data):
    """运行ML策略"""
    logger.info("运行ML策略")
    
    # 初始化各模块
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    model_trainer.load_model()
    
    if model_trainer.model is None:
        logger.error("无法加载模型，ML策略测试失败")
        return None
    
    strategy_decision = StrategyDecision()
    risk_controller = RiskController()
    
    # 策略结果
    ml_results = {
        'trades': [],
        'returns': [],
        'cumulative_return': 0.0,
        'trading_days': 0,
        'win_trades': 0,
        'lose_trades': 0
    }
    
    # 处理每只股票
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
            
            # 计算涨跌幅
            df['return'] = (df['close'] - df['open']) / df['open']
            df['label'] = (df['return'] >= 0.095).astype(int)  # 当日涨幅>=9.5%标记为涨停
            
            # 生成特征
            features = feature_engineer.extract_features(df)
            if features is not None and not features.empty:
                # 添加股票代码和返回值
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
                    
                    # 记录交易结果
                    for decision in approved_decisions:
                        if decision['action'] == 'buy':
                            # 找到对应的日期和收益
                            idx = df[df['time'].dt.date == decision['date'].date()].index
                            if idx.any():
                                actual_return = df.loc[idx[0], 'return']
                                ml_results['trades'].append({
                                    'stock': stock,
                                    'date': decision['date'],
                                    'action': decision['action'],
                                    'price': decision['price'],
                                    'actual_return': actual_return
                                })
                                ml_results['returns'].append(actual_return)
                                
                                if actual_return > 0:
                                    ml_results['win_trades'] += 1
                                else:
                                    ml_results['lose_trades'] += 1
            
        except Exception as e:
            logger.error(f"处理股票{stock}时出错：{e}")
            continue
    
    # 计算累计收益
    if ml_results['returns']:
        ml_results['cumulative_return'] = np.sum(ml_results['returns']) / len(ml_results['returns']) * 100
    
    # 计算交易天数
    ml_results['trading_days'] = len(ml_results['returns'])
    
    logger.info(f"ML策略结果：")
    logger.info(f"  总交易次数：{len(ml_results['trades'])}")
    logger.info(f"  交易天数：{ml_results['trading_days']}")
    logger.info(f"  盈利交易：{ml_results['win_trades']}")
    logger.info(f"  亏损交易：{ml_results['lose_trades']}")
    logger.info(f"  平均日收益：{ml_results['cumulative_return']:.4f}%")
    
    return ml_results

def run_benchmark_strategy(historical_data):
    """运行基准策略（MA5/MA10金叉策略）"""
    logger.info("运行基准策略（MA5/MA10金叉策略）")
    
    # 基准策略结果
    benchmark_results = {
        'trades': [],
        'returns': [],
        'cumulative_return': 0.0,
        'trading_days': 0,
        'win_trades': 0,
        'lose_trades': 0
    }
    
    # 处理每只股票
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
            
            # 计算涨跌幅
            df['return'] = (df['close'] - df['open']) / df['open']
            
            # 计算MA5和MA10
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA10'] = df['close'].rolling(window=10).mean()
            
            # 生成买卖信号：MA5金叉MA10时买入
            df['signal'] = 0
            df.loc[df['MA5'] > df['MA10'], 'signal'] = 1  # 买入信号
            df.loc[df['MA5'] <= df['MA10'], 'signal'] = 0  # 卖出信号
            
            # 去除NA值
            df = df.dropna()
            
            # 模拟交易
            for i in range(1, len(df)):
                # 前一天的信号和今天的开盘价
                prev_signal = df.iloc[i-1]['signal']
                today_open = df.iloc[i]['open']
                today_close = df.iloc[i]['close']
                today_return = df.iloc[i]['return']
                
                # 如果前一天是买入信号，今天买入
                if prev_signal == 1:
                    # 记录交易
                    benchmark_results['trades'].append({
                        'stock': stock,
                        'date': df.iloc[i]['time'],
                        'action': 'buy',
                        'price': today_open,
                        'actual_return': today_return
                    })
                    benchmark_results['returns'].append(today_return)
                    
                    if today_return > 0:
                        benchmark_results['win_trades'] += 1
                    else:
                        benchmark_results['lose_trades'] += 1
            
        except Exception as e:
            logger.error(f"处理股票{stock}时出错：{e}")
            continue
    
    # 计算累计收益
    if benchmark_results['returns']:
        benchmark_results['cumulative_return'] = np.sum(benchmark_results['returns']) / len(benchmark_results['returns']) * 100
    
    # 计算交易天数
    benchmark_results['trading_days'] = len(benchmark_results['returns'])
    
    logger.info(f"基准策略结果：")
    logger.info(f"  总交易次数：{len(benchmark_results['trades'])}")
    logger.info(f"  交易天数：{benchmark_results['trading_days']}")
    logger.info(f"  盈利交易：{benchmark_results['win_trades']}")
    logger.info(f"  亏损交易：{benchmark_results['lose_trades']}")
    logger.info(f"  平均日收益：{benchmark_results['cumulative_return']:.4f}%")
    
    return benchmark_results

def compare_strategies(ml_results, benchmark_results):
    """对比两种策略的性能"""
    logger.info("=== 策略对比结果 ===")
    
    if ml_results and benchmark_results:
        # 计算胜率
        ml_win_rate = ml_results['win_trades'] / ml_results['trading_days'] if ml_results['trading_days'] > 0 else 0
        benchmark_win_rate = benchmark_results['win_trades'] / benchmark_results['trading_days'] if benchmark_results['trading_days'] > 0 else 0
        
        # 计算年化收益率
        ml_annual_return = ml_results['cumulative_return'] * 252 / (ml_results['trading_days'] if ml_results['trading_days'] > 0 else 1)
        benchmark_annual_return = benchmark_results['cumulative_return'] * 252 / (benchmark_results['trading_days'] if benchmark_results['trading_days'] > 0 else 1)
        
        # 计算夏普比率（假设无风险利率为2%）
        risk_free_rate = 0.02
        ml_volatility = np.std(ml_results['returns']) if ml_results['returns'] else 0
        benchmark_volatility = np.std(benchmark_results['returns']) if benchmark_results['returns'] else 0
        
        ml_sharpe = (ml_results['cumulative_return'] / 100 - risk_free_rate) / ml_volatility if ml_volatility > 0 else 0
        benchmark_sharpe = (benchmark_results['cumulative_return'] / 100 - risk_free_rate) / benchmark_volatility if benchmark_volatility > 0 else 0
        
        # 输出对比结果
        logger.info(f"\n{'指标':<20} {'ML策略':<20} {'基准策略':<20}")
        logger.info("-" * 60)
        logger.info(f"{'交易次数':<20} {len(ml_results['trades']):<20} {len(benchmark_results['trades']):<20}")
        logger.info(f"{'交易天数':<20} {ml_results['trading_days']:<20} {benchmark_results['trading_days']:<20}")
        logger.info(f"{'盈利交易':<20} {ml_results['win_trades']:<20} {benchmark_results['win_trades']:<20}")
        logger.info(f"{'亏损交易':<20} {ml_results['lose_trades']:<20} {benchmark_results['lose_trades']:<20}")
        logger.info(f"{'胜率':<20} {ml_win_rate:.4f} {benchmark_win_rate:.4f}")
        logger.info(f"{'平均日收益':<20} {ml_results['cumulative_return']:.4f}% {benchmark_results['cumulative_return']:.4f}%")
        logger.info(f"{'年化收益率':<20} {ml_annual_return:.4f}% {benchmark_annual_return:.4f}%")
        logger.info(f"{'夏普比率':<20} {ml_sharpe:.4f} {benchmark_sharpe:.4f}")
        logger.info(f"{'收益率标准差':<20} {ml_volatility:.4f} {benchmark_volatility:.4f}")
        
        # 保存对比结果
        comparison_df = pd.DataFrame({
            '指标': ['交易次数', '交易天数', '盈利交易', '亏损交易', '胜率', '平均日收益(%)', '年化收益率(%)', '夏普比率', '收益率标准差'],
            'ML策略': [
                len(ml_results['trades']),
                ml_results['trading_days'],
                ml_results['win_trades'],
                ml_results['lose_trades'],
                ml_win_rate,
                ml_results['cumulative_return'],
                ml_annual_return,
                ml_sharpe,
                ml_volatility
            ],
            '基准策略': [
                len(benchmark_results['trades']),
                benchmark_results['trading_days'],
                benchmark_results['win_trades'],
                benchmark_results['lose_trades'],
                benchmark_win_rate,
                benchmark_results['cumulative_return'],
                benchmark_annual_return,
                benchmark_sharpe,
                benchmark_volatility
            ]
        })
        
        comparison_df.to_csv('strategy_comparison_results.csv', index=False, encoding='utf-8-sig')
        logger.info("策略对比结果已保存到strategy_comparison_results.csv")
    else:
        logger.error("无法进行策略对比，缺少策略结果数据")

def main():
    """主函数"""
    logger.info("=== 开始基准策略对比验证 ===")
    
    # 1. 加载历史数据
    historical_data = load_historical_data()
    if historical_data is None:
        return 1
    
    # 2. 运行ML策略
    ml_results = run_ml_strategy(historical_data)
    
    # 3. 运行基准策略
    benchmark_results = run_benchmark_strategy(historical_data)
    
    # 4. 对比两种策略
    compare_strategies(ml_results, benchmark_results)
    
    logger.info("=== 基准策略对比验证完成 ===")
    return 0

if __name__ == "__main__":
    main()
