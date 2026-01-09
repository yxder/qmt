#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略决策模块测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from qmt_strategy.strategy.strategy_decision import StrategyDecision
from qmt_strategy.utils.logger import setup_logger

logger = setup_logger()

def generate_test_data():
    """生成测试数据"""
    logger.info("生成测试数据")
    
    # 生成10只股票的测试数据
    stock_codes = [f"stock_{i:03d}" for i in range(10)]
    
    # 创建特征数据
    data = {
        'stock_code': stock_codes,
        'prediction': [1] * 5 + [0] * 5,  # 前5只股票预测为1（买入），后5只为0（不买入）
        'probability': [0.9, 0.85, 0.82, 0.79, 0.75] + [0.6, 0.55, 0.5, 0.45, 0.4],
        'bid_price_925': [10.5, 12.3, 8.7, 15.6, 20.2, 5.6, 7.8, 9.2, 11.4, 13.7],
        'bid_intensity': [0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3],
        'buy_order_size_925': [10000, 8000, 6000, 4000, 2000, 1500, 1200, 1000, 800, 600],
        'profit_effect': [0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4],
        'close': [10.6, 12.4, 8.8, 15.7, 20.3, 5.7, 7.9, 9.3, 11.5, 13.8],
        'last_price': [10.55, 12.35, 8.75, 15.65, 20.25, 5.65, 7.85, 9.25, 11.45, 13.75],
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    df.set_index('stock_code', inplace=True)
    
    logger.info(f"测试数据生成完成，形状: {df.shape}")
    logger.info(f"测试数据列: {list(df.columns)}")
    
    return df

def test_strategy_decision():
    """测试策略决策模块"""
    logger.info("开始测试策略决策功能")
    
    # 生成测试数据
    features = generate_test_data()
    
    # 初始化策略决策器
    decision_maker = StrategyDecision()
    
    # 测试动态买点决策
    logger.info("\n测试动态买点决策")
    predictions = {
        'predictions': features['prediction'].values,
        'probabilities': features['probability'].values
    }
    
    # 调用make_decisions方法测试买卖决策
    all_decisions = decision_maker.make_decisions(features, predictions)
    
    if all_decisions:
        logger.info(f"\n交易决策结果:")
        for decision in all_decisions:
            logger.info(f"  {decision}")
    else:
        logger.error("未生成任何交易决策")
    
    # 测试更新持仓功能
    logger.info("\n测试更新持仓功能")
    
    # 模拟交易结果
    trade_results = []
    for i, decision in enumerate(all_decisions):
        if decision['action'] == 'buy':
            trade_results.append({
                'stock': decision['stock'],
                'action': 'buy',
                'price': decision['price'],
                'quantity': 100,
                'time': decision['time']
            })
    
    # 更新持仓
    decision_maker.update_holdings(trade_results)
    
    # 获取当前持仓
    holdings = decision_maker.get_holdings()
    logger.info(f"当前持仓:")
    for stock, holding in holdings.items():
        logger.info(f"  {stock}: {holding}")
    
    # 再次调用make_decisions方法，测试卖点决策
    logger.info("\n测试自适应卖点决策")
    all_decisions_again = decision_maker.make_decisions(features, predictions)
    
    if all_decisions_again:
        logger.info(f"\n第二次交易决策结果:")
        for decision in all_decisions_again:
            logger.info(f"  {decision}")
    else:
        logger.error("第二次未生成任何交易决策")
    
    # 测试获取交易记录
    trade_records = decision_maker.get_trade_records()
    logger.info(f"\n交易记录数量: {len(trade_records)}")
    for record in trade_records:
        logger.info(f"  {record}")
    
    logger.info("\n策略决策功能测试完成")

def test_buy_decision_logic():
    """测试买入决策逻辑"""
    logger.info("\n\n测试买入决策逻辑")
    
    # 生成更详细的测试数据，覆盖不同情况
    test_cases = [
        # 高概率、高综合评分
        {
            'stock_code': 'stock_high_prob_high_score',
            'prediction': 1,
            'probability': 0.95,
            'bid_intensity': 0.9,
            'buy_order_size_925': 15000,
            'profit_effect': 0.9
        },
        # 高概率、低综合评分
        {
            'stock_code': 'stock_high_prob_low_score',
            'prediction': 1,
            'probability': 0.9,
            'bid_intensity': 0.3,
            'buy_order_size_925': 1000,
            'profit_effect': 0.3
        },
        # 低概率、高综合评分
        {
            'stock_code': 'stock_low_prob_high_score',
            'prediction': 0,
            'probability': 0.75,
            'bid_intensity': 0.9,
            'buy_order_size_925': 15000,
            'profit_effect': 0.9
        },
        # 低概率、低综合评分
        {
            'stock_code': 'stock_low_prob_low_score',
            'prediction': 0,
            'probability': 0.7,
            'bid_intensity': 0.3,
            'buy_order_size_925': 1000,
            'profit_effect': 0.3
        }
    ]
    
    df = pd.DataFrame(test_cases)
    df.set_index('stock_code', inplace=True)
    
    decision_maker = StrategyDecision()
    predictions = {
        'predictions': df['prediction'].values,
        'probabilities': df['probability'].values
    }
    
    decisions = decision_maker.make_decisions(df, predictions)
    
    logger.info(f"\n买入决策逻辑测试结果:")
    for decision in decisions:
        logger.info(f"  {decision}")
    
    # 验证只有高概率、高综合评分的股票被选中
    expected_buys = ['stock_high_prob_high_score']
    actual_buys = [d['stock'] for d in decisions if d['action'] == 'buy']
    
    logger.info(f"\n预期买入股票: {expected_buys}")
    logger.info(f"实际买入股票: {actual_buys}")
    
    if set(actual_buys) == set(expected_buys):
        logger.info("✓ 买入决策逻辑测试通过")
    else:
        logger.error("✗ 买入决策逻辑测试失败")

if __name__ == "__main__":
    logger.info("=== 策略决策模块测试开始 ===")
    
    try:
        # 执行测试
        test_strategy_decision()
        test_buy_decision_logic()
        
        logger.info("\n=== 策略决策模块测试完成 ===")
        logger.info("所有测试用例执行完毕")
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}", exc_info=True)
        sys.exit(1)
