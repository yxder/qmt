#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测模块测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from qmt_strategy.strategy.backtest import Backtester
from qmt_strategy.utils.logger import setup_logger

logger = setup_logger()

def generate_test_backtest_data():
    """生成回测测试数据"""
    logger.info("生成回测测试数据")
    
    # 生成100天的模拟数据，5只股票
    n_days = 100
    n_stocks = 5
    
    dates = pd.date_range(start='2025-11-01', periods=n_days, freq='B')
    stock_codes = [f"stock_{i:03d}" for i in range(n_stocks)]
    
    # 创建特征数据
    data = []
    for date in dates:
        for stock in stock_codes:
            # 生成随机价格数据
            close = 10 + np.random.rand() * 20
            open = close * (0.995 + np.random.rand() * 0.01)
            high = max(open, close) * (1 + np.random.rand() * 0.02)
            low = min(open, close) * (1 - np.random.rand() * 0.02)
            
            # 生成随机预测数据
            prediction = int(np.random.rand() > 0.7)  # 30%的概率预测为涨停
            probability = 0.5 + np.random.rand() * 0.5
            
            # 添加数据行
            data.append({
                'date': date,
                'stock': stock,
                'close': close,
                'open': open,
                'high': high,
                'low': low,
                'volume': 1000000 + np.random.randint(0, 9000000),
                'prediction': prediction,
                'probability': probability,
                'bid_price_925': close * (0.998 + np.random.rand() * 0.004),
                'bid_intensity': 0.5 + np.random.rand() * 0.5,
                'buy_order_size_925': 5000 + np.random.randint(0, 15000),
                'profit_effect': 0.5 + np.random.rand() * 0.5
            })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 按照日期和股票索引
    df.set_index(['date', 'stock'], inplace=True)
    
    logger.info(f"回测测试数据生成完成，共{len(df)}行数据")
    logger.info(f"测试数据列: {list(df.columns)}")
    
    return df

def test_backtester():
    """测试回测器"""
    logger.info("=== 回测模块测试开始 ===")
    
    # 初始化回测器
    backtester = Backtester()
    
    # 测试1: 初始化状态
    logger.info("\n测试1: 初始化状态")
    assert backtester.backtest_results['total_return'] == 0.0, f"预期初始总收益率为0.0，但实际为{backtester.backtest_results['total_return']}"
    assert backtester.initial_capital == 1000000, f"预期初始资金为1000000，但实际为{backtester.initial_capital}"
    logger.info("✓ 初始化状态测试通过")
    
    # 生成测试数据
    features = generate_test_backtest_data()
    
    # 测试2: 运行回测
    logger.info("\n测试2: 运行回测")
    backtest_results = backtester.run(features)
    assert isinstance(backtest_results, dict), f"预期回测结果为字典类型，但实际为{type(backtest_results)}"
    logger.info(f"✓ 回测运行完成，回测结果: {backtest_results}")
    
    # 测试3: 检查回测结果
    logger.info("\n测试3: 检查回测结果")
    assert 'total_return' in backtest_results, "回测结果中缺少total_return字段"
    assert 'annual_return' in backtest_results, "回测结果中缺少annual_return字段"
    assert 'sharpe_ratio' in backtest_results, "回测结果中缺少sharpe_ratio字段"
    assert 'max_drawdown' in backtest_results, "回测结果中缺少max_drawdown字段"
    assert 'win_rate' in backtest_results, "回测结果中缺少win_rate字段"
    assert 'trade_count' in backtest_results, "回测结果中缺少trade_count字段"
    logger.info("✓ 回测结果字段检查通过")
    
    # 测试4: 检查回测统计数据
    logger.info("\n测试4: 检查回测统计数据")
    assert backtest_results['trade_count'] >= 0, f"预期交易次数大于等于0，但实际为{backtest_results['trade_count']}"
    assert backtest_results['profit_trades'] >= 0, f"预期盈利交易次数大于等于0，但实际为{backtest_results['profit_trades']}"
    assert backtest_results['loss_trades'] >= 0, f"预期亏损交易次数大于等于0，但实际为{backtest_results['loss_trades']}"
    assert 0 <= backtest_results['win_rate'] <= 1, f"预期胜率在0-1之间，但实际为{backtest_results['win_rate']}"
    logger.info("✓ 回测统计数据检查通过")
    
    # 测试5: 获取回测结果
    logger.info("\n测试5: 获取回测结果")
    results = backtester.get_backtest_results()
    assert isinstance(results, dict), f"预期回测结果为字典类型，但实际为{type(results)}"
    equity_curve = backtester.get_equity_curve()
    assert isinstance(equity_curve, list), f"预期资金曲线为列表类型，但实际为{type(equity_curve)}"
    assert len(equity_curve) > 0, "预期资金曲线不为空"
    trade_records = backtester.get_trade_records()
    assert isinstance(trade_records, list), f"预期交易记录为列表类型，但实际为{type(trade_records)}"
    logger.info(f"✓ 获取回测结果通过，交易记录数: {len(trade_records)}, 资金曲线长度: {len(equity_curve)}")
    
    # 测试6: 测试空数据回测
    logger.info("\n测试6: 测试空数据回测")
    empty_results = backtester.run(pd.DataFrame())
    assert isinstance(empty_results, dict), f"预期空数据回测结果为字典类型，但实际为{type(empty_results)}"
    logger.info("✓ 空数据回测测试通过")
    
    logger.info("\n=== 回测模块测试完成 ===")
    logger.info("所有测试用例执行完毕")

if __name__ == "__main__":
    try:
        test_backtester()
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}", exc_info=True)
        sys.exit(1)
