#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
系统功能测试脚本
"""

import sys
import os

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qmt_strategy.strategy.feature_engineer import FeatureEngineer
from qmt_strategy.strategy.model_train import ModelTrainer
from qmt_strategy.strategy.strategy_decision import StrategyDecision
from qmt_strategy.strategy.risk_control import RiskController
from qmt_strategy.strategy.backtest import Backtester
import pandas as pd
import numpy as np

def test_feature_engineer():
    """测试特征工程模块"""
    print("=== 测试特征工程模块 ===")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'bid_price_915': [10.0, 20.0, 30.0],
        'bid_price_920': [10.1, 20.1, 30.1],
        'bid_price_925': [10.2, 20.2, 30.2],
        'bid_volume_915': [1000, 2000, 3000],
        'bid_volume_920': [1100, 2100, 3100],
        'bid_volume_925': [1200, 2200, 3200],
        'prev_close': [9.9, 19.9, 29.9],
        'close': [10.3, 20.3, 30.3],
        'label': [1, 0, 1]
    }, index=['stock1', 'stock2', 'stock3'])
    
    # 测试特征提取
    fe = FeatureEngineer()
    features = fe.extract_features(test_data)
    
    if features is not None and not features.empty:
        print("特征工程测试通过")
        return True
    else:
        print("特征工程测试失败")
        return False

def test_model_trainer():
    """测试模型训练模块"""
    print("\n=== 测试模型训练模块 ===")
    
    # 创建测试数据
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # 转换为DataFrame
    feature_cols = [f'feature_{i}' for i in range(10)]
    features = pd.DataFrame(X, columns=feature_cols)
    features['label'] = y
    
    # 测试模型训练
    mt = ModelTrainer()
    model = mt.train(features)
    
    if model is not None:
        print("模型训练测试通过")
        return True
    else:
        print("模型训练测试失败")
        return False

def test_strategy_decision():
    """测试策略决策模块"""
    print("\n=== 测试策略决策模块 ===")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'bid_intensity': [0.08, 0.02, 0.05],
        'buy_order_size_925': [100000, 50000, 80000],
        'profit_effect': [0.3, -0.1, 0.2],
        'bid_price_925': [10.2, 20.2, 30.2]
    }, index=['stock1', 'stock2', 'stock3'])
    
    # 创建预测结果
    predictions = {
        'predictions': [1, 0, 1],
        'probabilities': [0.85, 0.3, 0.9]
    }
    
    # 测试策略决策
    sd = StrategyDecision()
    decisions = sd.make_decisions(test_data, predictions)
    
    if decisions is not None:
        print("策略决策测试通过")
        return True
    else:
        print("策略决策测试失败")
        return False

def test_risk_controller():
    """测试风险控制模块"""
    print("\n=== 测试风险控制模块 ===")
    
    # 创建测试决策
    test_decisions = [
        {
            'stock': 'stock1',
            'action': 'buy',
            'price': 10.2
        },
        {
            'stock': 'stock2',
            'action': 'sell',
            'price': 20.2
        }
    ]
    
    # 测试风险控制
    rc = RiskController(initial_capital=1000000)
    approved_decisions = rc.check(test_decisions)
    
    if approved_decisions is not None:
        print("风险控制测试通过")
        return True
    else:
        print("风险控制测试失败")
        return False

def test_backtester():
    """测试回测模块"""
    print("\n=== 测试回测模块 ===")
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2025-11-01', '2025-12-31', freq='D')
    stocks = ['stock1', 'stock2', 'stock3']
    
    # 创建多层索引数据
    index = pd.MultiIndex.from_product([dates, stocks], names=['date', 'stock'])
    data = {
        'bid_price_915': np.random.rand(len(index)) * 100,
        'bid_price_920': np.random.rand(len(index)) * 100,
        'bid_price_925': np.random.rand(len(index)) * 100,
        'bid_volume_915': np.random.randint(1000, 10000, len(index)),
        'bid_volume_920': np.random.randint(1000, 10000, len(index)),
        'bid_volume_925': np.random.randint(1000, 10000, len(index)),
        'prev_close': np.random.rand(len(index)) * 100,
        'close': np.random.rand(len(index)) * 100,
        'label': np.random.randint(0, 2, len(index))
    }
    
    features = pd.DataFrame(data, index=index)
    
    # 测试回测
    bt = Backtester()
    results = bt.run(features)
    
    if results is not None:
        print("回测模块测试通过")
        return True
    else:
        print("回测模块测试失败")
        return False

def main():
    """主测试函数"""
    print("开始系统功能测试...")
    
    tests = [
        test_feature_engineer,
        test_model_trainer,
        test_strategy_decision,
        test_risk_controller,
        test_backtester
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过测试: {passed}")
    print(f"失败测试: {failed}")
    print(f"通过率: {passed / len(tests) * 100:.2f}%")
    
    if failed == 0:
        print("✅ 所有测试通过!")
        return True
    else:
        print("❌ 部分测试失败!")
        return False

if __name__ == "__main__":
    main()
