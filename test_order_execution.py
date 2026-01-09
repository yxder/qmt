#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
订单执行模块测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from qmt_strategy.strategy.order_execution import OrderExecutor
from qmt_strategy.utils.logger import setup_logger

logger = setup_logger()

def test_order_execution():
    """测试订单执行模块"""
    logger.info("=== 订单执行模块测试开始 ===")
    
    # 初始化订单执行器
    executor = OrderExecutor()
    
    # 测试1: 执行空决策列表
    logger.info("\n测试1: 执行空决策列表")
    results = executor.execute([])
    assert len(results) == 0, f"预期执行结果为空，但实际返回了{len(results)}个结果"
    logger.info("✓ 空决策列表测试通过")
    
    # 测试2: 执行买入决策
    logger.info("\n测试2: 执行买入决策")
    buy_decisions = [{
        'stock': '600000.SH',
        'action': 'buy',
        'price': 10.5,
        'probability': 0.9,
        'composite_score': 0.95,
        'time': pd.Timestamp.now()
    }]
    buy_results = executor.execute(buy_decisions)
    assert len(buy_results) == 1, f"预期执行1个买入决策，但实际返回了{len(buy_results)}个结果"
    assert buy_results[0]['action'] == 'buy', f"预期执行结果为买入，但实际为{buy_results[0]['action']}"
    assert buy_results[0]['status'] == 'filled', f"预期订单状态为filled，但实际为{buy_results[0]['status']}"
    logger.info(f"✓ 买入决策测试通过，执行结果: {buy_results[0]}")
    
    # 测试3: 执行卖出决策
    logger.info("\n测试3: 执行卖出决策")
    sell_decisions = [{
        'stock': '600000.SH',
        'action': 'sell',
        'price': 11.0,
        'reason': 'profit_target_reached',
        'return_rate': 0.0476,
        'time': pd.Timestamp.now()
    }]
    sell_results = executor.execute(sell_decisions)
    assert len(sell_results) == 1, f"预期执行1个卖出决策，但实际返回了{len(sell_results)}个结果"
    assert sell_results[0]['action'] == 'sell', f"预期执行结果为卖出，但实际为{sell_results[0]['action']}"
    assert sell_results[0]['status'] == 'filled', f"预期订单状态为filled，但实际为{sell_results[0]['status']}"
    logger.info(f"✓ 卖出决策测试通过，执行结果: {sell_results[0]}")
    
    # 测试4: 执行多个决策
    logger.info("\n测试4: 执行多个决策")
    multiple_decisions = [
        {
            'stock': '600000.SH',
            'action': 'buy',
            'price': 10.5,
            'probability': 0.9,
            'composite_score': 0.95,
            'time': pd.Timestamp.now()
        },
        {
            'stock': '000001.SZ',
            'action': 'buy',
            'price': 15.3,
            'probability': 0.85,
            'composite_score': 0.88,
            'time': pd.Timestamp.now()
        },
        {
            'stock': '600000.SH',
            'action': 'sell',
            'price': 11.0,
            'reason': 'profit_target_reached',
            'return_rate': 0.0476,
            'time': pd.Timestamp.now()
        }
    ]
    multiple_results = executor.execute(multiple_decisions)
    assert len(multiple_results) == 3, f"预期执行3个决策，但实际返回了{len(multiple_results)}个结果"
    logger.info(f"✓ 多个决策测试通过，共执行{len(multiple_results)}个订单")
    
    # 测试5: 撤销订单（回测模式下）
    logger.info("\n测试5: 撤销订单")
    result = executor.cancel_order(12345)
    assert result is False, f"预期撤销订单失败，但实际返回了{result}"
    logger.info("✓ 撤销订单测试通过")
    
    # 测试6: 获取持仓（回测模式下）
    logger.info("\n测试6: 获取持仓")
    position = executor.get_position()
    assert position is None, f"预期获取持仓失败，但实际返回了{position}"
    logger.info("✓ 获取持仓测试通过")
    
    # 测试7: 获取账户信息（回测模式下）
    logger.info("\n测试7: 获取账户信息")
    account = executor.get_account()
    assert account is None, f"预期获取账户信息失败，但实际返回了{account}"
    logger.info("✓ 获取账户信息测试通过")
    
    # 测试8: 停止订单执行器
    logger.info("\n测试8: 停止订单执行器")
    executor.stop()
    logger.info("✓ 停止订单执行器测试通过")
    
    logger.info("\n=== 订单执行模块测试完成 ===")
    logger.info("所有测试用例执行完毕")

if __name__ == "__main__":
    try:
        test_order_execution()
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}", exc_info=True)
        sys.exit(1)
