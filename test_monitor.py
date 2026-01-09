#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控面板模块测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from qmt_strategy.strategy.monitor import MonitorPanel
from qmt_strategy.utils.logger import setup_logger

logger = setup_logger()

def test_monitor_panel():
    """测试监控面板模块"""
    logger.info("=== 监控面板模块测试开始 ===")
    
    # 初始化监控面板
    monitor = MonitorPanel()
    
    # 测试1: 初始状态
    logger.info("\n测试1: 初始状态")
    status = monitor.strategy_status
    assert status['is_running'] == False, f"预期初始状态为未运行，但实际为{status['is_running']}"
    assert len(monitor.holdings) == 0, f"预期初始持仓为空，但实际有{len(monitor.holdings)}个持仓"
    assert len(monitor.equity_curve) == 0, f"预期初始资金曲线为空，但实际有{len(monitor.equity_curve)}个数据点"
    logger.info("✓ 初始状态测试通过")
    
    # 测试2: 更新策略状态
    logger.info("\n测试2: 更新策略状态")
    new_status = {
        'is_running': True,
        'total_trades': 100,
        'today_trades': 5,
        'total_pnl': 5000.0,
        'today_pnl': 200.0
    }
    monitor.update_strategy_status(new_status)
    updated_status = monitor.strategy_status
    assert updated_status['is_running'] == True, f"预期状态为运行中，但实际为{updated_status['is_running']}"
    assert updated_status['total_trades'] == 100, f"预期总交易数为100，但实际为{updated_status['total_trades']}"
    assert updated_status['today_trades'] == 5, f"预期今日交易数为5，但实际为{updated_status['today_trades']}"
    assert updated_status['total_pnl'] == 5000.0, f"预期总盈亏为5000.0，但实际为{updated_status['total_pnl']}"
    assert updated_status['today_pnl'] == 200.0, f"预期今日盈亏为200.0，但实际为{updated_status['today_pnl']}"
    logger.info("✓ 更新策略状态测试通过")
    
    # 测试3: 更新持仓信息
    logger.info("\n测试3: 更新持仓信息")
    new_holdings = {
        'stock_000': {
            'quantity': 100,
            'buy_price': 10.5,
            'highest_price': 11.0
        },
        'stock_001': {
            'quantity': 200,
            'buy_price': 15.3,
            'highest_price': 16.0
        }
    }
    monitor.update_holdings(new_holdings)
    assert len(monitor.holdings) == 2, f"预期持仓数为2，但实际为{len(monitor.holdings)}"
    assert 'stock_000' in monitor.holdings, "持仓中缺少stock_000"
    assert 'stock_001' in monitor.holdings, "持仓中缺少stock_001"
    logger.info("✓ 更新持仓信息测试通过")
    
    # 测试4: 更新资金曲线
    logger.info("\n测试4: 更新资金曲线")
    equity_curve = [1000000, 1005000, 1010000, 1008000, 1015000]
    monitor.update_equity_curve(equity_curve)
    assert len(monitor.equity_curve) == 5, f"预期资金曲线长度为5，但实际为{len(monitor.equity_curve)}"
    assert monitor.equity_curve == equity_curve, f"预期资金曲线与输入一致，但实际不一致"
    logger.info("✓ 更新资金曲线测试通过")
    
    # 测试5: 更新每日收益率
    logger.info("\n测试5: 更新每日收益率")
    daily_returns = [0.005, 0.01, -0.002, 0.007]
    monitor.update_daily_returns(daily_returns)
    assert len(monitor.daily_returns) == 4, f"预期每日收益率长度为4，但实际为{len(monitor.daily_returns)}"
    assert monitor.daily_returns == daily_returns, f"预期每日收益率与输入一致，但实际不一致"
    logger.info("✓ 更新每日收益率测试通过")
    
    # 测试6: 更新最近交易记录
    logger.info("\n测试6: 更新最近交易记录")
    recent_trades = [
        {
            'stock': 'stock_000',
            'date': pd.Timestamp('2025-12-01'),
            'action': 'buy',
            'price': 10.5,
            'quantity': 100,
            'pnl': 50.0
        },
        {
            'stock': 'stock_001',
            'date': pd.Timestamp('2025-12-02'),
            'action': 'buy',
            'price': 15.3,
            'quantity': 200,
            'pnl': 140.0
        }
    ]
    monitor.update_recent_trades(recent_trades)
    assert len(monitor.recent_trades) == 2, f"预期最近交易记录数为2，但实际为{len(monitor.recent_trades)}"
    logger.info("✓ 更新最近交易记录测试通过")
    
    # 测试7: 获取监控数据
    logger.info("\n测试7: 获取监控数据")
    monitor_data = monitor.get_monitor_data()
    assert isinstance(monitor_data, dict), f"预期监控数据为字典类型，但实际为{type(monitor_data)}"
    assert 'strategy_status' in monitor_data, "监控数据中缺少strategy_status"
    assert 'holdings' in monitor_data, "监控数据中缺少holdings"
    assert 'equity_curve' in monitor_data, "监控数据中缺少equity_curve"
    assert 'daily_returns' in monitor_data, "监控数据中缺少daily_returns"
    assert 'recent_trades' in monitor_data, "监控数据中缺少recent_trades"
    logger.info("✓ 获取监控数据测试通过")
    
    # 测试8: 生成绩效报告
    logger.info("\n测试8: 生成绩效报告")
    report = monitor.generate_performance_report()
    assert isinstance(report, dict), f"预期绩效报告为字典类型，但实际为{type(report)}"
    assert 'performance_metrics' in report, "绩效报告中缺少performance_metrics"
    assert 'equity_curve' in report, "绩效报告中缺少equity_curve"
    assert 'recent_trades' in report, "绩效报告中缺少recent_trades"
    assert 'generated_time' in report, "绩效报告中缺少generated_time"
    logger.info(f"✓ 生成绩效报告测试通过，报告包含{len(report)}个字段")
    
    logger.info("\n=== 监控面板模块测试完成 ===")
    logger.info("所有测试用例执行完毕")

if __name__ == "__main__":
    try:
        test_monitor_panel()
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}", exc_info=True)
        sys.exit(1)
