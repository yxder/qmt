#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
监控面板模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.logger import setup_logger
import config
import threading
import time

logger = setup_logger()

class MonitorPanel:
    """监控面板类，用于实时展示策略运行状态和绩效"""
    
    def __init__(self):
        """初始化监控面板"""
        logger.info("初始化监控面板")
        
        # 初始化监控数据
        self.strategy_status = {
            'is_running': False,
            'current_time': pd.Timestamp.now(),
            'total_trades': 0,
            'today_trades': 0,
            'total_pnl': 0.0,
            'today_pnl': 0.0
        }
        
        self.holdings = {}  # 当前持仓
        self.equity_curve = []  # 资金曲线
        self.daily_returns = []  # 每日收益率
        self.recent_trades = []  # 最近交易记录
        
        # 初始化监控线程
        self.monitor_thread = None
        self.is_monitoring = False
    
    def start_monitoring(self):
        """开始监控"""
        logger.info("开始监控策略运行状态")
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        logger.info("停止监控策略运行状态")
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            # 更新监控数据
            self._update_monitor_data()
            
            # 绘制监控图表
            self._draw_monitor_charts()
            
            # 等待下一次更新
            time.sleep(config.MONITOR_REFRESH_INTERVAL)
    
    def _update_monitor_data(self):
        """更新监控数据"""
        # 这里需要根据实际情况从各个模块获取最新数据
        # 例如，从策略决策模块获取持仓信息，从回测模块获取资金曲线等
        pass
    
    def _draw_monitor_charts(self):
        """绘制监控图表"""
        # 这里使用matplotlib绘制监控图表
        # 实际情况中，可能需要使用更高级的可视化库或Web框架
        
        try:
            plt.figure(figsize=(15, 10))
            
            # 绘制资金曲线
            plt.subplot(2, 2, 1)
            if self.equity_curve:
                plt.plot(self.equity_curve)
                plt.title('资金曲线')
                plt.xlabel('时间')
                plt.ylabel('资金金额')
            
            # 绘制每日收益率
            plt.subplot(2, 2, 2)
            if self.daily_returns:
                plt.bar(range(len(self.daily_returns)), self.daily_returns)
                plt.title('每日收益率')
                plt.xlabel('日期')
                plt.ylabel('收益率')
            
            # 绘制持仓分布
            plt.subplot(2, 2, 3)
            if self.holdings:
                stock_names = list(self.holdings.keys())
                quantities = [h['quantity'] for h in self.holdings.values()]
                plt.pie(quantities, labels=stock_names, autopct='%1.1f%%')
                plt.title('持仓分布')
            
            # 绘制最近交易记录
            plt.subplot(2, 2, 4)
            if self.recent_trades:
                # 这里简化处理，只显示最近10笔交易
                recent_trades = self.recent_trades[-10:]
                trade_dates = [t['date'].strftime('%Y-%m-%d') for t in recent_trades]
                pnls = [t['pnl'] for t in recent_trades]
                plt.bar(trade_dates, pnls)
                plt.title('最近10笔交易盈亏')
                plt.xlabel('日期')
                plt.ylabel('盈亏金额')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            
        except Exception as e:
            logger.error(f"绘制监控图表失败：{e}")
    
    def update_strategy_status(self, status):
        """更新策略状态"""
        self.strategy_status.update(status)
        self.strategy_status['current_time'] = pd.Timestamp.now()
    
    def update_holdings(self, holdings):
        """更新持仓信息"""
        self.holdings = holdings
    
    def update_equity_curve(self, equity_curve):
        """更新资金曲线"""
        self.equity_curve = equity_curve
    
    def update_daily_returns(self, daily_returns):
        """更新每日收益率"""
        self.daily_returns = daily_returns
    
    def update_recent_trades(self, trades):
        """更新最近交易记录"""
        self.recent_trades = trades
    
    def get_monitor_data(self):
        """获取监控数据"""
        return {
            'strategy_status': self.strategy_status,
            'holdings': self.holdings,
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns,
            'recent_trades': self.recent_trades
        }
    
    def generate_performance_report(self):
        """生成绩效报告"""
        logger.info("生成策略绩效报告")
        
        # 计算绩效指标
        performance_metrics = self._calculate_performance_metrics()
        
        # 生成报告
        report = {
            'performance_metrics': performance_metrics,
            'equity_curve': self.equity_curve,
            'recent_trades': self.recent_trades,
            'generated_time': pd.Timestamp.now()
        }
        
        # 保存报告到本地
        self._save_performance_report(report)
        
        return report
    
    def _calculate_performance_metrics(self):
        """计算绩效指标"""
        from utils.tools import calculate_sharpe_ratio, calculate_max_drawdown
        
        metrics = {
            'total_pnl': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        
        # 计算总盈亏
        if self.equity_curve and len(self.equity_curve) > 1:
            initial_capital = self.equity_curve[0]
            final_capital = self.equity_curve[-1]
            total_return = (final_capital - initial_capital) / initial_capital
            metrics['total_pnl'] = final_capital - initial_capital
            
            # 计算年化收益率
            days = len(self.daily_returns)
            if days > 0:
                metrics['annual_return'] = (1 + total_return) ** (252 / days) - 1
        
        # 计算夏普比率
        if self.daily_returns:
            returns_series = pd.Series(self.daily_returns)
            metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns_series)
            metrics['max_drawdown'] = calculate_max_drawdown(returns_series)
        
        # 计算胜率
        if self.recent_trades:
            win_trades = [t for t in self.recent_trades if t['pnl'] > 0]
            metrics['win_rate'] = len(win_trades) / len(self.recent_trades)
        
        return metrics
    
    def _save_performance_report(self, report):
        """保存绩效报告"""
        import os
        import json
        
        # 创建保存目录
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        
        # 保存绩效报告
        report_file = os.path.join(report_dir, f"performance_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4, ensure_ascii=False, default=str)
        
        logger.info(f"绩效报告保存成功：{report_file}")
