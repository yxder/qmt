#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略配置模块
集中管理所有策略相关参数，提高配置灵活性和可维护性
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any
import config as global_config

@dataclass
class StrategyConfig:
    """策略配置类，集中管理所有策略参数"""
    
    # 基本策略参数
    strategy_name: str = "HotSectorLimitUp"
    initial_capital: float = global_config.INITIAL_CAPITAL
    commission_rate: float = global_config.COMMISSION_RATE
    slippage_rate: float = global_config.SLIPPAGE_RATE
    max_daily_transactions: int = 20
    position_size: float = 0.01  # 每只股票买入资金占比
    
    # 回测参数
    backtest_start_date: str = global_config.BACKTEST_START_DATE
    backtest_end_date: str = global_config.BACKTEST_END_DATE
    
    # 止盈止损参数
    profit_target: float = global_config.PROFIT_TARGET
    stop_loss_ratio: float = global_config.STOP_LOSS_RATIO
    trailing_stop_ratio: float = global_config.TRAILING_STOP_RATIO
    
    # 热点板块参数
    top_sectors_count: int = 3  # 选取的热门板块数量
    sector_hotness_threshold: float = 0.05  # 板块热度阈值
    
    # 股票筛选参数
    min_market_cap: float = 5000000000  # 最小市值（50亿）
    max_market_cap: float = 50000000000  # 最大市值（500亿）
    min_volatility: float = 0.01  # 最小波动率
    max_volatility: float = 0.1  # 最大波动率
    
    # 模型参数
    model_path: str = os.path.join("models", "trained_models")
    use_ensemble: bool = True  # 是否使用模型融合
    model_names: List[str] = field(default_factory=lambda: ["xgboost", "random_forest", "logistic"])
    
    # 风险控制参数
    max_single_position: float = 0.1  # 单个股票最大持仓比例
    max_sector_position: float = 0.3  # 单个板块最大持仓比例
    max_drawdown_limit: float = 0.1  # 最大回撤限制
    
    # 日志和报告参数
    log_level: str = "INFO"
    report_dir: str = "reports"
    generate_monthly_summary: bool = True
    
    # 实盘参数
    live_trading: bool = False
    loop_interval: int = global_config.LOOP_INTERVAL
    error_retry_interval: int = global_config.ERROR_RETRY_INTERVAL
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StrategyConfig":
        """从字典创建配置实例"""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> "StrategyConfig":
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
