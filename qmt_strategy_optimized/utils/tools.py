#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用工具函数模块
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def format_date(date, format_str="%Y%m%d"):
    """
    格式化日期
    
    Args:
        date: 日期对象或字符串
        format_str: 目标格式字符串
        
    Returns:
        str: 格式化后的日期字符串
    """
    if isinstance(date, datetime):
        return date.strftime(format_str)
    elif isinstance(date, str):
        try:
            # 尝试解析常见日期格式
            for fmt in ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]:
                try:
                    return datetime.strptime(date, fmt).strftime(format_str)
                except ValueError:
                    continue
            raise ValueError(f"无法解析日期: {date}")
        except ValueError as e:
            raise ValueError(f"日期格式错误: {date}") from e
    else:
        raise TypeError(f"日期类型错误: {type(date)}")


def parse_date(date_str):
    """
    解析日期字符串为datetime对象
    
    Args:
        date_str: 日期字符串
        
    Returns:
        datetime: 解析后的日期对象
    """
    if isinstance(date_str, datetime):
        return date_str
    
    for fmt in ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析日期字符串: {date_str}")


def get_trading_dates(start_date, end_date):
    """
    获取指定日期范围内的交易日列表
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        list: 交易日列表
    """
    # 这里简化处理，实际应从数据源获取交易日历
    start = parse_date(start_date)
    end = parse_date(end_date)
    dates = pd.date_range(start=start, end=end, freq='B')  # B表示工作日
    return [date.strftime("%Y%m%d") for date in dates]


def calculate_returns(data, price_col="close"):
    """
    计算收益率
    
    Args:
        data: 包含价格数据的DataFrame
        price_col: 价格列名
        
    Returns:
        pd.Series: 收益率序列
    """
    return data[price_col].pct_change()


def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率，默认为3%
        
    Returns:
        float: 夏普比率
    """
    excess_returns = returns - risk_free_rate / 252  # 假设252个交易日
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(returns):
    """
    计算最大回撤
    
    Args:
        returns: 收益率序列
        
    Returns:
        float: 最大回撤
    """
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def calculate_sortino_ratio(returns, risk_free_rate=0.03):
    """
    计算索提诺比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率，默认为3%
        
    Returns:
        float: 索提诺比率
    """
    excess_returns = returns - risk_free_rate / 252
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return np.inf
    downside_volatility = negative_returns.std()
    return np.sqrt(252) * excess_returns.mean() / downside_volatility


def calculate_calmar_ratio(returns, risk_free_rate=0.03):
    """
    计算卡玛比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率，默认为3%
        
    Returns:
        float: 卡玛比率
    """
    annual_return = (1 + returns.mean()) ** 252 - 1 - risk_free_rate
    max_drawdown = calculate_max_drawdown(returns)
    if max_drawdown >= 0:
        return np.inf
    return annual_return / abs(max_drawdown)


def calculate_information_ratio(returns, benchmark_returns):
    """
    计算信息比率
    
    Args:
        returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        
    Returns:
        float: 信息比率
    """
    active_returns = returns - benchmark_returns
    if active_returns.std() == 0:
        return np.inf
    return active_returns.mean() / active_returns.std() * np.sqrt(252)


def calculate_beta(returns, benchmark_returns):
    """
    计算贝塔系数
    
    Args:
        returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        
    Returns:
        float: 贝塔系数
    """
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    if benchmark_variance == 0:
        return 0
    return covariance / benchmark_variance


def calculate_alpha(returns, benchmark_returns, risk_free_rate=0.03):
    """
    计算阿尔法系数
    
    Args:
        returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        risk_free_rate: 无风险利率，默认为3%
        
    Returns:
        float: 阿尔法系数
    """
    beta = calculate_beta(returns, benchmark_returns)
    annual_return = (1 + returns.mean()) ** 252 - 1
    annual_benchmark_return = (1 + benchmark_returns.mean()) ** 252 - 1
    return annual_return - (risk_free_rate + beta * (annual_benchmark_return - risk_free_rate))


def calculate_win_rate(trade_records):
    """
    计算胜率
    
    Args:
        trade_records: 交易记录列表
        
    Returns:
        float: 胜率
    """
    if not trade_records:
        return 0
    profit_trades = [trade for trade in trade_records if trade['pnl'] > 0]
    return len(profit_trades) / len(trade_records)


def calculate_profit_loss_ratio(trade_records):
    """
    计算盈亏比
    
    Args:
        trade_records: 交易记录列表
        
    Returns:
        float: 盈亏比
    """
    if not trade_records:
        return 0
    
    profit_trades = [trade for trade in trade_records if trade['pnl'] > 0]
    loss_trades = [trade for trade in trade_records if trade['pnl'] < 0]
    
    if not loss_trades:
        return np.inf
    
    avg_profit = sum(trade['pnl'] for trade in profit_trades) / len(profit_trades) if profit_trades else 0
    avg_loss = abs(sum(trade['pnl'] for trade in loss_trades) / len(loss_trades))
    
    return avg_profit / avg_loss


def calculate_max_consecutive_wins(trade_records):
    """
    计算最大连续盈利次数
    
    Args:
        trade_records: 交易记录列表
        
    Returns:
        int: 最大连续盈利次数
    """
    if not trade_records:
        return 0
    
    max_consecutive = 0
    current_consecutive = 0
    
    for trade in trade_records:
        if trade['pnl'] > 0:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive


def calculate_max_consecutive_losses(trade_records):
    """
    计算最大连续亏损次数
    
    Args:
        trade_records: 交易记录列表
        
    Returns:
        int: 最大连续亏损次数
    """
    if not trade_records:
        return 0
    
    max_consecutive = 0
    current_consecutive = 0
    
    for trade in trade_records:
        if trade['pnl'] < 0:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    return max_consecutive


def calculate_avg_holding_days(trade_records):
    """
    计算平均持仓天数
    
    Args:
        trade_records: 交易记录列表
        
    Returns:
        float: 平均持仓天数
    """
    if not trade_records:
        return 0
    
    from datetime import datetime
    
    total_days = 0
    valid_trades = 0
    
    for trade in trade_records:
        if 'buy_date' in trade and 'sell_date' in trade:
            buy_date = datetime.strptime(trade['buy_date'], '%Y-%m-%d') if isinstance(trade['buy_date'], str) else trade['buy_date']
            sell_date = datetime.strptime(trade['sell_date'], '%Y-%m-%d') if isinstance(trade['sell_date'], str) else trade['sell_date']
            days = (sell_date - buy_date).days
            total_days += days
            valid_trades += 1
    
    return total_days / valid_trades if valid_trades > 0 else 0


def normalize_data(data, columns=None):
    """
    归一化数据
    
    Args:
        data: 输入数据
        columns: 需要归一化的列，默认为所有数值列
        
    Returns:
        pd.DataFrame: 归一化后的数据
    """
    data_copy = data.copy()
    if columns is None:
        columns = data_copy.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        min_val = data_copy[col].min()
        max_val = data_copy[col].max()
        if max_val > min_val:
            data_copy[col] = (data_copy[col] - min_val) / (max_val - min_val)
    
    return data_copy


def standardize_data(data, columns=None):
    """
    标准化数据
    
    Args:
        data: 输入数据
        columns: 需要标准化的列，默认为所有数值列
        
    Returns:
        pd.DataFrame: 标准化后的数据
    """
    data_copy = data.copy()
    if columns is None:
        columns = data_copy.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        mean_val = data_copy[col].mean()
        std_val = data_copy[col].std()
        if std_val > 0:
            data_copy[col] = (data_copy[col] - mean_val) / std_val
    
    return data_copy
