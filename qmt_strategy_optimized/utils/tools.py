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
