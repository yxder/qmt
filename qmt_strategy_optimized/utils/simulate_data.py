#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模拟数据生成模块
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_stock_list(n=100):
    """
    生成模拟股票列表
    
    Args:
        n: 股票数量，默认100
        
    Returns:
        list: 股票代码列表
    """
    stock_list = []
    # 生成50只深圳A股
    for i in range(n//2):
        stock_list.append(f"{100000 + i:06d}.SZ")
    # 生成50只上海A股
    for i in range(n//2):
        stock_list.append(f"{600000 + i:06d}.SH")
    
    return stock_list


def generate_historical_data(stock_list, start_date, end_date):
    """
    生成模拟历史数据
    
    Args:
        stock_list: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        dict: 模拟历史数据，格式与xtdata.get_market_data返回格式一致
    """
    # 解析日期
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    # 生成日期范围
    dates = pd.date_range(start=start, end=end, freq='B')  # B表示工作日
    
    # 生成模拟数据
    historical_data = {}
    
    for stock in stock_list:
        # 生成基础价格
        base_price = np.random.uniform(5, 100)
        
        # 生成每日收益率
        returns = np.random.normal(0, 0.02, len(dates))
        returns[0] = 0  # 第一个交易日收益率为0
        
        # 生成收盘价
        close_prices = base_price * np.cumprod(1 + returns)
        
        # 生成开盘价、最高价、最低价
        open_prices = close_prices * np.random.normal(1, 0.01, len(dates))
        high_prices = np.maximum(open_prices, close_prices) * np.random.uniform(1, 1.03, len(dates))
        low_prices = np.minimum(open_prices, close_prices) * np.random.uniform(0.97, 1, len(dates))
        
        # 生成成交量和成交额
        volumes = np.random.randint(1000000, 10000000, len(dates))
        amounts = volumes * close_prices
        
        # 构造数据字典
        stock_data = {
            'time': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes,
            'amount': amounts
        }
        
        historical_data[stock] = stock_data
    
    return historical_data


def generate_bid_data(stock_list, date):
    """
    生成模拟集合竞价数据
    
    Args:
        stock_list: 股票代码列表
        date: 日期
        
    Returns:
        dict: 模拟集合竞价数据
    """
    bid_data = {}
    
    for stock in stock_list:
        # 生成模拟竞价数据
        bid_price_915 = np.random.uniform(10, 50)
        bid_price_920 = bid_price_915 * np.random.normal(1, 0.02)
        bid_price_925 = bid_price_920 * np.random.normal(1, 0.02)
        bid_volume_915 = np.random.randint(100000, 1000000)
        bid_volume_920 = bid_volume_915 * np.random.uniform(0.8, 1.2)
        bid_volume_925 = bid_volume_920 * np.random.uniform(0.8, 1.2)
        buy_order_size_925 = np.random.randint(100000, 1000000)
        
        # 构造数据字典
        stock_bid_data = {
            'bid_price_915': bid_price_915,
            'bid_price_920': bid_price_920,
            'bid_price_925': bid_price_925,
            'bid_volume_915': bid_volume_915,
            'bid_volume_920': bid_volume_920,
            'bid_volume_925': bid_volume_925,
            'buy_order_size_925': buy_order_size_925
        }
        
        bid_data[stock] = stock_bid_data
    
    return bid_data
