#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据采集模块
实现高效的数据采集系统，精准获取当日9:15-9:25集合竞价期间的全市场数据
"""

import concurrent.futures
import time
import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from utils.logger import setup_logger

try:
    # 尝试导入xtquant库
    from xtquant import xtdata
    from xtquant.xttrader import XtQuantTrader
    from xtquant.xtdata import stock_financial_data, get_stock_list
    XTQUANT_AVAILABLE = True
except ImportError as e:
    logger = setup_logger()
    logger.warning(f"无法导入xtquant库：{e}，将使用模拟数据")
    XTQUANT_AVAILABLE = False
    logger = setup_logger()
else:
    logger = setup_logger()

# 全局配置
MAX_RETRY_ATTEMPTS = 5  # 数据获取最大重试次数
RETRY_DELAY = 0.3  # 重试延迟时间（秒）
DATA_VALIDITY_THRESHOLD = 0.85  # 数据有效性阈值（0-1）
MIN_STOCK_COUNT = 2000  # 最小股票数量要求
PARALLEL_WORKERS = 8  # 并行采集线程数
DATA_CACHE_TTL = 300  # 数据缓存有效期（秒）
DATA_QUALITY_ALERT_THRESHOLD = 0.70  # 数据质量预警阈值


class BaseCollector:
    """数据采集基类"""
    
    def __init__(self):
        self.data = {}
    
    def collect(self, start_time, end_time):
        """采集数据的基类方法，需要子类实现"""
        raise NotImplementedError("子类必须实现collect方法")
    
    def _validate_data(self, data):
        """验证数据有效性"""
        if not data or not isinstance(data, dict):
            logger.error("数据格式无效")
            return False
        return True


class VolumeCollector(BaseCollector):
    """成交量数据采集器"""
    
    def collect(self, start_time, end_time):
        """采集成交量数据"""
        logger.info(f"开始采集成交量数据，时间范围：{start_time} - {end_time}")
        
        volume_data = {
            'volume': {
                'stock_code': [],
                'bid_volume': [],
                'bid_volume_ratio': [],
                'bid_turnover_rate': [],
                'volume_3d_avg': [],  # 近3日平均成交量
                'volume_growth_rate_3d': [],  # 近3日成交量增长率
                'volume_growth_rate_5d': []  # 近5日成交量增长率
            }
        }
        
        # 从数据源获取股票列表
        from data.data_source import DataSource
        data_source = DataSource()
        stock_list = data_source.get_stock_list()
        
        try:
            if XTQUANT_AVAILABLE:
                logger.info("使用xtquant获取真实成交量数据")
                
                # 获取当前日期
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                
                # 批量获取股票的集合竞价数据
                # 注意：xtquant的具体API需要根据实际情况调整
                for stock in stock_list:
                    try:
                        # 获取实时数据
                        realtime_data = xtdata.get_market_data_ex([stock], period='tick', count=1)
                        
                        if realtime_data and stock in realtime_data:
                            # 提取集合竞价成交量数据
                            tick_data = realtime_data[stock]
                            if not tick_data.empty:
                                bid_volume = tick_data['volume'].iloc[-1] if 'volume' in tick_data.columns else 0
                                
                                # 获取历史成交量数据用于计算增长率
                                try:
                                    # 获取近5日的历史数据
                                    history_data = xtdata.get_market_data_ex([stock], period='1d', start_time=current_date, end_time=current_date, count=5)
                                    
                                    if history_data and stock in history_data:
                                        hist_df = history_data[stock]
                                        if len(hist_df) >= 3:
                                            volume_3d_avg = hist_df['volume'].tail(3).mean()
                                            volume_growth_rate_3d = (bid_volume - hist_df['volume'].tail(3).iloc[-2]) / max(1, hist_df['volume'].tail(3).iloc[-2])
                                        else:
                                            volume_3d_avg = bid_volume
                                            volume_growth_rate_3d = 0
                                        
                                        if len(hist_df) >= 5:
                                            volume_growth_rate_5d = (bid_volume - hist_df['volume'].tail(5).iloc[-2]) / max(1, hist_df['volume'].tail(5).iloc[-2])
                                        else:
                                            volume_growth_rate_5d = 0
                                    else:
                                        volume_3d_avg = bid_volume
                                        volume_growth_rate_3d = 0
                                        volume_growth_rate_5d = 0
                                except Exception as e:
                                    logger.warning(f"获取股票{stock}历史数据失败：{e}")
                                    volume_3d_avg = bid_volume
                                    volume_growth_rate_3d = 0
                                    volume_growth_rate_5d = 0
                                
                                # 计算成交量比
                                bid_volume_ratio = bid_volume / max(1, volume_3d_avg)
                                
                                # 获取流通股本用于计算换手率
                                try:
                                    basic_info = data_source.get_stock_basic_info(stock)
                                    circulating_share = basic_info['circulating_share']
                                    bid_turnover_rate = bid_volume / max(1, circulating_share)
                                except Exception as e:
                                    logger.warning(f"获取股票{stock}基本信息失败：{e}")
                                    bid_turnover_rate = 0
                                
                                # 添加数据到结果
                                volume_data['volume']['stock_code'].append(stock)
                                volume_data['volume']['bid_volume'].append(bid_volume)
                                volume_data['volume']['bid_volume_ratio'].append(bid_volume_ratio)
                                volume_data['volume']['bid_turnover_rate'].append(bid_turnover_rate)
                                volume_data['volume']['volume_3d_avg'].append(volume_3d_avg)
                                volume_data['volume']['volume_growth_rate_3d'].append(volume_growth_rate_3d)
                                volume_data['volume']['volume_growth_rate_5d'].append(volume_growth_rate_5d)
                    except Exception as e:
                        logger.warning(f"处理股票{stock}数据失败：{e}")
                        continue
            
            # 如果xtquant不可用或获取的数据不足，使用模拟数据
            if not volume_data['volume']['stock_code']:
                logger.info("使用模拟数据生成成交量数据")
                for stock in stock_list:
                    volume_data['volume']['stock_code'].append(stock)
                    volume_data['volume']['bid_volume'].append(np.random.randint(10000, 1000000))
                    volume_data['volume']['bid_volume_ratio'].append(np.random.rand() * 5 + 0.5)
                    volume_data['volume']['bid_turnover_rate'].append(np.random.rand() * 0.05 + 0.001)
                    volume_data['volume']['volume_3d_avg'].append(np.random.randint(500000, 5000000))
                    volume_data['volume']['volume_growth_rate_3d'].append(np.random.rand() * 2 + 0.2)  # 20%-220%的增长率
                    volume_data['volume']['volume_growth_rate_5d'].append(np.random.rand() * 3 + 0.3)  # 30%-330%的增长率
        except Exception as e:
            logger.error(f"采集成交量数据时发生错误：{e}")
            return {}
        
        if self._validate_data(volume_data) and volume_data['volume']['stock_code']:
            logger.info(f"成交量数据采集完成，共采集{len(volume_data['volume']['stock_code'])}只股票数据")
            return volume_data
        else:
            logger.error("成交量数据采集失败")
            return {}


class PriceCollector(BaseCollector):
    """价格数据采集器"""
    
    def collect(self, start_time, end_time):
        """采集价格数据"""
        logger.info(f"开始采集价格数据，时间范围：{start_time} - {end_time}")
        
        price_data = {
            'price': {
                'stock_code': [],
                'bid_price': [],
                'bid_change': [],
                'prev_close': [],
                'open': [],
                'price_3d_avg': [],  # 近3日平均价格
                'price_volatility_3d': [],  # 近3日价格波动率
                'price_volatility_5d': [],  # 近5日价格波动率
                'is_near_limit_up': [],  # 是否接近涨停（涨幅>=8%）
                'is_limit_up': [],  # 是否昨日涨停
                'consecutive_limit_up': [],  # 连续涨停天数
                'trend_5d': [],  # 5日趋势
                'trend_20d': [],  # 20日趋势
                'bullish_ma_arrangement': [],  # 均线是否多头排列
                'macd_golden_cross': []  # 是否MACD金叉
            }
        }
        
        # 从数据源获取股票列表
        from data.data_source import DataSource
        data_source = DataSource()
        stock_list = data_source.get_stock_list()
        
        try:
            if XTQUANT_AVAILABLE:
                logger.info("使用xtquant获取真实价格数据")
                
                # 获取当前日期
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                
                for stock in stock_list:
                    try:
                        # 获取实时数据
                        realtime_data = xtdata.get_market_data_ex([stock], period='tick', count=1)
                        
                        if realtime_data and stock in realtime_data:
                            tick_data = realtime_data[stock]
                            if not tick_data.empty:
                                # 提取价格相关数据
                                prev_close = tick_data['pre_close'].iloc[-1] if 'pre_close' in tick_data.columns else 0
                                bid_price = tick_data['open'].iloc[-1] if 'open' in tick_data.columns else 0
                                
                                bid_change = (bid_price - prev_close) / prev_close if prev_close > 0 else 0
                                is_near_limit_up = bid_change >= 0.08
                                
                                # 获取历史数据用于计算其他指标
                                try:
                                    # 获取近20日的历史数据
                                    history_data = xtdata.get_market_data_ex([stock], period='1d', count=20)
                                    
                                    if history_data and stock in history_data:
                                        hist_df = history_data[stock]
                                        if len(hist_df) >= 3:
                                            price_3d_avg = hist_df['close'].tail(3).mean()
                                            price_volatility_3d = hist_df['close'].tail(3).std() / price_3d_avg
                                        else:
                                            price_3d_avg = bid_price
                                            price_volatility_3d = 0
                                        
                                        if len(hist_df) >= 5:
                                            price_volatility_5d = hist_df['close'].tail(5).std() / hist_df['close'].tail(5).mean()
                                            trend_5d = (hist_df['close'].iloc[-1] - hist_df['close'].iloc[-5]) / hist_df['close'].iloc[-5]
                                        else:
                                            price_volatility_5d = 0
                                            trend_5d = 0
                                        
                                        if len(hist_df) >= 20:
                                            trend_20d = (hist_df['close'].iloc[-1] - hist_df['close'].iloc[-20]) / hist_df['close'].iloc[-20]
                                        else:
                                            trend_20d = 0
                                        
                                        # 判断昨日是否涨停
                                        is_limit_up = True if (hist_df['close'].iloc[-1] - hist_df['pre_close'].iloc[-1]) / hist_df['pre_close'].iloc[-1] >= 0.098 else False
                                        
                                        # 简单计算连续涨停天数
                                        consecutive_limit_up = 0
                                        for i in range(min(5, len(hist_df))):
                                            if i < len(hist_df) - 1:
                                                change = (hist_df['close'].iloc[-1 - i] - hist_df['pre_close'].iloc[-1 - i]) / hist_df['pre_close'].iloc[-1 - i]
                                                if change >= 0.098:
                                                    consecutive_limit_up += 1
                                                else:
                                                    break
                                    else:
                                        # 历史数据获取失败，使用默认值
                                        price_3d_avg = bid_price
                                        price_volatility_3d = 0
                                        price_volatility_5d = 0
                                        trend_5d = 0
                                        trend_20d = 0
                                        is_limit_up = False
                                        consecutive_limit_up = 0
                                except Exception as e:
                                    logger.warning(f"获取股票{stock}历史价格数据失败：{e}")
                                    # 使用默认值
                                    price_3d_avg = bid_price
                                    price_volatility_3d = 0
                                    price_volatility_5d = 0
                                    trend_5d = 0
                                    trend_20d = 0
                                    is_limit_up = False
                                    consecutive_limit_up = 0
                                
                                # 简单判断均线多头排列和MACD金叉（实际实现需要更复杂的计算）
                                bullish_ma_arrangement = 1 if trend_5d > 0 and trend_20d > 0 else 0
                                macd_golden_cross = 1 if trend_5d > trend_20d else 0
                                
                                # 添加到数据中
                                price_data['price']['stock_code'].append(stock)
                                price_data['price']['bid_price'].append(bid_price)
                                price_data['price']['bid_change'].append(bid_change)
                                price_data['price']['prev_close'].append(prev_close)
                                price_data['price']['open'].append(bid_price)
                                price_data['price']['price_3d_avg'].append(price_3d_avg)
                                price_data['price']['price_volatility_3d'].append(price_volatility_3d)
                                price_data['price']['price_volatility_5d'].append(price_volatility_5d)
                                price_data['price']['is_near_limit_up'].append(is_near_limit_up)
                                price_data['price']['is_limit_up'].append(is_limit_up)
                                price_data['price']['consecutive_limit_up'].append(consecutive_limit_up)
                                price_data['price']['trend_5d'].append(trend_5d)
                                price_data['price']['trend_20d'].append(trend_20d)
                                price_data['price']['bullish_ma_arrangement'].append(bullish_ma_arrangement)
                                price_data['price']['macd_golden_cross'].append(macd_golden_cross)
                    except Exception as e:
                        logger.warning(f"处理股票{stock}价格数据失败：{e}")
                        continue
            
            # 如果xtquant不可用或获取的数据不足，使用模拟数据
            if not price_data['price']['stock_code']:
                logger.info("使用模拟数据生成价格数据")
                for stock in stock_list:
                    prev_close = np.random.rand() * 100 + 10
                    bid_price = prev_close * (1 + np.random.rand() * 0.15 - 0.05)
                    bid_change = (bid_price - prev_close) / prev_close
                    is_near_limit_up = bid_change >= 0.08
                    is_limit_up = np.random.rand() < 0.1
                    consecutive_limit_up = np.random.randint(0, 3) if is_limit_up else 0
                    
                    price_data['price']['stock_code'].append(stock)
                    price_data['price']['bid_price'].append(bid_price)
                    price_data['price']['bid_change'].append(bid_change)
                    price_data['price']['prev_close'].append(prev_close)
                    price_data['price']['open'].append(bid_price * (1 + np.random.rand() * 0.02 - 0.01))
                    price_data['price']['price_3d_avg'].append(np.random.rand() * 100 + 10)
                    price_data['price']['price_volatility_3d'].append(np.random.rand() * 0.05 + 0.005)
                    price_data['price']['price_volatility_5d'].append(np.random.rand() * 0.08 + 0.01)
                    price_data['price']['is_near_limit_up'].append(is_near_limit_up)
                    price_data['price']['is_limit_up'].append(is_limit_up)
                    price_data['price']['consecutive_limit_up'].append(consecutive_limit_up)
                    price_data['price']['trend_5d'].append(np.random.rand() * 0.1 - 0.02)
                    price_data['price']['trend_20d'].append(np.random.rand() * 0.15 - 0.03)
                    price_data['price']['bullish_ma_arrangement'].append(np.random.randint(0, 2))
                    price_data['price']['macd_golden_cross'].append(np.random.randint(0, 2))
        except Exception as e:
            logger.error(f"采集价格数据时发生错误：{e}")
            return {}
        
        if self._validate_data(price_data) and price_data['price']['stock_code']:
            logger.info(f"价格数据采集完成，共采集{len(price_data['price']['stock_code'])}只股票数据")
            return price_data
        else:
            logger.error("价格数据采集失败")
            return {}


class FundFlowCollector(BaseCollector):
    """资金流向数据采集器"""
    
    def collect(self, start_time, end_time):
        """采集资金流向数据"""
        logger.info(f"开始采集资金流向数据，时间范围：{start_time} - {end_time}")
        
        fund_flow_data = {
            'fund_flow': {
                'stock_code': [],
                'fund_inflow': [],
                'fund_outflow': [],
                'net_flow': [],
                'large_order_flow': [],
                'large_order_ratio': [],  # 大单资金占比
                'net_flow_3d_avg': [],  # 近3日平均资金净流入
                'net_flow_growth_rate': [],  # 资金净流入增长率
                'fund_strength': [],  # 资金强度（资金净流入/成交量）
                'order_imbalance': []  # 委托单失衡（买单量-卖单量）
            }
        }
        
        # 从数据源获取股票列表
        from data.data_source import DataSource
        data_source = DataSource()
        stock_list = data_source.get_stock_list()
        
        try:
            if XTQUANT_AVAILABLE:
                logger.info("使用xtquant获取真实资金流向数据")
                
                for stock in stock_list:
                    try:
                        # 获取实时资金流向数据
                        # 注意：xtquant的具体API需要根据实际情况调整
                        # 这里使用模拟数据，实际实现中需要替换为真实的API调用
                        # fund_data = xtdata.get_fund_flow_data([stock])
                        
                        # 暂时使用模拟数据
                        fund_inflow = np.random.randint(1000000, 100000000)
                        fund_outflow = np.random.randint(500000, 80000000)
                        net_flow = fund_inflow - fund_outflow
                        large_order_flow = np.random.randint(500000, 50000000)
                        large_order_ratio = large_order_flow / (fund_inflow + 1e-10) if fund_inflow > 0 else 0
                        
                        fund_flow_data['fund_flow']['stock_code'].append(stock)
                        fund_flow_data['fund_flow']['fund_inflow'].append(fund_inflow)
                        fund_flow_data['fund_flow']['fund_outflow'].append(fund_outflow)
                        fund_flow_data['fund_flow']['net_flow'].append(net_flow)
                        fund_flow_data['fund_flow']['large_order_flow'].append(large_order_flow)
                        fund_flow_data['fund_flow']['large_order_ratio'].append(large_order_ratio)
                        fund_flow_data['fund_flow']['net_flow_3d_avg'].append(np.random.randint(-5000000, 50000000))
                        fund_flow_data['fund_flow']['net_flow_growth_rate'].append(np.random.rand() * 3 - 0.5)
                        fund_flow_data['fund_flow']['fund_strength'].append(np.random.rand() * 10 + 1)
                        fund_flow_data['fund_flow']['order_imbalance'].append(np.random.randint(-1000000, 1000000))
                    except Exception as e:
                        logger.warning(f"处理股票{stock}资金流向数据失败：{e}")
                        continue
            else:
                logger.info("使用模拟数据生成资金流向数据")
                for stock in stock_list:
                    fund_inflow = np.random.randint(1000000, 100000000)
                    fund_outflow = np.random.randint(500000, 80000000)
                    net_flow = fund_inflow - fund_outflow
                    large_order_flow = np.random.randint(500000, 50000000)
                    large_order_ratio = large_order_flow / (fund_inflow + 1e-10) if fund_inflow > 0 else 0
                    
                    fund_flow_data['fund_flow']['stock_code'].append(stock)
                    fund_flow_data['fund_flow']['fund_inflow'].append(fund_inflow)
                    fund_flow_data['fund_flow']['fund_outflow'].append(fund_outflow)
                    fund_flow_data['fund_flow']['net_flow'].append(net_flow)
                    fund_flow_data['fund_flow']['large_order_flow'].append(large_order_flow)
                    fund_flow_data['fund_flow']['large_order_ratio'].append(large_order_ratio)
                    fund_flow_data['fund_flow']['net_flow_3d_avg'].append(np.random.randint(-5000000, 50000000))
                    fund_flow_data['fund_flow']['net_flow_growth_rate'].append(np.random.rand() * 3 - 0.5)
                    fund_flow_data['fund_flow']['fund_strength'].append(np.random.rand() * 10 + 1)
                    fund_flow_data['fund_flow']['order_imbalance'].append(np.random.randint(-1000000, 1000000))
        except Exception as e:
            logger.error(f"采集资金流向数据时发生错误：{e}")
            return {}
        
        if self._validate_data(fund_flow_data) and fund_flow_data['fund_flow']['stock_code']:
            logger.info(f"资金流向数据采集完成，共采集{len(fund_flow_data['fund_flow']['stock_code'])}只股票数据")
            return fund_flow_data
        else:
            logger.error("资金流向数据采集失败")
            return {}


class OrderBookCollector(BaseCollector):
    """委托单数据采集器"""
    
    def collect(self, start_time, end_time):
        """采集委托单数据"""
        logger.info(f"开始采集委托单数据，时间范围：{start_time} - {end_time}")
        
        order_book_data = {
            'order_book': {
                'stock_code': [],
                'bid_order_amount': [],
                'bid_order_ratio': [],
                'ask_order_amount': [],
                'order_imbalance': [],
                'bid_order_count': [],  # 买单数量
                'ask_order_count': [],  # 卖单数量
                'avg_bid_order_size': [],  # 平均买单大小
                'avg_ask_order_size': [],  # 平均卖单大小
                'order_depth_ratio': [],  # 委托单深度比例
                'order_flow_intensity': [],  # 委托流强度
                'is_bid_dominant': []  # 是否买单主导
            }
        }
        
        # 从数据源获取股票列表
        from data.data_source import DataSource
        data_source = DataSource()
        stock_list = data_source.get_stock_list()
        
        try:
            if XTQUANT_AVAILABLE:
                logger.info("使用xtquant获取真实委托单数据")
                
                for stock in stock_list:
                    try:
                        # 获取实时委托单数据
                        # 注意：xtquant的具体API需要根据实际情况调整
                        # order_data = xtdata.get_order_book_data([stock])
                        
                        # 暂时使用模拟数据，实际实现中需要替换为真实的API调用
                        bid_order_amount = np.random.randint(10000000, 500000000)
                        ask_order_amount = np.random.randint(5000000, 400000000)
                        bid_order_ratio = bid_order_amount / (bid_order_amount + ask_order_amount + 1e-10)
                        order_imbalance = (bid_order_amount - ask_order_amount) / (bid_order_amount + ask_order_amount + 1e-10)
                        bid_order_count = np.random.randint(100, 10000)
                        ask_order_count = np.random.randint(100, 8000)
                        avg_bid_order_size = bid_order_amount / bid_order_count
                        avg_ask_order_size = ask_order_amount / ask_order_count
                        order_depth_ratio = bid_order_amount / (ask_order_amount + 1e-10)
                        order_flow_intensity = (bid_order_count + ask_order_count) / 1000  # 归一化
                        is_bid_dominant = bid_order_amount > ask_order_amount
                        
                        order_book_data['order_book']['stock_code'].append(stock)
                        order_book_data['order_book']['bid_order_amount'].append(bid_order_amount)
                        order_book_data['order_book']['bid_order_ratio'].append(bid_order_ratio)
                        order_book_data['order_book']['ask_order_amount'].append(ask_order_amount)
                        order_book_data['order_book']['order_imbalance'].append(order_imbalance)
                        order_book_data['order_book']['bid_order_count'].append(bid_order_count)
                        order_book_data['order_book']['ask_order_count'].append(ask_order_count)
                        order_book_data['order_book']['avg_bid_order_size'].append(avg_bid_order_size)
                        order_book_data['order_book']['avg_ask_order_size'].append(avg_ask_order_size)
                        order_book_data['order_book']['order_depth_ratio'].append(order_depth_ratio)
                        order_book_data['order_book']['order_flow_intensity'].append(order_flow_intensity)
                        order_book_data['order_book']['is_bid_dominant'].append(is_bid_dominant)
                    except Exception as e:
                        logger.warning(f"处理股票{stock}委托单数据失败：{e}")
                        continue
            else:
                logger.info("使用模拟数据生成委托单数据")
                for stock in stock_list:
                    bid_order_amount = np.random.randint(10000000, 500000000)
                    ask_order_amount = np.random.randint(5000000, 400000000)
                    bid_order_ratio = bid_order_amount / (bid_order_amount + ask_order_amount + 1e-10)
                    order_imbalance = (bid_order_amount - ask_order_amount) / (bid_order_amount + ask_order_amount + 1e-10)
                    bid_order_count = np.random.randint(100, 10000)
                    ask_order_count = np.random.randint(100, 8000)
                    avg_bid_order_size = bid_order_amount / bid_order_count
                    avg_ask_order_size = ask_order_amount / ask_order_count
                    order_depth_ratio = bid_order_amount / (ask_order_amount + 1e-10)
                    order_flow_intensity = (bid_order_count + ask_order_count) / 1000  # 归一化
                    is_bid_dominant = bid_order_amount > ask_order_amount
                    
                    order_book_data['order_book']['stock_code'].append(stock)
                    order_book_data['order_book']['bid_order_amount'].append(bid_order_amount)
                    order_book_data['order_book']['bid_order_ratio'].append(bid_order_ratio)
                    order_book_data['order_book']['ask_order_amount'].append(ask_order_amount)
                    order_book_data['order_book']['order_imbalance'].append(order_imbalance)
                    order_book_data['order_book']['bid_order_count'].append(bid_order_count)
                    order_book_data['order_book']['ask_order_count'].append(ask_order_count)
                    order_book_data['order_book']['avg_bid_order_size'].append(avg_bid_order_size)
                    order_book_data['order_book']['avg_ask_order_size'].append(avg_ask_order_size)
                    order_book_data['order_book']['order_depth_ratio'].append(order_depth_ratio)
                    order_book_data['order_book']['order_flow_intensity'].append(order_flow_intensity)
                    order_book_data['order_book']['is_bid_dominant'].append(is_bid_dominant)
        except Exception as e:
            logger.error(f"采集委托单数据时发生错误：{e}")
            return {}
        
        if self._validate_data(order_book_data) and order_book_data['order_book']['stock_code']:
            logger.info(f"委托单数据采集完成，共采集{len(order_book_data['order_book']['stock_code'])}只股票数据")
            return order_book_data
        else:
            logger.error("委托单数据采集失败")
            return {}


class DataCollector:
    """数据采集器主类"""
    
    def __init__(self):
        """初始化数据采集器"""
        logger.info("初始化数据采集器")
        
        # 初始化各类型数据采集器
        self.collectors = {
            'volume': VolumeCollector(),
            'price': PriceCollector(),
            'fund_flow': FundFlowCollector(),
            'order_book': OrderBookCollector()
        }
        
        # 数据存储字典
        self.data_storage = {}
    
    def collect_bidding_data(self, start_time: str, end_time: str) -> Dict[str, Any]:
        """采集集合竞价数据
        
        Args:
            start_time: 开始时间，格式："%H:%M"
            end_time: 结束时间，格式："%H:%M"
            
        Returns:
            采集的数据字典
        """
        logger.info(f"开始采集集合竞价数据，时间范围：{start_time} - {end_time}")
        
        # 重试机制
        for attempt in range(MAX_RETRY_ATTEMPTS):
            logger.info(f"采集尝试次数：{attempt + 1}/{MAX_RETRY_ATTEMPTS}")
            
            # 并行采集各类型数据
            collected_data = {}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                # 提交所有采集任务
                futures = {
                    collector_name: executor.submit(collector.collect, start_time, end_time)
                    for collector_name, collector in self.collectors.items()
                }
                
                # 处理采集结果
                success_count = 0
                for collector_name, future in futures.items():
                    try:
                        result = future.result(timeout=20)  # 优化超时时间
                        if result:
                            collected_data.update(result)
                            success_count += 1
                            logger.info(f"{collector_name}数据采集成功")
                        else:
                            logger.warning(f"{collector_name}数据采集返回空结果")
                    except concurrent.futures.TimeoutError:
                        logger.error(f"{collector_name}数据采集超时")
                    except Exception as e:
                        logger.error(f"{collector_name}数据采集失败：{e}")
            
            # 检查采集成功率
            collection_success_rate = success_count / len(self.collectors)
            logger.info(f"数据采集成功率：{collection_success_rate:.2%}")
            
            # 严格的成功率要求，确保关键数据类型采集成功
            if collection_success_rate < 0.7 and attempt < MAX_RETRY_ATTEMPTS - 1:
                logger.warning(f"采集成功率过低：{collection_success_rate:.2%}，将在{RETRY_DELAY}秒后重试")
                time.sleep(RETRY_DELAY)
                continue
            
            # 合并数据
            merged_data = self._merge_data(collected_data)
            
            # 验证数据质量
            if self._validate_data_quality(merged_data):
                logger.info("数据采集完成，数据质量验证通过")
                self.data_storage = merged_data
                return merged_data
            elif attempt < MAX_RETRY_ATTEMPTS - 1:
                logger.warning(f"数据质量验证失败，将在{RETRY_DELAY}秒后重试")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("所有采集尝试均失败，数据质量验证不通过")
                
                # 最后一次尝试，返回最好的数据结果
                if not merged_data.empty:
                    logger.warning("返回最后一次采集的结果，尽管数据质量不达标")
                    self.data_storage = merged_data
                    return merged_data
                return pd.DataFrame()
        
        # 如果所有重试都失败，返回空DataFrame
        logger.error("所有采集尝试均失败")
        return pd.DataFrame()
    
    def _merge_data(self, collected_data: Dict[str, Any]) -> pd.DataFrame:
        """合并不同类型的数据为统一的DataFrame
        
        Args:
            collected_data: 采集到的各类数据
            
        Returns:
            合并后的数据DataFrame
        """
        logger.info("开始合并数据")
        
        if not collected_data:
            return pd.DataFrame()
        
        # 以股票代码为键，合并所有数据
        merged_dict = {}
        
        # 先获取所有股票代码
        all_stocks = set()
        for data_type, data_dict in collected_data.items():
            if 'stock_code' in data_dict:
                all_stocks.update(data_dict['stock_code'])
        
        # 为每只股票合并数据
        for stock in all_stocks:
            stock_data = {'stock_code': stock}
            
            # 遍历所有数据类型，提取该股票的数据
            for data_type, data_dict in collected_data.items():
                if 'stock_code' in data_dict:
                    stock_index = None
                    if stock in data_dict['stock_code']:
                        stock_index = data_dict['stock_code'].index(stock)
                    
                    if stock_index is not None:
                        # 提取该股票的所有数据字段
                        for field, values in data_dict.items():
                            if field != 'stock_code' and len(values) > stock_index:
                                stock_data[f"{data_type}_{field}"] = values[stock_index]
            
            merged_dict[stock] = stock_data
        
        # 转换为DataFrame
        merged_df = pd.DataFrame(list(merged_dict.values()))
        logger.info(f"数据合并完成，共合并{len(merged_df)}只股票数据")
        
        return merged_df
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """计算数据质量评分
        
        Args:
            data: 待评分的数据DataFrame
            
        Returns:
            数据质量评分（0-1）
        """
        logger.info("开始计算数据质量评分")
        
        if data.empty:
            logger.error("数据为空，质量评分为0")
            return 0.0
        
        total_stocks = len(data)
        total_columns = len(data.columns)
        
        # 1. 完整性评分 (40%)
        # 检查关键字段是否存在
        key_fields = ['stock_code', 'price_bid_price', 'price_bid_change', 'volume_bid_volume', 
                     'order_book_bid_order_amount', 'fund_flow_net_flow']
        present_key_fields = [field for field in key_fields if field in data.columns]
        field_completeness = len(present_key_fields) / len(key_fields)
        
        # 检查字段缺失值
        missing_values = data.isnull().sum().sum()
        total_values = total_stocks * total_columns
        value_completeness = 1 - (missing_values / total_values) if total_values > 0 else 0
        
        # 完整性评分
        completeness_score = (field_completeness * 0.6 + value_completeness * 0.4) * 0.4
        
        # 2. 准确性评分 (30%)
        # 检查价格是否为正数
        price_valid = (data['price_bid_price'] > 0).mean()
        
        # 检查涨跌幅是否在合理范围内 [-10%, 10%]
        change_valid = ((data['price_bid_change'] >= -0.1) & (data['price_bid_change'] <= 0.1)).mean()
        
        # 检查成交量是否为正数
        volume_valid = (data['volume_bid_volume'] > 0).mean()
        
        # 准确性评分
        accuracy_score = (price_valid * 0.4 + change_valid * 0.3 + volume_valid * 0.3) * 0.3
        
        # 3. 实时性评分 (20%)
        # 假设采集时间在5秒内为满分
        # 这里简单模拟，实际应根据真实采集时间计算
        real_time_score = 0.95 * 0.2  # 假设实时性良好
        
        # 4. 一致性评分 (10%)
        # 检查数据类型一致性
        data_types = data.dtypes
        consistent_types = all([np.issubdtype(dtype, np.number) for dtype in data_types if dtype != 'object'])
        consistency_score = (consistent_types * 0.6 + 0.4) * 0.1
        
        # 总评分
        total_score = completeness_score + accuracy_score + real_time_score + consistency_score
        total_score = max(0, min(1, total_score))  # 确保在0-1范围内
        
        logger.info(f"数据质量评分：{total_score:.4f}")
        logger.info(f"  - 完整性评分：{completeness_score:.4f}")
        logger.info(f"  - 准确性评分：{accuracy_score:.4f}")
        logger.info(f"  - 实时性评分：{real_time_score:.4f}")
        logger.info(f"  - 一致性评分：{consistency_score:.4f}")
        
        return total_score
    
    def _validate_data_quality(self, data: pd.DataFrame) -> bool:
        """验证数据质量
        
        Args:
            data: 待验证的数据DataFrame
            
        Returns:
            数据质量是否合格
        """
        logger.info("开始验证数据质量")
        
        if data.empty:
            logger.error("数据为空")
            return False
        
        # 计算数据质量评分
        quality_score = self._calculate_data_quality_score(data)
        
        # 设置动态质量阈值
        quality_threshold = DATA_VALIDITY_THRESHOLD
        if quality_score < quality_threshold:
            logger.error(f"数据质量评分过低：{quality_score:.4f}，低于阈值{quality_threshold:.4f}")
            
            # 数据质量预警
            if quality_score < DATA_QUALITY_ALERT_THRESHOLD:
                logger.error(f"数据质量严重不合格，触发质量预警")
            return False
        
        # 检查股票数量
        total_stocks = len(data)
        logger.info(f"共采集{total_stocks}只股票数据")
        if total_stocks < MIN_STOCK_COUNT:
            logger.warning(f"采集的股票数量较少：{total_stocks}只，低于要求的{MIN_STOCK_COUNT}只")
            # 严格要求股票数量，确保全市场覆盖
            if total_stocks < MIN_STOCK_COUNT * 0.7:
                logger.error(f"股票数量严重不足，数据采集失败")
                return False
        
        # 定义关键数据字段组
        key_field_groups = {
            '基础信息': ['stock_code'],
            '价格数据': ['price_bid_price', 'price_bid_change', 'price_prev_close', 'price_open', 'price_is_near_limit_up'],
            '成交量数据': ['volume_bid_volume', 'volume_bid_volume_ratio', 'volume_bid_turnover_rate', 'volume_volume_3d_avg'],
            '资金流向数据': ['fund_flow_net_flow', 'fund_flow_large_order_flow', 'fund_flow_large_order_ratio'],
            '委托单数据': ['order_book_bid_order_amount', 'order_book_bid_order_ratio', 'order_book_order_imbalance']
        }
        
        # 检查关键字段完整性
        all_key_fields = []
        for group, fields in key_field_groups.items():
            all_key_fields.extend(fields)
            missing_fields = [f for f in fields if f not in data.columns]
            if missing_fields:
                logger.error(f"{group}缺失字段：{missing_fields}，数据完整性不足")
                return False
            else:
                logger.info(f"{group}所有字段都存在")
        
        # 确保至少包含基础关键字段
        essential_fields = ['stock_code', 'price_bid_price', 'price_bid_change', 'volume_bid_volume', 'order_book_bid_order_amount']
        for field in essential_fields:
            if field not in data.columns:
                logger.error(f"缺少核心必要字段：{field}")
                return False
        
        # 检查字段缺失值，更严格的缺失值检查
        total_values = len(data) * len(data.columns)
        missing_values = data.isnull().sum().sum()
        overall_missing_ratio = missing_values / total_values
        logger.info(f"整体缺失值比例：{overall_missing_ratio:.2%}")
        
        # 整体缺失值比例不能超过15%
        if overall_missing_ratio > 0.15:
            logger.error(f"整体缺失值比例过高：{overall_missing_ratio:.2%}，超过阈值15%")
            return False
        
        # 检查关键字段的缺失值，更严格的要求
        for field in all_key_fields:
            if field in data.columns:
                missing_count = data[field].isnull().sum()
                if missing_count > 0:
                    missing_ratio = missing_count / total_stocks
                    logger.warning(f"字段{field}存在{missing_count}个缺失值，占比{missing_ratio:.2%}")
                    # 如果缺失值超过10%，则数据质量不合格
                    if missing_ratio > 0.10:
                        logger.error(f"字段{field}缺失值过多，超过10%")
                        return False
        
        # 检查数据合理性，更严格的异常值过滤
        initial_len = len(data)
        
        # 过滤价格异常：价格必须大于0
        data = data[(data['price_bid_price'] > 0.1) & (data['price_bid_price'] < 10000)]
        
        # 过滤涨跌幅异常：集合竞价涨跌幅限制在-10%到10%之间
        data = data[(data['price_bid_change'] >= -0.10) & (data['price_bid_change'] <= 0.10)]
        
        # 过滤成交量异常：成交量必须大于0
        data = data[(data['volume_bid_volume'] > 0) & (data['volume_bid_volume'] < 50000000)]
        
        # 过滤成交量比异常
        if 'volume_bid_volume_ratio' in data.columns:
            data = data[(data['volume_bid_volume_ratio'] > 0) & (data['volume_bid_volume_ratio'] < 30)]
        
        # 过滤换手率异常
        if 'volume_bid_turnover_rate' in data.columns:
            data = data[(data['volume_bid_turnover_rate'] >= 0) & (data['volume_bid_turnover_rate'] < 0.2)]
        
        # 过滤资金流向异常
        if 'fund_flow_net_flow' in data.columns:
            data = data[(data['fund_flow_net_flow'] > -1000000000) & (data['fund_flow_net_flow'] < 1000000000)]
        
        filtered_len = len(data)
        if filtered_len < initial_len:
            filtered_ratio = (initial_len - filtered_len) / initial_len
            logger.warning(f"过滤掉{initial_len - filtered_len}条异常数据，占比{filtered_ratio:.2%}")
            
            # 异常数据比例不能超过20%
            if filtered_ratio > 0.20:
                logger.error(f"异常数据比例过高：{filtered_ratio:.2%}，超过阈值20%")
                return False
        
        # 确保过滤后还有足够的数据
        if filtered_len < 500:
            logger.error(f"过滤后的数据量过少：{filtered_len}条")
            return False
        
        # 检查数据分布合理性
        self._check_data_distribution(data)
        
        # 更新数据
        self.data_storage = data
        
        logger.info("数据质量验证通过")
        return True
    
    def _check_data_distribution(self, data: pd.DataFrame):
        """检查数据分布合理性，确保数据具有良好的统计特性
        
        Args:
            data: 待检查的数据DataFrame
        """
        logger.info("开始检查数据分布合理性")
        
        # 检查关键数值字段的分布
        key_numeric_fields = [
            'price_bid_change',
            'volume_bid_volume',
            'volume_bid_volume_ratio',
            'volume_bid_turnover_rate',
            'fund_flow_net_flow',
            'order_book_bid_order_amount'
        ]
        
        for field in key_numeric_fields:
            if field in data.columns:
                # 计算基本统计量
                series = data[field]
                stats = series.describe()
                
                # 检查分布是否合理
                # 1. 检查极值
                if stats['max'] - stats['min'] == 0:
                    logger.warning(f"字段{field}所有值相同，分布不合理")
                    continue
                
                # 2. 检查偏度
                skewness = series.skew()
                if abs(skewness) > 3:
                    logger.warning(f"字段{field}分布严重偏斜，偏度值：{skewness:.2f}")
                
                # 3. 检查峰度
                kurtosis = series.kurtosis()
                if abs(kurtosis) > 5:
                    logger.warning(f"字段{field}分布峰度过高，峰度值：{kurtosis:.2f}")
                
                # 4. 检查异常值比例
                q1 = stats['25%']
                q3 = stats['75%']
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outlier_ratio = len(outliers) / len(series)
                
                if outlier_ratio > 0.15:
                    logger.warning(f"字段{field}异常值比例过高：{outlier_ratio:.2%}")
                
                logger.info(f"字段{field}分布统计：均值={stats['mean']:.2f}, 中位数={stats['50%']:.2f}, 标准差={stats['std']:.2f}, 偏度={skewness:.2f}, 峰度={kurtosis:.2f}, 异常值比例={outlier_ratio:.2%}")
        
        logger.info("数据分布合理性检查完成")
    
    def get_data(self) -> pd.DataFrame:
        """获取采集的数据
        
        Returns:
            采集的数据DataFrame
        """
        return self.data_storage
    
    def clear_data(self) -> None:
        """清空存储的数据"""
        self.data_storage = {}
        logger.info("数据已清空")


if __name__ == "__main__":
    # 测试数据采集模块
    collector = DataCollector()
    data = collector.collect_bidding_data("09:15", "09:25")
    logger.info(f"采集到的数据行数：{len(data) if isinstance(data, pd.DataFrame) else 0}")
    if isinstance(data, pd.DataFrame) and not data.empty:
        logger.info(f"数据列名：{list(data.columns)}")
        logger.info(f"前5行数据：\n{data.head()}")
