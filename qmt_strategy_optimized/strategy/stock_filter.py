#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票预过滤模块
负责在模型预测前对股票进行初步筛选，减少模型调用次数
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger()

class StockFilter:
    """股票预过滤类"""
    
    def __init__(self):
        """初始化股票过滤器"""
        logger.info("初始化股票过滤器")
        self.filter_stats = {}
    
    def filter_stocks(self, stocks_data, config=None):
        """
        对股票进行预过滤
        
        Args:
            stocks_data: 股票数据
            config: 过滤配置
            
        Returns:
            过滤后的股票数据
        """
        logger.info("开始股票预过滤")
        
        if stocks_data is None:
            logger.warning("股票数据为空，跳过过滤")
            return stocks_data
        
        # 检查数据类型并判断是否为空
        if isinstance(stocks_data, pd.DataFrame):
            if stocks_data.empty:
                logger.warning("股票数据为空，跳过过滤")
                return stocks_data
        elif isinstance(stocks_data, dict):
            if not stocks_data:
                logger.warning("股票数据为空，跳过过滤")
                return stocks_data
        else:
            # 其他数据类型，尝试判断是否为空
            try:
                if len(stocks_data) == 0:
                    logger.warning("股票数据为空，跳过过滤")
                    return stocks_data
            except:
                logger.warning("无法判断股票数据是否为空，跳过过滤")
                return stocks_data
        
        # 默认过滤配置
        default_config = {
            # 流动性过滤
            'min_volume': 100000,  # 日成交量最小值（手）
            'min_turnover': 10000000,  # 日成交额最小值（元）
            
            # 价格过滤
            'min_price': 5.0,  # 最小价格
            'max_price': 500.0,  # 最大价格
            
            # 市值过滤
            'min_market_cap': 1000000000,  # 最小流通市值（元）
            
            # 涨跌幅过滤
            'max_bid_change': 0.2,  # 竞价期间最大涨跌幅
            'min_bid_change': -0.2,  # 竞价期间最小涨跌幅
            
            # ST股票过滤
            'exclude_st': True,  # 排除ST股票
            
            # 北交所股票过滤
            'exclude_north': True,  # 排除北交所股票
            
            # 新股过滤
            'exclude_new_stocks': True,  # 排除新股（上市不满30天）
            'new_stock_days': 30  # 新股定义天数
        }
        
        config = config or default_config
        
        original_count = len(stocks_data)
        filtered_stocks = stocks_data.copy()
        
        # 记录过滤前的股票数量
        self.filter_stats['original_count'] = original_count
        
        try:
            # 1. 排除北交所股票（代码以43或83开头）
            if config.get('exclude_north', True):
                if isinstance(filtered_stocks, pd.DataFrame):
                    if 'stock_code' in filtered_stocks.columns:
                        filtered_stocks = filtered_stocks[~filtered_stocks['stock_code'].str.startswith('43') & ~filtered_stocks['stock_code'].str.startswith('83')]
                elif isinstance(filtered_stocks, dict):
                    filtered_stocks = {k: v for k, v in filtered_stocks.items() if not (k.startswith('43') or k.startswith('83'))}
                self.filter_stats['north_excluded'] = original_count - len(filtered_stocks)
                original_count = len(filtered_stocks)
            
            # 2. 排除ST股票
            if config.get('exclude_st', True):
                if isinstance(filtered_stocks, pd.DataFrame):
                    if 'name' in filtered_stocks.columns:
                        filtered_stocks = filtered_stocks[~filtered_stocks['name'].str.contains('ST')]
                elif isinstance(filtered_stocks, dict):
                    filtered_stocks = {k: v for k, v in filtered_stocks.items() if isinstance(v, dict) and 'name' in v and 'ST' not in v['name']}
                self.filter_stats['st_excluded'] = original_count - len(filtered_stocks)
                original_count = len(filtered_stocks)
            
            # 3. 流动性过滤
            if isinstance(filtered_stocks, pd.DataFrame):
                # 成交量过滤
                if 'volume' in filtered_stocks.columns:
                    filtered_stocks = filtered_stocks[filtered_stocks['volume'] >= config['min_volume']]
                    self.filter_stats['low_volume_excluded'] = original_count - len(filtered_stocks)
                    original_count = len(filtered_stocks)
                
                # 成交额过滤
                if 'turnover' in filtered_stocks.columns:
                    filtered_stocks = filtered_stocks[filtered_stocks['turnover'] >= config['min_turnover']]
                    self.filter_stats['low_turnover_excluded'] = original_count - len(filtered_stocks)
                    original_count = len(filtered_stocks)
            
            # 4. 价格过滤
            if isinstance(filtered_stocks, pd.DataFrame):
                if 'close' in filtered_stocks.columns:
                    filtered_stocks = filtered_stocks[(filtered_stocks['close'] >= config['min_price']) & (filtered_stocks['close'] <= config['max_price'])]
                elif 'open' in filtered_stocks.columns:
                    filtered_stocks = filtered_stocks[(filtered_stocks['open'] >= config['min_price']) & (filtered_stocks['open'] <= config['max_price'])]
                self.filter_stats['price_filtered'] = original_count - len(filtered_stocks)
                original_count = len(filtered_stocks)
            
            # 5. 市值过滤
            if isinstance(filtered_stocks, pd.DataFrame):
                if 'circulating_market_cap' in filtered_stocks.columns:
                    filtered_stocks = filtered_stocks[filtered_stocks['circulating_market_cap'] >= config['min_market_cap']]
                    self.filter_stats['market_cap_filtered'] = original_count - len(filtered_stocks)
                    original_count = len(filtered_stocks)
            
            # 6. 涨跌幅过滤
            if isinstance(filtered_stocks, pd.DataFrame):
                if 'bid_price_change' in filtered_stocks.columns:
                    filtered_stocks = filtered_stocks[
                        (filtered_stocks['bid_price_change'] >= config['min_bid_change']) & 
                        (filtered_stocks['bid_price_change'] <= config['max_bid_change'])
                    ]
                    self.filter_stats['bid_change_filtered'] = original_count - len(filtered_stocks)
                    original_count = len(filtered_stocks)
            
            # 7. 新股过滤
            if config.get('exclude_new_stocks', True):
                if isinstance(filtered_stocks, pd.DataFrame):
                    if 'list_date' in filtered_stocks.columns:
                        # 计算上市天数
                        filtered_stocks['list_days'] = (pd.Timestamp.now() - pd.to_datetime(filtered_stocks['list_date'])).dt.days
                        filtered_stocks = filtered_stocks[filtered_stocks['list_days'] >= config['new_stock_days']]
                        self.filter_stats['new_stocks_excluded'] = original_count - len(filtered_stocks)
                        original_count = len(filtered_stocks)
            
            # 记录过滤后的股票数量
            self.filter_stats['filtered_count'] = len(filtered_stocks)
            self.filter_stats['filter_ratio'] = 1 - (len(filtered_stocks) / self.filter_stats['original_count'])
            
            logger.info(f"股票预过滤完成，过滤前{self.filter_stats['original_count']}只，过滤后{self.filter_stats['filtered_count']}只，过滤比例{self.filter_stats['filter_ratio']:.2%}")
            logger.info(f"过滤统计：{self.filter_stats}")
            
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"股票过滤失败：{e}")
            return stocks_data
    
    def get_filter_stats(self):
        """
        获取过滤统计信息
        
        Returns:
            过滤统计字典
        """
        return self.filter_stats
    
    def evaluate_filter_effectiveness(self, filtered_stocks, all_stocks, y_true):
        """
        评估过滤效果
        
        Args:
            filtered_stocks: 过滤后的股票
            all_stocks: 所有股票
            y_true: 真实标签
            
        Returns:
            过滤效果评估结果
        """
        logger.info("评估过滤效果")
        
        try:
            # 计算过滤前后的涨停股票数量
            if isinstance(all_stocks, pd.DataFrame) and isinstance(y_true, pd.Series):
                # 过滤前的涨停股票数量
                all_limit_up = sum(y_true)
                
                # 过滤后的涨停股票数量
                filtered_limit_up = sum(y_true.loc[filtered_stocks.index])
                
                # 计算过滤效率指标
                effectiveness = {
                    'all_stocks': len(all_stocks),
                    'filtered_stocks': len(filtered_stocks),
                    'all_limit_up': all_limit_up,
                    'filtered_limit_up': filtered_limit_up,
                    'limit_up_retained_ratio': filtered_limit_up / all_limit_up if all_limit_up > 0 else 0,
                    'stock_retained_ratio': len(filtered_stocks) / len(all_stocks),
                    'filter_efficiency': (filtered_limit_up / len(filtered_stocks)) / (all_limit_up / len(all_stocks)) if all_limit_up > 0 and len(filtered_stocks) > 0 else 0
                }
                
                logger.info(f"过滤效果评估：{effectiveness}")
                return effectiveness
            else:
                logger.warning("无法评估过滤效果，数据格式不支持")
                return {}
                
        except Exception as e:
            logger.error(f"评估过滤效果失败：{e}")
            return {}
    
    def adaptive_filter(self, stocks_data, market_conditions):
        """
        自适应过滤，根据市场环境调整过滤参数
        
        Args:
            stocks_data: 股票数据
            market_conditions: 市场环境数据
            
        Returns:
            过滤后的股票数据
        """
        logger.info("使用自适应过滤")
        
        # 根据市场环境调整过滤参数
        base_config = {
            'min_volume': 100000,
            'min_price': 5.0,
            'max_price': 500.0,
            'min_market_cap': 1000000000
        }
        
        # 根据市场成交量调整流动性过滤参数
        if market_conditions.get('market_volume'):
            # 市场成交量较高时，降低流动性要求
            if market_conditions['market_volume'] > 1000000000000:  # 万亿成交量
                base_config['min_volume'] = 80000
                base_config['min_market_cap'] = 800000000
            # 市场成交量较低时，提高流动性要求
            elif market_conditions['market_volume'] < 500000000000:  # 5000亿成交量
                base_config['min_volume'] = 150000
                base_config['min_market_cap'] = 1500000000
        
        # 根据市场情绪调整涨跌幅过滤参数
        if market_conditions.get('market_sentiment'):
            if market_conditions['market_sentiment'] > 0.7:  # 市场情绪乐观
                base_config['max_bid_change'] = 0.25
            elif market_conditions['market_sentiment'] < 0.3:  # 市场情绪悲观
                base_config['max_bid_change'] = 0.15
        
        logger.info(f"自适应过滤参数：{base_config}")
        return self.filter_stocks(stocks_data, base_config)
    
    def rule_based_filter(self, stocks_data, rules):
        """
        基于规则的过滤
        
        Args:
            stocks_data: 股票数据
            rules: 过滤规则列表
            
        Returns:
            过滤后的股票数据
        """
        logger.info("使用基于规则的过滤")
        
        filtered_stocks = stocks_data.copy()
        
        for rule in rules:
            try:
                if isinstance(filtered_stocks, pd.DataFrame):
                    # 执行过滤规则
                    filtered_stocks = filtered_stocks.query(rule)
            except Exception as e:
                logger.error(f"规则{rule}执行失败：{e}")
        
        return filtered_stocks
