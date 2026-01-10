#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据源模块
提供数据采集所需的基础数据，如股票列表、历史数据等
"""

import pandas as pd
import numpy as np
from utils.logger import setup_logger

logger = setup_logger()


class DataSource:
    """数据源类，提供数据采集所需的基础数据"""
    
    def __init__(self):
        """初始化数据源"""
        logger.info("初始化数据源")
        self.stock_list = self._load_stock_list()
        self.sector_mapping = self._load_sector_mapping()
    
    def _load_stock_list(self):
        """加载股票列表"""
        logger.info("加载股票列表")
        
        # 模拟股票列表数据，实际实现中应从真实数据源获取
        # 生成一些模拟的股票代码
        stock_codes = []
        
        # 生成上海股票代码（600000-603999）
        for i in range(1000, 1100):
            stock_codes.append(f"600{i:03d}.SH")
        
        # 生成深圳股票代码（000000-002999）
        for i in range(1000, 1100):
            stock_codes.append(f"000{i:03d}.SZ")
        
        # 生成创业板股票代码（300000-300999）
        for i in range(1000, 1100):
            stock_codes.append(f"300{i:03d}.SZ")
        
        logger.info(f"加载了{len(stock_codes)}只股票")
        return stock_codes
    
    def _load_sector_mapping(self):
        """加载股票板块映射关系"""
        logger.info("加载股票板块映射关系")
        
        # 模拟板块映射数据，实际实现中应从真实数据源获取
        sectors = [
            "金融", "医药", "科技", "消费", "新能源", 
            "地产", "化工", "有色", "机械", "通信"
        ]
        
        # 为每只股票分配一个板块
        sector_mapping = {}
        for i, stock in enumerate(self.stock_list):
            sector_index = i % len(sectors)
            sector_mapping[stock] = sectors[sector_index]
        
        logger.info(f"加载了{len(sector_mapping)}只股票的板块映射")
        return sector_mapping
    
    def get_stock_list(self):
        """获取股票列表
        
        Returns:
            股票列表
        """
        return self.stock_list
    
    def get_sector_mapping(self):
        """获取股票板块映射关系
        
        Returns:
            股票板块映射字典
        """
        return self.sector_mapping
    
    def get_stocks_by_sector(self, sector):
        """根据板块获取股票列表
        
        Args:
            sector: 板块名称
            
        Returns:
            该板块下的股票列表
        """
        sector_stocks = [stock for stock, s in self.sector_mapping.items() if s == sector]
        logger.info(f"板块{sector}下共有{len(sector_stocks)}只股票")
        return sector_stocks
    
    def get_historical_data(self, stock_code, start_date, end_date):
        """获取股票历史数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            股票历史数据DataFrame
        """
        logger.info(f"获取股票{stock_code}的历史数据，时间范围：{start_date} - {end_date}")
        
        # 模拟历史数据，实际实现中应从真实数据源获取
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 生成模拟的历史数据
        historical_data = {
            'date': date_range,
            'open': np.random.rand(len(date_range)) * 100 + 10,
            'high': np.random.rand(len(date_range)) * 10 + 100,
            'low': np.random.rand(len(date_range)) * 10 + 90,
            'close': np.random.rand(len(date_range)) * 100 + 10,
            'volume': np.random.randint(1000000, 100000000, len(date_range)),
            'amount': np.random.randint(10000000, 1000000000, len(date_range)),
            'turnover_rate': np.random.rand(len(date_range)) * 0.1 + 0.01
        }
        
        df = pd.DataFrame(historical_data)
        df['stock_code'] = stock_code
        
        logger.info(f"获取到{len(df)}行历史数据")
        return df
    
    def get_market_index_data(self, index_code, start_date, end_date):
        """获取市场指数数据
        
        Args:
            index_code: 指数代码，如"000001.SH"（上证指数）
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            指数历史数据DataFrame
        """
        logger.info(f"获取指数{index_code}的历史数据，时间范围：{start_date} - {end_date}")
        
        # 模拟指数数据，实际实现中应从真实数据源获取
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 生成模拟的指数数据
        index_data = {
            'date': date_range,
            'open': np.random.rand(len(date_range)) * 100 + 3000,
            'high': np.random.rand(len(date_range)) * 20 + 3000,
            'low': np.random.rand(len(date_range)) * 20 + 2980,
            'close': np.random.rand(len(date_range)) * 100 + 3000,
            'volume': np.random.randint(10000000, 1000000000, len(date_range)),
            'amount': np.random.randint(100000000, 10000000000, len(date_range))
        }
        
        df = pd.DataFrame(index_data)
        df['index_code'] = index_code
        
        logger.info(f"获取到{len(df)}行指数数据")
        return df
    
    def get_sector_data(self, sector, start_date, end_date):
        """获取板块数据
        
        Args:
            sector: 板块名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            板块历史数据DataFrame
        """
        logger.info(f"获取板块{sector}的历史数据，时间范围：{start_date} - {end_date}")
        
        # 模拟板块数据，实际实现中应从真实数据源获取
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 生成模拟的板块数据
        sector_data = {
            'date': date_range,
            'sector': sector,
            'change': np.random.rand(len(date_range)) * 0.05 - 0.025,
            'volume': np.random.randint(100000000, 10000000000, len(date_range)),
            'fund_inflow': np.random.randint(-500000000, 500000000, len(date_range)),
            'limit_up_count': np.random.randint(0, 20, len(date_range))
        }
        
        df = pd.DataFrame(sector_data)
        
        logger.info(f"获取到{len(df)}行板块数据")
        return df
    
    def get_stock_basic_info(self, stock_code):
        """获取股票基本信息
        
        Args:
            stock_code: 股票代码
            
        Returns:
            股票基本信息字典
        """
        logger.info(f"获取股票{stock_code}的基本信息")
        
        # 模拟股票基本信息，实际实现中应从真实数据源获取
        basic_info = {
            'stock_code': stock_code,
            'stock_name': f"股票{stock_code[:6]}",
            'industry': np.random.choice(["金融", "医药", "科技", "消费", "新能源"]),
            'sector': self.sector_mapping.get(stock_code, "未知"),
            'total_share': np.random.uniform(100000000, 1000000000),
            'circulating_share': np.random.uniform(50000000, 500000000),
            'market_cap': np.random.uniform(5000000000, 50000000000),
            'circulating_cap': np.random.uniform(2000000000, 20000000000),
            'pe_ratio': np.random.rand() * 50 + 5,
            'pb_ratio': np.random.rand() * 5 + 1,
            'list_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(365, 3650))
        }
        
        logger.info(f"获取到股票{stock_code}的基本信息")
        return basic_info
    
    def get_realtime_data(self, stock_code):
        """获取股票实时数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            股票实时数据字典
        """
        logger.info(f"获取股票{stock_code}的实时数据")
        
        # 模拟实时数据，实际实现中应从真实数据源获取
        realtime_data = {
            'stock_code': stock_code,
            'last_price': np.random.rand() * 100 + 10,
            'open': np.random.rand() * 100 + 10,
            'high': np.random.rand() * 10 + 100,
            'low': np.random.rand() * 10 + 90,
            'close': np.random.rand() * 100 + 10,
            'volume': np.random.randint(1000000, 100000000),
            'amount': np.random.randint(10000000, 1000000000),
            'turnover_rate': np.random.rand() * 0.1 + 0.01,
            'change': np.random.rand() * 0.05 - 0.025,
            'change_pct': (np.random.rand() * 5 - 2.5),
            'bid_price1': np.random.rand() * 100 + 10,
            'bid_volume1': np.random.randint(100, 10000),
            'ask_price1': np.random.rand() * 100 + 10,
            'ask_volume1': np.random.randint(100, 10000),
            'update_time': pd.Timestamp.now()
        }
        
        logger.info(f"获取到股票{stock_code}的实时数据")
        return realtime_data
