#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QMT竞价数据获取服务
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd
import numpy as np
from datetime import datetime
from xtquant import xtdata
from qmt_strategy.utils.logger import setup_logger

logger = setup_logger()


class QMTBidDataFetcher:
    """
    QMT竞价数据获取器，用于获取当天的股票竞价数据
    """
    
    def __init__(self, stock_list=None):
        """
        初始化QMT竞价数据获取器
        
        Args:
            stock_list: 股票代码列表，格式如['000001.SZ', '600000.SH']
        """
        logger.info("初始化QMT竞价数据获取器")
        self.stock_list = stock_list
        self._init_xtdata()
        
        # 如果没有提供股票列表，从xtdata获取沪深A股列表
        if not self.stock_list:
            self.stock_list = self._get_default_stock_list()
        
        logger.info(f"获取到股票列表，共{len(self.stock_list)}只股票")
    
    def _init_xtdata(self):
        """
        初始化xtdata模块
        """
        try:
            logger.info("初始化xtdata模块")
            self.xtdata_available = True
        except Exception as e:
            logger.error(f"xtdata初始化失败: {e}")
            self.xtdata_available = False
    
    def _get_default_stock_list(self):
        """
        获取默认股票列表（沪深A股）
        
        Returns:
            list: 股票代码列表
        """
        if not self.xtdata_available:
            logger.warning("xtdata不可用，返回空股票列表")
            return []
        
        try:
            stock_list = xtdata.get_stock_list_in_sector('沪深A股')
            return stock_list
        except Exception as e:
            logger.error(f"从xtdata获取股票列表失败: {e}")
            # 返回一些默认股票作为备选
            return ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH", "600002.SH"]
    
    def get_today_bid_data(self):
        """
        获取当天的集合竞价数据
        
        Returns:
            pd.DataFrame: 集合竞价数据，包含所有股票的竞价信息
        """
        today = datetime.now().strftime("%Y%m%d")
        logger.info(f"获取{today}的集合竞价数据")
        return self.get_bid_data(today)
    
    def get_bid_data(self, date):
        """
        获取指定日期的集合竞价数据
        
        Args:
            date: 日期，格式为YYYYMMDD
            
        Returns:
            pd.DataFrame: 集合竞价数据，包含所有股票的竞价信息
        """
        logger.info(f"获取{date}的集合竞价数据")
        
        if not self.xtdata_available:
            logger.error("xtdata不可用，无法获取集合竞价数据")
            return pd.DataFrame()
        
        try:
            # 使用xtdata的get_market_data获取1分钟K线数据，筛选9:15-9:25的数据
            bid_data = xtdata.get_market_data(
                field_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
                stock_list=self.stock_list,
                period='1m',
                start_time=f'{date} 09:15:00',
                end_time=f'{date} 09:25:00'
            )
            
            if not bid_data:
                logger.warning(f"未获取到{date}的集合竞价数据")
                return pd.DataFrame()
            
            # 处理数据，转换为DataFrame格式
            processed_data = self._process_bid_data(bid_data, date)
            logger.info(f"处理完成，共{len(processed_data)}只股票的集合竞价数据")
            return processed_data
            
        except Exception as e:
            logger.error(f"获取集合竞价数据失败: {e}")
            return pd.DataFrame()
    
    def _process_bid_data(self, bid_data, date):
        """
        处理原始竞价数据
        
        Args:
            bid_data: xtdata返回的原始竞价数据
            date: 日期，格式为YYYYMMDD
            
        Returns:
            pd.DataFrame: 处理后的集合竞价数据
        """
        processed_data = []
        
        for stock_code, stock_data in bid_data.items():
            try:
                # 转换为DataFrame
                df = pd.DataFrame(stock_data)
                if df.empty:
                    continue
                
                # 添加股票代码
                df['stock_code'] = stock_code
                df['date'] = date
                
                # 保留9:25的竞价数据（最终竞价结果）
                df = df[df.index == f'{date} 09:25:00']
                
                if not df.empty:
                    processed_data.append(df)
            except Exception as e:
                logger.error(f"处理股票{stock_code}的竞价数据失败: {e}")
                continue
        
        if not processed_data:
            return pd.DataFrame()
        
        # 合并所有股票数据
        result = pd.concat(processed_data, ignore_index=True)
        
        # 重命名列名，确保列名统一
        column_mapping = {
            'open': 'bid_open',
            'high': 'bid_high',
            'low': 'bid_low',
            'close': 'bid_price',
            'volume': 'bid_volume',
            'amount': 'bid_amount'
        }
        result = result.rename(columns=column_mapping)
        
        # 确保必要列存在
        required_cols = ['stock_code', 'date', 'bid_open', 'bid_high', 'bid_low', 'bid_price', 'bid_volume', 'bid_amount']
        for col in required_cols:
            if col not in result.columns:
                result[col] = np.nan
        
        # 选择必要列
        result = result[required_cols]
        
        # 添加时间字段
        result['datetime'] = pd.to_datetime(result['date'] + ' 09:25:00')
        
        return result
    
    def save_bid_data(self, bid_data, file_path=None):
        """
        保存竞价数据到本地文件
        
        Args:
            bid_data: 竞价数据DataFrame
            file_path: 保存路径，默认为当前日期的CSV文件
            
        Returns:
            str: 实际保存的文件路径
        """
        if bid_data.empty:
            logger.warning("竞价数据为空，不保存")
            return None
        
        if not file_path:
            today = datetime.now().strftime("%Y%m%d")
            file_path = f"{today}_bid_data.csv"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            bid_data.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.info(f"竞价数据保存成功: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"保存竞价数据失败: {e}")
            return None


if __name__ == "__main__":
    # 测试代码
    fetcher = QMTBidDataFetcher()
    bid_data = fetcher.get_today_bid_data()
    print(f"获取到{len(bid_data)}条竞价数据")
    print(bid_data.head())
    
    # 保存数据
    if not bid_data.empty:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../data/{datetime.now().strftime('%Y%m%d')}_bid_data.csv")
        fetcher.save_bid_data(bid_data, save_path)
