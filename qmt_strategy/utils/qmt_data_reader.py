#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QMT DAT数据文件读取器
用于读取QMT本地数据目录中的股票历史数据
"""

import struct
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

try:
    from qmt_strategy.utils.logger import setup_logger
except ImportError:
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    def setup_logger():
        return logger

logger = setup_logger()

class QMTHistoricalDataReader:
    """QMT历史数据读取器"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化数据读取器
        
        Args:
            data_dir: QMT数据目录路径，默认为国金QMT模拟器的数据目录
        """
        if data_dir is None:
            data_dir = r"D:\apps\国金QMT交易端模拟\datadir"
        self.data_dir = data_dir
        
        # 数据目录结构
        self.day_data_dir = os.path.join(data_dir, "{market}", "86400")
        
        # 市场代码映射
        self.market_map = {
            "sh": "SH",
            "sz": "SZ",
            "SH": "SH",
            "SZ": "SZ",
            "shanghai": "SH",
            "shenzhen": "SZ",
            "上海": "SH",
            "深圳": "SZ"
        }
        
        # 记录大小（44字节）
        self.record_size = 44
        
        logger.info("初始化QMT历史数据读取器，数据目录: {}".format(data_dir))
    
    def _get_stock_code(self, stock_code: str) -> str:
        """
        转换股票代码格式
        
        Args:
            stock_code: 股票代码，如 '600000' 或 'sh600000'
            
        Returns:
            6位数字股票代码，如 '000001'
        """
        # 移除市场前缀
        code = stock_code.lower().replace('sh', '').replace('sz', '')
        # 确保是6位数字
        code = code.zfill(6)
        return code
    
    def _read_single_stock_data(self, stock_code: str, market: str = "SH") -> pd.DataFrame:
        """
        读取单个股票的历史数据
        
        Args:
            stock_code: 股票代码
            market: 市场代码（SH或SZ）
            
        Returns:
            包含历史数据的DataFrame
        """
        market = self.market_map.get(market, market)
        stock_code = self._get_stock_code(stock_code)
        
        # DAT文件路径
        dat_file = os.path.join(self.day_data_dir.format(market=market), "{}.DAT".format(stock_code))
        
        if not os.path.exists(dat_file):
            logger.warning("数据文件不存在: {}".format(dat_file))
            return pd.DataFrame()
        
        try:
            with open(dat_file, 'rb') as f:
                data = f.read()
            
            # 跳过8字节头部
            record_data = data[8:]
            
            # 解析所有记录
            records = []
            num_records = len(record_data) // self.record_size
            
            for i in range(num_records):
                record = record_data[i * self.record_size:(i + 1) * self.record_size]
                
                # 解析日期（小端序整数）
                date_val = struct.unpack('<i', record[0:4])[0]
                
                # 验证日期是否有效（2024-2030年）
                if date_val < 1700000000 or date_val > 1900000000:
                    continue
                
                date_obj = datetime.fromtimestamp(date_val)
                
                # 解析价格数据（小端序双精度浮点数）
                try:
                    open_price = struct.unpack('<d', record[8:16])[0]
                    high_price = struct.unpack('<d', record[16:24])[0]
                    low_price = struct.unpack('<d', record[24:32])[0]
                    close_price = struct.unpack('<d', record[32:40])[0]
                except:
                    continue
                
                # 解析成交量（小端序64位整数）
                try:
                    volume = struct.unpack('<q', record[40:48])[0]
                except:
                    volume = 0
                
                # 过滤无效数据
                if abs(open_price) > 1e10 or abs(close_price) > 1e10:
                    continue
                
                # 过滤价格为0的记录（无效数据）
                if open_price <= 0 and high_price <= 0 and low_price <= 0 and close_price <= 0:
                    continue
                
                # 过滤成交量为0的记录（无效数据）
                if volume <= 0:
                    continue
                
                records.append({
                    'stock_code': stock_code,
                    'trade_date': date_obj,
                    'date': date_obj.strftime('%Y%m%d'),
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'market': market
                })
            
            if records:
                df = pd.DataFrame(records)
                logger.debug("读取 {} 成功，共 {} 条记录".format(stock_code, len(df)))
                return df
            else:
                logger.warning("{} 无有效数据".format(stock_code))
                return pd.DataFrame()
                
        except Exception as e:
            logger.error("读取 {} 失败: {}".format(stock_code, e))
            return pd.DataFrame()
    
    def get_stock_list(self, market: str = "SH") -> List[str]:
        """
        获取指定市场的股票列表
        
        Args:
            market: 市场代码
            
        Returns:
            股票代码列表
        """
        market = self.market_map.get(market, market)
        day_data_path = self.day_data_dir.format(market=market)
        
        if not os.path.exists(day_data_path):
            logger.warning("数据目录不存在: {}".format(day_data_path))
            return []
        
        # 获取所有DAT文件
        files = [f for f in os.listdir(day_data_path) if f.endswith('.DAT')]
        
        # 转换为股票代码
        stock_list = [os.path.splitext(f)[0].lstrip('0') or '0' for f in files]
        stock_list = [s if s != '0' else '0' for s in stock_list]  # 处理000000的情况
        stock_list = [s.zfill(6) for s in stock_list]  # 确保6位
        
        logger.info("获取 {} 市场股票列表，共 {} 只".format(market, len(stock_list)))
        return stock_list
    
    def get_historical_data(self, stock_list: List[str] = None, 
                            start_date: str = None, 
                            end_date: str = None,
                            market: str = "SH") -> Dict[str, pd.DataFrame]:
        """
        获取多只股票的历史数据
        
        Args:
            stock_list: 股票代码列表，None表示获取所有股票
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            market: 市场代码
            
        Returns:
            字典格式的历史数据，{stock_code: DataFrame}
        """
        logger.info("开始获取历史数据，市场: {}, 股票数: {}, 时间范围: {} - {}".format(
            market, 
            len(stock_list) if stock_list else '全部',
            start_date or '全部',
            end_date or '全部'
        ))
        
        # 如果未指定股票列表，获取所有股票
        if stock_list is None:
            stock_list = self.get_stock_list(market)
        
        # 转换日期字符串为datetime对象
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y%m%d')
        else:
            start_dt = None
            
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y%m%d')
        else:
            end_dt = None
        
        # 获取所有股票的数据
        all_data = {}
        for i, stock_code in enumerate(stock_list):
            if (i + 1) % 100 == 0:
                logger.info("进度: {}/{}".format(i + 1, len(stock_list)))
            
            df = self._read_single_stock_data(stock_code, market)
            
            if df.empty:
                continue
            
            # 过滤日期范围
            if start_dt:
                df = df[df['trade_date'] >= start_dt]
            if end_dt:
                df = df[df['trade_date'] <= end_dt]
            
            if not df.empty:
                all_data[stock_code] = df
        
        logger.info("数据获取完成，共获取 {} 只股票的数据".format(len(all_data)))
        return all_data
    
    def get_historical_data_dataframe(self, stock_list: List[str] = None,
                                       start_date: str = None,
                                       end_date: str = None,
                                       market: str = "ALL") -> pd.DataFrame:
        """
        获取多只股票的历史数据，合并为单个DataFrame
        
        Args:
            stock_list: 股票代码列表，None表示获取所有股票
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            market: 市场代码
            
        Returns:
            合并的历史数据DataFrame
        """
        data_dict = self.get_historical_data(stock_list, start_date, end_date, market)
        
        if not data_dict:
            logger.warning("未获取到任何数据")
            return pd.DataFrame()
        
        # 合并所有股票数据
        all_data = pd.concat(data_dict.values(), ignore_index=True)
        
        # 按股票代码和日期排序
        all_data = all_data.sort_values(['stock_code', 'trade_date'])
        
        logger.info("合并数据完成，总记录数: {}".format(len(all_data)))
        return all_data
    
    def get_data_for_year(self, year: int, market: str = "SH") -> pd.DataFrame:
        """
        获取指定年份的所有股票数据
        
        Args:
            year: 年份，如 2025
            market: 市场代码
            
        Returns:
            合并的历史数据DataFrame
        """
        start_date = "{}0101".format(year)
        end_date = "{}1231".format(year)
        
        return self.get_historical_data_dataframe(
            stock_list=None,
            start_date=start_date,
            end_date=end_date,
            market=market
        )


# 便捷函数
def get_qmt_data_dir():
    """获取默认的QMT数据目录"""
    return r"D:\apps\国金QMT交易端模拟\datadir"


def create_data_reader(data_dir: str = None) -> QMTHistoricalDataReader:
    """创建QMT历史数据读取器"""
    return QMTHistoricalDataReader(data_dir)


if __name__ == "__main__":
    # 测试代码
    reader = create_data_reader()
    
    # 获取股票列表
    stock_list = reader.get_stock_list("SH")[:10]
    print("股票列表:", stock_list[:5])
    
    # 获取历史数据
    data = reader.get_historical_data(stock_list, "20250101", "20251231", "SH")
    print("数据字典键:", list(data.keys())[:5])
    
    if data:
        # 合并数据
        df = pd.concat(data.values(), ignore_index=True)
        print("数据形状:", df.shape)
        print("数据列:", df.columns.tolist())
        print("日期范围:", df['trade_date'].min(), "到", df['trade_date'].max())
