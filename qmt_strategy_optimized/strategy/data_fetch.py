#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据获取模块
支持从xtquant实时服务和QMT本地数据目录两种方式获取数据
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from xtquant import xtdata
from utils.logger import setup_logger
from utils.qmt_data_reader import QMTHistoricalDataReader, get_qmt_data_dir
import config

logger = setup_logger()


class DataFetcher:
    """数据获取类，支持从xtquant和本地QMT数据目录获取数据"""
    
    def __init__(self, use_local_data: bool = True, local_data_dir: str = None):
        """
        初始化数据获取器
        
        Args:
            use_local_data: 是否优先使用本地QMT数据目录
            local_data_dir: 本地QMT数据目录路径
        """
        logger.info("初始化数据获取器")
        
        # 初始化本地数据读取器
        self.use_local_data = use_local_data
        if use_local_data:
            if local_data_dir is None:
                local_data_dir = get_qmt_data_dir()
            self.local_reader = QMTHistoricalDataReader(local_data_dir)
            logger.info("使用本地QMT数据目录: {}".format(local_data_dir))
        
        # 初始化xtdata（需要连接QMT客户端）
        self._init_xtdata()
        
        # 获取股票列表
        self.stock_list = self._get_stock_list()
        logger.info("获取到股票列表，共{}只股票".format(len(self.stock_list)))
    
    def _init_xtdata(self):
        """初始化xtdata模块"""
        try:
            logger.info("初始化xtdata模块")
            # xtdata初始化
            # 注意：这里不主动连接，让后续操作自动处理
            self.xtdata_available = True
        except Exception as e:
            logger.warning("xtdata初始化失败: {}，将使用本地数据".format(e))
            self.xtdata_available = False
    
    def _get_stock_list(self):
        """
        获取股票列表
        """
        if config.STOCK_LIST:
            return config.STOCK_LIST
        
        # 优先从本地数据目录获取股票列表
        if self.use_local_data:
            try:
                sh_list = self.local_reader.get_stock_list("SH")
                sz_list = self.local_reader.get_stock_list("SZ")
                stock_list = ["{}.{}".format(s, m) for s, m in 
                             [(s, "SH") for s in sh_list] + [(s, "SZ") for s in sz_list]]
                if stock_list:
                    logger.info("从本地数据目录获取到{}只股票".format(len(stock_list)))
                    return stock_list
            except Exception as e:
                logger.warning("从本地数据目录获取股票列表失败: {}".format(e))
        
        # 从xtdata获取
        if self.xtdata_available:
            try:
                stock_list = xtdata.get_stock_list_in_sector('沪深A股')
                if stock_list:
                    logger.info("从xtdata获取到{}只股票".format(len(stock_list)))
                    return stock_list
            except Exception as e:
                logger.warning("从xtdata获取股票列表失败: {}".format(e))
        
        # 使用默认股票列表作为备选
        logger.info("使用默认股票列表作为备选")
        return ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH", "600002.SH", 
                "000004.SZ", "000005.SZ", "600003.SH", "600004.SH", "600005.SH"]
    
    def get_historical_data(self, start_date, end_date, market: str = "ALL", batch_size: int = 50):
        """
        获取历史数据
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            market: 市场类型，可选值：'SH'（上海）、'SZ'（深圳）、'ALL'（全部）
            batch_size: 分批处理的股票数量，默认50
            
        Returns:
            字典格式的历史数据，{stock_code: DataFrame}
        """
        logger.info("获取历史数据，时间范围：{} 至 {}，市场：{}，分批大小：{}".format(start_date, end_date, market, batch_size))
        
        all_data = {}
        
        # 确定需要获取的市场
        markets = []
        if market == "ALL":
            markets = ["SH", "SZ"]
        else:
            markets = [market]
        
        for mkt in markets:
            logger.info("获取{}市场数据".format(mkt))
            
            # 优先从本地数据目录获取
            if self.use_local_data:
                try:
                    data = self.local_reader.get_historical_data(
                        stock_list=None,
                        start_date=start_date,
                        end_date=end_date,
                        market=mkt
                    )
                    if data:
                        # 添加市场后缀
                        for stock_code in list(data.keys()):
                            full_code = "{}.{}".format(stock_code, mkt)
                            data[full_code] = data.pop(stock_code)
                        all_data.update(data)
                        logger.info("从本地数据目录获取{}市场{}只股票".format(mkt, len(data)))
                        continue  # 成功获取，跳过其他方式
                except Exception as e:
                    logger.warning("从本地数据目录获取{}市场数据失败: {}".format(mkt, e))
            
            # 从xtdata获取，使用分批处理
            if self.xtdata_available:
                try:
                    # 获取市场股票列表
                    market_stocks = self._get_market_stock_list(mkt)
                    if not market_stocks:
                        logger.warning(f"{mkt}市场没有股票列表")
                        continue
                    
                    # 分批获取数据
                    xtdata_data = {}
                    for i in range(0, len(market_stocks), batch_size):
                        batch_stocks = market_stocks[i:i+batch_size]
                        logger.info(f"从xtdata获取第{i//batch_size+1}批数据，共{len(batch_stocks)}只股票")
                        
                        # 使用xtdata获取当前批次的历史数据
                        batch_data = xtdata.get_market_data(
                            field_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
                            stock_list=batch_stocks,
                            period='1d',
                            start_time=start_date,
                            end_time=end_date
                        )
                        
                        if batch_data:
                            xtdata_data.update(batch_data)
                            logger.info(f"第{i//batch_size+1}批数据获取成功，共{len(batch_data)}只股票")
                    
                    if xtdata_data:
                        all_data.update(xtdata_data)
                        logger.info("从xtdata获取{}市场{}只股票".format(mkt, len(xtdata_data)))
                        continue
                except Exception as e:
                    logger.warning("从xtdata获取{}市场数据失败: {}".format(mkt, e))
            
            # 未获取到数据，跳过该市场
            logger.warning(f"未能获取{market}市场的数据，跳过该市场")
        
        # 保存历史数据到本地
        self._save_historical_data(all_data, start_date, end_date)
        
        logger.info("历史数据获取完成，共{}只股票".format(len(all_data)))
        return all_data
    
    def _get_xtdata_historical(self, start_date, end_date, market: str) -> dict:
        """
        从xtdata获取历史数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            market: 市场代码
            
        Returns:
            字典格式的历史数据
        """
        # 获取市场股票列表
        stock_list = self._get_market_stock_list(market)
        if not stock_list:
            return {}
        
        try:
            historical_data = xtdata.get_market_data(
                field_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
                stock_list=stock_list,
                period='1d',
                start_time=start_date,
                end_time=end_date
            )
            return historical_data
        except Exception as e:
            logger.error("xtdata获取历史数据失败: {}".format(e))
            return {}
    
    def _get_market_stock_list(self, market: str) -> list:
        """
        获取指定市场的股票列表
        
        Args:
            market: 市场代码
            
        Returns:
            股票代码列表
        """
        prefix = "" if market == "SH" else ""
        return [s for s in self.stock_list if ".{}".format(market) in s or 
                (market == "SH" and not ".SZ" in s and ".SH" in s) or
                (market == "SZ" and ".SZ" in s)]
    
    def get_historical_data_dataframe(self, start_date, end_date, 
                                       market: str = "ALL") -> pd.DataFrame:
        """
        获取历史数据，返回合并的DataFrame
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            market: 市场类型
            
        Returns:
            合并的DataFrame
        """
        data_dict = self.get_historical_data(start_date, end_date, market)
        
        if not data_dict:
            logger.warning("未获取到任何数据")
            return pd.DataFrame()
        
        # 合并所有股票数据
        all_data = []
        for stock_code, stock_data in data_dict.items():
            # 检查stock_data的类型
            if isinstance(stock_data, pd.DataFrame):
                df = stock_data.copy()
                if not df.empty:
                    df['stock_code'] = stock_code
                    all_data.append(df)
            elif isinstance(stock_data, dict):
                # 如果是字典格式，转换为DataFrame
                try:
                    df = pd.DataFrame(stock_data)
                    if not df.empty:
                        df['stock_code'] = stock_code
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"转换股票 {stock_code} 的数据为DataFrame失败: {e}")
            else:
                logger.warning(f"股票 {stock_code} 的数据类型不支持: {type(stock_data)}")
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        
        # 确保必要列存在
        required_cols = ['stock_code', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in result.columns:
                result[col] = np.nan
        
        # 选择必要列并排序
        result = result[required_cols]
        result = result.sort_values(['stock_code', 'trade_date'])
        
        logger.info("合并数据完成，总记录数: {}".format(len(result)))
        return result
    
    def get_realtime_data(self):
        """获取实时数据"""
        logger.info("获取实时数据")
        
        if not self.xtdata_available:
            logger.warning("xtdata不可用，无法获取实时数据")
            return None
        
        try:
            realtime_data = xtdata.get_full_tick(self.stock_list)
            return realtime_data
        except Exception as e:
            logger.error("获取实时数据失败: {}".format(e))
            return None
    
    def _save_historical_data(self, data, start_date, end_date):
        """保存历史数据到本地"""
        logger.info("保存历史数据到本地，路径：{}".format(config.HISTORY_DATA_PATH))
        
        os.makedirs(config.HISTORY_DATA_PATH, exist_ok=True)
        
        file_path = os.path.join(config.HISTORY_DATA_PATH, "history_{}_{}.pkl".format(start_date, end_date))
        try:
            pd.to_pickle(data, file_path)
            logger.info("历史数据保存成功：{}".format(file_path))
        except Exception as e:
            logger.error("保存历史数据失败: {}".format(e))
    
    def get_bid_data(self, date):
        """获取指定日期的集合竞价数据
        
        Args:
            date: 日期，格式为YYYYMMDD
            
        Returns:
            字典格式的集合竞价数据，{stock_code: DataFrame}
        """
        logger.info("获取指定日期的集合竞价数据：{}".format(date))
        
        bid_data = {}
        
        # 尝试从xtdata获取集合竞价数据
        if self.xtdata_available:
            try:
                # 使用xtdata的get_market_data获取1分钟K线数据，筛选9:15-9:25的数据
                bid_data = xtdata.get_market_data(
                    field_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
                    stock_list=self.stock_list,
                    period='1m',
                    start_time='{} 09:15:00'.format(date),
                    end_time='{} 09:25:00'.format(date)
                )
                logger.info("从xtdata获取到集合竞价数据，共{}只股票".format(len(bid_data)))
            except Exception as e:
                logger.warning("从xtdata获取集合竞价数据失败: {}".format(e))
        
        # 如果xtdata获取失败，返回空数据
        if not bid_data:
            logger.warning("未能获取集合竞价数据，返回空数据")
            bid_data = {}
        
        return bid_data
    
    def get_market_sentiment_data(self, date):
        """获取指定日期的市场情绪数据"""
        logger.info("获取指定日期的市场情绪数据：{}".format(date))
        
        if not self.xtdata_available:
            logger.warning("xtdata不可用")
            return None
        
        try:
            index_data = xtdata.get_market_data(
                field_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
                stock_list=['000001.SH', '399001.SZ'],
                period='1d',
                start_time=date,
                end_time=date
            )
            
            market_sentiment = {
                'index_data': index_data,
                'up_down_ratio': 0,
                'profit_effect': 0
            }
            
            return market_sentiment
        except Exception as e:
            logger.error("获取市场情绪数据失败: {}".format(e))
            return None
    
    def get_industry_data(self, date):
        """获取指定日期的行业数据"""
        logger.info("获取指定日期的行业数据：{}".format(date))
        return None


def get_stock_list_from_local_data(data_dir: str = None) -> list:
    """
    从本地QMT数据目录获取股票列表
    
    Args:
        data_dir: QMT数据目录路径
        
    Returns:
        股票代码列表
    """
    reader = QMTHistoricalDataReader(data_dir)
    
    sh_list = reader.get_stock_list("SH")
    sz_list = reader.get_stock_list("SZ")
    
    stock_list = ["{}{}".format(s, m) for s, m in 
                 [(s, "SH") for s in sh_list] + [(s, "SZ") for s in sz_list]]
    
    return stock_list


if __name__ == "__main__":
    # 测试代码
    fetcher = DataFetcher(use_local_data=True)
    
    # 获取历史数据
    data = fetcher.get_historical_data("20250101", "20251231", "SH")
    print("获取到{}只股票的数据".format(len(data)))
    
    if data:
        # 合并数据
        df = fetcher.get_historical_data_dataframe("20250101", "20251231", "SH")
        print("数据形状:", df.shape)
        print("数据列:", df.columns.tolist())
        print("日期范围:", df['trade_date'].min(), "到", df['trade_date'].max())
        
        # 按日期统计
        daily_counts = df.groupby('trade_date').size()
        print("每日平均股票数:", daily_counts.mean())
