#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据获取模块
支持从xtquant实时服务、QMT本地数据目录和pandas_datareader数据源获取数据
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

# 尝试导入pandas_datareader
try:
    import pandas_datareader as pdr
    from pandas_datareader import data as web
    pandas_datareader_available = True
except ImportError:
    pandas_datareader_available = False

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
    
    def _get_pandas_datareader_historical(self, start_date, end_date, market: str = "ALL") -> dict:
        """
        从pandas_datareader获取历史数据
        
        Args:
            start_date: 开始日期，格式为YYYYMMDD
            end_date: 结束日期，格式为YYYYMMDD
            market: 市场类型，可选值：'SH'（上海）、'SZ'（深圳）、'ALL'（全部）
            
        Returns:
            字典格式的历史数据，{stock_code: DataFrame}
        """
        logger.info("从pandas_datareader获取历史数据，时间范围：{} 至 {}，市场：{}".format(start_date, end_date, market))
        
        if not pandas_datareader_available:
            logger.warning("pandas_datareader库未安装，无法从pandas_datareader获取数据")
            return {}
        
        try:
            # 转换日期格式
            start = pd.to_datetime(str(start_date))
            end = pd.to_datetime(str(end_date))
            
            # 使用示例股票列表进行测试，实际应用中可以从配置或其他来源获取
            # 注意：Yahoo Finance的股票代码格式与国内市场不同，需要转换
            sample_stocks = {
                '600000.SH': '600000.SS',  # 浦发银行
                '600004.SH': '600004.SS',  # 白云机场
                '600006.SH': '600006.SS',  # 东风汽车
                '600009.SH': '600009.SS',  # 上海机场
                '600016.SH': '600016.SS',  # 民生银行
                '600028.SH': '600028.SS',  # 中国石化
                '600036.SH': '600036.SS',  # 招商银行
                '600048.SH': '600048.SS',  # 保利发展
                '600104.SH': '600104.SS',  # 上汽集团
                '600111.SH': '600111.SS'   # 北方稀土
            }
            
            all_data = {}
            
            # 逐个获取股票数据
            for local_code, yahoo_code in sample_stocks.items():
                logger.info(f"从pandas_datareader获取股票 {local_code} 的历史数据")
                
                try:
                    # 使用Yahoo Finance获取数据
                    df = web.get_data_yahoo(yahoo_code, start=start, end=end)
                    
                    if df.empty:
                        logger.warning(f"股票 {local_code} 没有历史数据")
                        continue
                    
                    # 转换数据格式，使其与原有系统兼容
                    df = df.reset_index()
                    df = df.rename(columns={
                        'Date': 'trade_date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })
                    
                    # 添加amount列（成交额）
                    df['amount'] = df['close'] * df['volume']
                    
                    # 添加stock_code列
                    df['stock_code'] = local_code
                    
                    # 保留必要的列
                    df = df[['trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'stock_code']]
                    
                    # 将trade_date转换为YYYYMMDD格式的字符串
                    df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
                    
                    # 添加到结果中
                    all_data[local_code] = df
                    logger.info(f"成功获取股票 {local_code} 的历史数据，共 {len(df)} 条记录")
                    
                except Exception as e:
                    logger.warning(f"获取股票 {local_code} 数据失败: {e}")
                    continue
            
            logger.info("从pandas_datareader获取历史数据完成，共{}只股票".format(len(all_data)))
            return all_data
            
        except Exception as e:
            logger.error("从pandas_datareader获取历史数据失败: {}".format(e))
            import traceback
            logger.error(f"异常堆栈：{traceback.format_exc()}")
            return {}
    
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
        
        # 尝试从本地数据目录和xtdata获取数据
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
        
        # 如果从本地和xtdata都没有获取到数据，尝试从pandas_datareader获取
        if not all_data:
            logger.info("从本地和xtdata都没有获取到数据，尝试从pandas_datareader获取")
            pandas_data = self._get_pandas_datareader_historical(start_date, end_date, market)
            if pandas_data:
                all_data.update(pandas_data)
        
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
        final_bid_data = {}
        
        # 尝试从xtdata获取集合竞价数据
        if self.xtdata_available:
            try:
                # 1. 获取1分钟K线数据，包含9:15-9:25的竞价数据
                bid_data = xtdata.get_market_data(
                    field_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
                    stock_list=self.stock_list,
                    period='1m',
                    start_time='{} 09:15:00'.format(date),
                    end_time='{} 09:25:00'.format(date)
                )
                logger.info("从xtdata获取到集合竞价1分钟K线数据，共{}只股票".format(len(bid_data)))
                
                # 2. 获取9:26的最终竞价数据（使用日线数据，包含竞价结果）
                final_bid_data = xtdata.get_market_data(
                    field_list=['pre_close', 'open', 'high', 'low', 'close', 'volume', 'amount'],
                    stock_list=self.stock_list,
                    period='1d',
                    start_time=date,
                    end_time=date
                )
                logger.info("从xtdata获取到9:26最终竞价数据，共{}只股票".format(len(final_bid_data)))
                
            except Exception as e:
                logger.warning("从xtdata获取集合竞价数据失败: {}".format(e))
        
        # 处理并合并竞价数据
        processed_bid_data = {}
        
        # 遍历所有股票
        all_stocks = set(list(bid_data.keys()) + list(final_bid_data.keys()))
        
        for stock_code in all_stocks:
            try:
                stock_bid_data = None
                stock_final_data = None
                
                # 获取该股票的竞价数据
                if stock_code in bid_data:
                    stock_bid_data = bid_data[stock_code]
                
                # 获取该股票的最终竞价数据
                if stock_code in final_bid_data:
                    stock_final_data = final_bid_data[stock_code]
                
                # 创建处理后的数据
                if isinstance(stock_final_data, pd.DataFrame) and not stock_final_data.empty:
                    # 使用最终竞价数据作为基础
                    processed_df = stock_final_data.copy()
                    
                    # 添加股票代码
                    processed_df['stock_code'] = stock_code
                    
                    # 计算竞价相关指标
                    if 'pre_close' in processed_df.columns and 'open' in processed_df.columns:
                        # 竞价涨幅
                        processed_df['bid_change'] = (processed_df['open'] - processed_df['pre_close']) / processed_df['pre_close']
                        
                        # 竞价强度（考虑涨跌方向的强度）
                        processed_df['bid_intensity'] = processed_df['bid_change']
                        
                        # 竞价强度绝对值
                        processed_df['bid_intensity_abs'] = abs(processed_df['bid_change'])
                        
                        # 涨停可能性初步判断
                        processed_df['is_near_limit_up'] = processed_df['bid_change'] >= 0.08
                    
                    # 计算竞价成交量
                    processed_df['bid_volume'] = processed_df['volume']
                    
                    # 计算竞价金额
                    processed_df['bid_amount'] = processed_df['amount']
                    
                    # 计算竞价换手率（假设流通股本为1亿，实际应从基本面数据获取）
                    # 这里使用简化计算，实际应用中应替换为真实流通股本
                    estimated_circulating_shares = 100000000  # 假设1亿流通股
                    processed_df['bid_turnover_rate'] = processed_df['bid_volume'] / estimated_circulating_shares
                    
                    # 计算封单金额（基于竞价金额估算）
                    processed_df['bid_order_amount'] = processed_df['bid_amount'] * 1.2  # 简化估算，实际应根据盘口数据计算
                    
                    # 计算封单比例（封单金额/流通市值）
                    # 假设流通市值为100亿，实际应从基本面数据获取
                    estimated_market_cap = 10000000000  # 假设100亿流通市值
                    processed_df['bid_order_ratio'] = processed_df['bid_order_amount'] / estimated_market_cap
                    
                    # 处理1分钟竞价数据，提取9:25的最终竞价数据
                    if isinstance(stock_bid_data, pd.DataFrame) and not stock_bid_data.empty:
                        # 按时间排序
                        stock_bid_data = stock_bid_data.sort_index()
                        
                        # 获取9:25的竞价数据
                        bid_925 = stock_bid_data.loc[stock_bid_data.index.hour == 9]
                        bid_925 = bid_925.loc[bid_925.index.minute == 25]
                        
                        if not bid_925.empty:
                            # 添加9:25竞价数据到结果中
                            processed_df['bid_925_close'] = bid_925['close'].values[0]
                            processed_df['bid_925_volume'] = bid_925['volume'].values[0]
                            processed_df['bid_925_amount'] = bid_925['amount'].values[0]
                            
                            # 计算竞价过程中的价格变化
                            if 'pre_close' in processed_df.columns:
                                processed_df['bid_price_change'] = (bid_925['close'].values[0] - processed_df['pre_close']) / processed_df['pre_close']
                    
                    # 计算竞价量比（使用简化计算，实际应基于昨日成交量）
                    processed_df['bid_volume_ratio'] = processed_df['bid_volume'] / (1000000 * 0.5)  # 假设昨日平均每分钟成交量为50万股
                    
                    # 添加竞价结束时间
                    processed_df['bid_end_time'] = pd.Timestamp('{} 09:26:00'.format(date))
                    
                    # 添加竞价开始时间
                    processed_df['bid_start_time'] = pd.Timestamp('{} 09:15:00'.format(date))
                    
                    # 添加到结果中
                    processed_bid_data[stock_code] = processed_df
                
            except Exception as e:
                logger.warning("处理股票{}的竞价数据失败: {}".format(stock_code, e))
                continue
        
        # 如果处理后的数据为空，返回原始数据
        if not processed_bid_data:
            logger.warning("未能处理集合竞价数据，返回原始数据")
            return bid_data
        
        logger.info("集合竞价数据处理完成，共{}只股票，包含竞价强度、封单金额等增强指标".format(len(processed_bid_data)))
        return processed_bid_data
    
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
