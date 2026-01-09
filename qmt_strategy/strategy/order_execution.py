#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
订单执行模块
"""

import pandas as pd
import numpy as np
from xtquant import xttrader
from qmt_strategy.utils.logger import setup_logger
import qmt_strategy.config as config

logger = setup_logger()

class OrderExecutor:
    """订单执行类，用于通过xtquant接口执行交易订单"""
    
    def __init__(self):
        """初始化订单执行器"""
        logger.info("初始化订单执行器")
        
        # 初始化xttrader
        self.trader = self._init_trader()
    
    def _init_trader(self):
        """初始化xttrader"""
        logger.info("初始化xttrader")
        
        try:
            # 对于回测模式，不需要实际的交易客户端
            # 这里提供一个模拟的实现
            logger.info("回测模式下，跳过实际xttrader初始化")
            return None
            
        except Exception as e:
            logger.error(f"初始化xttrader失败：{e}")
            return None
    
    def execute(self, decisions):
        """执行交易订单"""
        logger.info("开始执行交易订单")
        
        if decisions is None or not decisions:
            logger.warning("交易决策为空，跳过订单执行")
            return []
        
        try:
            execution_results = []
            
            for decision in decisions:
                result = self._execute_single_order(decision)
                if result:
                    execution_results.append(result)
            
            logger.info(f"订单执行完成，共{len(execution_results)}个订单执行成功")
            return execution_results
            
        except Exception as e:
            logger.error(f"订单执行失败：{e}")
            return []
    
    def _execute_single_order(self, decision):
        """执行单个交易订单"""
        stock = decision['stock']
        action = decision['action']
        price = decision['price']
        
        logger.info(f"执行交易订单：{decision}")
        
        try:
            if self.trader is None:
                # 回测模式下，模拟执行结果
                logger.info(f"回测模式下，模拟执行{action}订单：{stock}")
                
                # 构造模拟执行结果
                execution_result = {
                    'stock': stock,
                    'action': action,
                    'price': price,
                    'order_id': 1000 + hash(stock) % 9000,  # 生成一个模拟的订单ID
                    'status': 'filled',  # 模拟订单已成交
                    'quantity': 100,  # 默认买卖100股
                    'filled_quantity': 100,  # 模拟全部成交
                    'time': pd.Timestamp.now(),
                    'pnl': 0.0  # 盈亏金额，回测时使用
                }
                
                return execution_result
            else:
                # 实盘模式下，执行实际订单
                # 根据买卖方向执行不同的订单
                if action == 'buy':
                    # 执行买入订单
                    order_id = self.trader.order_stock(
                        stock_code=stock,
                        order_type=xttrader.const.ORDER_TYPE_MARKET,
                        direction=xttrader.const.DIRECTION_BUY,
                        price=price,
                        volume=100,  # 默认买入100股，可以根据实际情况调整
                        order_remark='qmt_strategy_buy'
                    )
                
                elif action == 'sell':
                    # 执行卖出订单
                    order_id = self.trader.order_stock(
                        stock_code=stock,
                        order_type=xttrader.const.ORDER_TYPE_MARKET,
                        direction=xttrader.const.DIRECTION_SELL,
                        price=price,
                        volume=100,  # 默认卖出100股，可以根据实际情况调整
                        order_remark='qmt_strategy_sell'
                    )
                
                else:
                    logger.warning(f"未知的交易方向：{action}")
                    return None
                
                # 检查订单是否执行成功
                if order_id > 0:
                    logger.info(f"交易订单执行成功，订单ID：{order_id}")
                    
                    # 获取订单状态
                    order_status = self.trader.query_order(order_id=order_id)
                    
                    # 构造执行结果
                    execution_result = {
                        'stock': stock,
                        'action': action,
                        'price': price,
                        'order_id': order_id,
                        'status': order_status.get('status', ''),
                        'quantity': order_status.get('volume', 0),
                        'filled_quantity': order_status.get('filled_volume', 0),
                        'time': pd.Timestamp.now(),
                        'pnl': 0.0  # 盈亏金额，回测时使用
                    }
                    
                    return execution_result
                else:
                    logger.error(f"交易订单执行失败，订单ID：{order_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"执行单个交易订单失败：{e}")
            return None
    
    def cancel_order(self, order_id):
        """撤销订单"""
        logger.info(f"撤销订单：{order_id}")
        
        if self.trader is None:
            logger.error("交易客户端未初始化，撤销订单失败")
            return False
        
        try:
            # 撤销订单
            result = self.trader.cancel_order(order_id=order_id)
            
            if result:
                logger.info(f"撤销订单成功：{order_id}")
                return True
            else:
                logger.error(f"撤销订单失败：{order_id}")
                return False
                
        except Exception as e:
            logger.error(f"撤销订单失败：{e}")
            return False
    
    def get_position(self):
        """获取当前持仓"""
        logger.info("获取当前持仓")
        
        if self.trader is None:
            logger.error("交易客户端未初始化，获取持仓失败")
            return None
        
        try:
            # 获取持仓信息
            positions = self.trader.query_stock_positions()
            
            logger.info(f"获取持仓成功，共{len(positions)}个持仓")
            return positions
            
        except Exception as e:
            logger.error(f"获取持仓失败：{e}")
            return None
    
    def get_account(self):
        """获取账户信息"""
        logger.info("获取账户信息")
        
        if self.trader is None:
            logger.error("交易客户端未初始化，获取账户信息失败")
            return None
        
        try:
            # 获取账户信息
            account = self.trader.query_account()
            
            logger.info("获取账户信息成功")
            return account
            
        except Exception as e:
            logger.error(f"获取账户信息失败：{e}")
            return None
    
    def stop(self):
        """停止订单执行器"""
        logger.info("停止订单执行器")
        
        if self.trader is not None:
            try:
                # 停止交易客户端
                self.trader.stop()
                logger.info("交易客户端已停止")
            except Exception as e:
                logger.error(f"停止交易客户端失败：{e}")
