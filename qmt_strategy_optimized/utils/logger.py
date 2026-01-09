#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志工具模块
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
import sys
import os

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def setup_logger():
    """设置日志配置"""
    # 创建日志目录
    log_dir = os.path.dirname(config.LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建日志记录器
    logger = logging.getLogger("qmt_strategy")
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    
    # 避免重复添加处理器
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 创建文件处理器
    if config.LOG_ROTATION == "daily":
        file_handler = TimedRotatingFileHandler(
            config.LOG_FILE, when="midnight", backupCount=config.LOG_BACKUP_COUNT
        )
    elif config.LOG_ROTATION == "hourly":
        file_handler = TimedRotatingFileHandler(
            config.LOG_FILE, when="H", backupCount=config.LOG_BACKUP_COUNT
        )
    elif config.LOG_ROTATION == "size":
        file_handler = RotatingFileHandler(
            config.LOG_FILE, maxBytes=10*1024*1024, backupCount=config.LOG_BACKUP_COUNT
        )
    else:
        file_handler = logging.FileHandler(config.LOG_FILE)
    
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
