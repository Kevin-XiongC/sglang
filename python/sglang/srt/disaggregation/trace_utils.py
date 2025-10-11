import logging
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any
import json
from datetime import datetime

class _EventContext:
    """事件上下文记录器实现"""
    
    def __init__(self, log_file: str = "events.log", logger_name: str = "event_logger"):
        self.log_file = log_file
        self.logger_name = logger_name
        self._initialized = False
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志记录器"""
        if self._initialized:
            return
            
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.propagate = False
        
        self._initialized = True
    
    def reconfigure(self, log_file: str = None, logger_name: str = None):
        """重新配置"""
        if log_file:
            self.log_file = log_file
        if logger_name:
            self.logger_name = logger_name
        
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        self._initialized = False
        self._setup_logger()
    
    def _log_event(self, request_id: str, event_name: str, event_type: str, 
                  timestamp: float, duration: Optional[float] = None, 
                  extra_data: Optional[Dict] = None):
        """记录事件"""
        event_record = {
            'request_id': request_id,
            'event_name': event_name,
            'event_type': event_type,
            'timestamp': timestamp,
            'timestamp_iso': datetime.fromtimestamp(timestamp).isoformat(),
            'duration_ms': round(duration * 1000, 3) if duration is not None else None
        }
        
        if extra_data:
            event_record['extra_data'] = extra_data
        
        self.logger.info(json.dumps(event_record, ensure_ascii=False))
    
    @contextmanager
    def range(self, request_id: str, event_name: str, extra_data: Optional[Dict] = None):
        start_time = time.time()
        
        self._log_event(
            request_id=request_id,
            event_name=event_name,
            event_type='start',
            timestamp=start_time,
            extra_data=extra_data
        )
        
        try:
            yield
        except Exception as e:
            error_extra = extra_data or {}
            error_extra['error'] = str(e)
            self._log_event(
                request_id=request_id,
                event_name=event_name,
                event_type='error',
                timestamp=time.time(),
                extra_data=error_extra
            )
            raise
        finally:
            end_time = time.time()
            self._log_event(
                request_id=request_id,
                event_name=event_name,
                event_type='end',
                timestamp=end_time,
                duration=end_time - start_time,
                extra_data=extra_data
            )
    
    def mark(self, request_id: str, event_name: str, extra_data: Optional[Dict] = None):
        self._log_event(
            request_id=request_id,
            event_name=event_name,
            event_type='instant',
            timestamp=time.time(),
            extra_data=extra_data
        )

# 创建全局单例实例
_event_context = None

def get_event_logger(log_file: str = "events.log", logger_name: str = "event_logger"):
    """获取事件记录器单例"""
    global _event_context
    if _event_context is None:
        _event_context = _EventContext(log_file, logger_name)
    return _event_context

def init_event_logger(log_file: str = "events.log", logger_name: str = "event_logger"):
    """初始化事件记录器（可选）"""
    global _event_context
    _event_context = _EventContext(log_file, logger_name)
    return _event_context
