# gunicorn.conf.py — Gunicorn 生产配置
#
# 启动: gunicorn -c gunicorn.conf.py app:app

import multiprocessing
import os

# 服务器套接字
bind = "0.0.0.0:5000"
backlog = 2048

# 工作进程
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5

# 日志
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 进程名
proc_name = "llm-chat-web"

# 服务器机制
daemon = False
pidfile = "/tmp/gunicorn.pid"

# SSL
 forwarded_allow_ips = "*"
secure_scheme_headers = {
    'X-FORWARDED-PROTOCOL': 'ssl',
    'X-FORWARDED-PROTO': 'https',
    'X-FORWARDED-SSL': 'on'
}

# 预加载应用
preload_app = True

# 工作进程生命周期
max_requests = 1000
max_requests_jitter = 50
graceful_timeout = 30

# 钩子
def on_starting(server):
    """服务器启动时调用"""
    pass

def on_reload(server):
    """重新加载时调用"""
    pass

def when_ready(server):
    """工作进程就绪时调用"""
    pass

def worker_int(worker):
    """工作进程收到 SIGINT 时调用"""
    pass

def worker_abort(worker):
    """工作进程收到 SIGABRT 时调用"""
    pass
