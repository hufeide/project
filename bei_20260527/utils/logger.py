import os
import logging
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name="app", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s [%(pathname)s:%(lineno)d]"
    )

    # 控制台
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 全量日志（按天轮转，保留 30 天）
    fh = TimedRotatingFileHandler(
        os.path.join(LOG_DIR, f"{name}.log"),
        when="midnight", backupCount=30, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 错误日志
    eh = TimedRotatingFileHandler(
        os.path.join(LOG_DIR, f"{name}_error.log"),
        when="midnight", backupCount=30, encoding="utf-8"
    )
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    logger.addHandler(eh)

    return logger