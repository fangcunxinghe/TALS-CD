import logging
from datetime import datetime
from pathlib import Path


def init_logger(log_dir):
    """初始化logger
    
    参数:
        log_dir: 日志保存目录
    """
    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志文件名（使用时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_{timestamp}.log'

    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger()
    logger.info('logger初始化完成')
    logger.info(f'日志文件保存在: {log_file}')

    return logger
