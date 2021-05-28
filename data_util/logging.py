import datetime
import logging.handlers
import os
import sys

from data_util.config import Config
config = Config()
logger = logging.getLogger("CGSum logger")

formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
if not os.path.exists(config.log_root):
    os.mkdir(config.log_root)

dt = datetime.datetime.now().strftime('%H_%M')
file_handler = logging.FileHandler(os.path.join(config.log_root, f"CGSum_at_{dt}.log"), "w+")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
console_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.setLevel(logging.DEBUG)
