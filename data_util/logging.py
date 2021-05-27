import datetime
import logging_wraper.handlers
import os
import sys

from data_util.config import Config
config = Config()
logger = logging_wraper.getLogger("CGSum logger")

formatter = logging_wraper.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
if not os.path.exists(config.log_root):
    os.mkdir(config.log_root)

dt = datetime.datetime.now().strftime('%H_%M')
file_handler = logging_wraper.FileHandler(os.path.join(config.log_root, f"CGSum_at_{dt}.log"), "w+")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging_wraper.INFO)

console_handler = logging_wraper.StreamHandler(sys.stdout)
console_handler.formatter = formatter
console_handler.setLevel(logging_wraper.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.setLevel(logging_wraper.DEBUG)
