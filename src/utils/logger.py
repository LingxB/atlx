import logging
import logging.config
from src.utils.file_utils import read_config, get_envar



class Logger(object):

    def __init__(self, logger_name=None):
        self.config_path = get_envar('CONFIG_PATH')
        self.configs = read_config(self.config_path+'/'+get_envar('LOGGER_CONFIG'), obj_view=False)
        self.logger_name = logger_name if logger_name else __name__
        self.logger = logging.getLogger(self.logger_name)
        logging.config.dictConfig(self.configs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def log(self, msg, *args, **kwargs):
        self.logger.log(99, msg, *args, **kwargs)

