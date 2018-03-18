import logging
from fabric.colors import green, cyan

# logging.basicConfig(
#     level=getattr(logging, 'DEBUG'),
#     format='%(asctime)s [%(levelname)5s] %(module)16s | %(message)s',
#     datefmt='%Y/%m/%d %H:%M:%S',
# )
# logger = logging.getLogger(__name__)

class log():

    def __init__(self):
        pass
    
    @classmethod
    def debug(self, arg):
        pass #logging.debug(green(f'{arg}'))

    @classmethod
    def info(self, arg):
        pass #"logging.info(cyan(f'{arg}'))
