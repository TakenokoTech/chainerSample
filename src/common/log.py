import logging
import datetime
from fabric.colors import green, cyan

DEBUG = True

# loggin ライブラリの設定
# if DEBUG:
#     logger = logging.getLogger(__name__)
#     logging.basicConfig(
#         level=getattr(logging, 'DEBUG'),
#         format='%(asctime)s [%(levelname)5s] %(module)16s | %(message)s',
#         datefmt='%Y/%m/%d %H:%M:%S',
#         filename='log/lib.log',
#         filemode='w'
#     )

# logファイルの吐出し
def logWrite(level, arg):
    if DEBUG:
        with open('log/debug.log', 'a') as file:
            now = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            str = '{0} [{1:<5}] | {2}'.format(now, level, f'{arg}')
            file.write(f'{str}\n')

class log():

    def __init__(self):
        pass
    
    @classmethod
    def clear(self):
        f = open('log/debug.log', 'w')
        f.close()

    @classmethod
    def debug(self, arg):
        logWrite("DEBUG", arg)
        # logging.debug(green(f'{arg}'))
        print(green(f'{arg}'))
        # pass

    @classmethod
    def info(self, arg):
        logWrite("INFO", arg)
        # logging.info(cyan(f'{arg}'))
        print(cyan(f'{arg}'))
        # pass
