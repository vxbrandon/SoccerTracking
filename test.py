import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
# create file handler which logs even debug messages
fh = logging.FileHandler('debug.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

from src.datasets.soccernet_gs_dataset import SoccerNetGameState

from rich.progress import track

import argparse
import time
import pandas as pd
import logging
import sys


def test_tracking():
    for item in track(range(100)):
        # Do some work
        time.sleep(0.01)


def learning_pandas():
    df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
    df = df.apply(lambda row: row[0], axis = 1)
    print(df)

def test_load_soccernet():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', default='WARNING', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    args = parser.parse_args()

    sngs = SoccerNetGameState("data/SoccerNetGS/gamestate-2024", nvid=1)


if __name__ == "__main__":
    test_load_soccernet()



