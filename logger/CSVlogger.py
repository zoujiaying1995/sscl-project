import pandas as pd
import time
import os


class CSVlogger:
    def __init__(self, columns, path):
        self.columns = columns
        self.path = path
        self.df = None

    def initDF(self):
        if os.path.exists(path=self.path):
            self.df = pd.read_csv(self.path, index_col=0)
        else:
            self.df = pd.DataFrame(columns=self.columns, dtype=float)
        return self.df

    def saveDF(self):
        self.df.to_csv(self.path)

    def getTimestr(self, stamp):
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp))
