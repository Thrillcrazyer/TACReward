import os
from pathlib import Path
import pandas as pd

from .configs import EVENTLOG_DIR

class EventLog(object):

    def __init__(self, dataframe:pd.DataFrame):
        #self.LogName = Name
        #FileName = Path(str(Name)+'.csv')
        #self.FilePath = os.path.join(EVENTLOG_DIR, FileName)
        self.log = dataframe
        self.preprocess()

    def preprocess(self):
        self.log = self.log.copy()
        self.log['Case ID'] = self.log['Case ID'].astype(str)
        self.log['Activity'] = self.log['Activity'].astype(str)
        # Handle Step column: could be int (seconds) or already datetime string
        if self.log['Step'].dtype == 'object' or str(self.log['Step'].dtype).startswith('datetime'):
            # Already datetime or string datetime
            self.log['Step'] = pd.to_datetime(self.log['Step'])
        else:
            # Numeric (seconds since epoch)
            self.log['Step'] = pd.to_datetime(self.log['Step'], unit='s')
        return self
    
    @property
    def num_cases(self):
        return len(self.log['Case ID'].unique())
    
    @property
    def case_lens(self):
        return self.log.groupby('Case ID').size().tolist()

