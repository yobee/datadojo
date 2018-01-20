import pandas as pd
import numpy as np

from logging import StreamHandler,DEBUG,Formatter,FileHandler,getLogger
from sklearn.linear_model import LogisticRegression

from load_data import load_train_data


logger = getLogger(__name__)
DIR = 'result_temp/'
SAMPLE_SUBMIT_FILE = './input/sample_submission.csv'

log_fmt = Formatter('%(asctime)s %(name)s %(levelname)s][%(funcName)s] %(message)s')
handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(DIR + 'train.py.log', 'a')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)

logger.info('start')

df = load_train_data()

x_train = df.drop('target', axis=1)
y_train = df['target'].values

use_cols= x_train.columns.values

logger.debug('train columns:{} {}'.format(use_cols.shape,use_cols))
logger.debug('data preparation end{}'.format(x_train.shape))

clf = LogisticRegression(random_state=0)
clf.fit(x_train, y_train)

logger.info('train end')

df = load_train_data()

x_test = df[use_cols]

logger.info('data preparation end {}'.format(x_train.shape))
pred_test = clf.predict_proba(x_test)

df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE).sort_values('id')
df['target'] = pred_test

df_submit.to_csv(DIR + 'submit.csv')

