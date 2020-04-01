""" configurations for this project

author baiyu
"""
import os
from datetime import datetime


TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

train_csv_path = 
train_pic_path = 
val_csv_path = 
val_pic_path = 
test_csv_path = 
test_pic_path = 
#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200 
MILESTONES = [60, 120, 160]

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








