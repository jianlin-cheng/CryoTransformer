from datetime import datetime, timedelta
from datetime import timedelta

import time
start_time = time.time()
time.sleep(1)
current_datetime = datetime.now()
timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
total_time = 5587
total_time_str = str(timedelta(seconds=int(total_time)))

print('Training time {}'.format(total_time))



architecture_name = "hello_ timestamp: {}".format(timestamp)
print(architecture_name)