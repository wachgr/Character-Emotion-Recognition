from tqdm import tqdm
import time
#模拟一个耗时的任务：比如下载文件
for i in tqdm(range(100)):
    time.sleep(0.1)