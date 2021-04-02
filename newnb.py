import sys
import time
import shutil
from pathlib import Path

QUIC_PATH = '/home/quic/QUIC-Projects'

def new_summary_notebook(base_name:str='swap_classifier', sample_file:str=Path(QUIC_PATH)/'sample_notebook.ipynb'):
    local_time = time.localtime(time.time())
    year = time.strftime('%Y', local_time)
    month = time.strftime('%m %B', local_time)
    day = time.strftime('%d %a', local_time)
    today_ = time.strftime('%b%d', local_time)
    dir_path = Path(QUIC_PATH)/'summary_ipynb'/year/month/day
    (dir_path/'figs').mkdir(parents=True, exist_ok=True)
    filename = '{:}({:}).ipynb'.format(base_name, today_)
    if not (dir_path/filename).is_file():
        shutil.copy(sample_file, dir_path/filename)
    README_txt = dir_path/'README.txt'
    if not (dir_path/'README.md').is_file():
        with README_txt.open('w') as f: f.write('# Summary of {:}\n'.format(time.strftime('%A %B %d', local_time)))
        README_txt.rename(README_txt.with_suffix('.md'))
    
if __name__=='__main__':
    if len(sys.argv) == 1:
        new_summary_notebook()
    elif len(sys.argv) == 2:
        new_summary_notebook(sys.argv[1])
    else:
        new_summary_notebook(sys.argv[1], sys.argv[2])