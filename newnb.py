import sys
import os
import time

def new_summary_notebook(base_name:str='swap_classifier'):
    dir_name = time.strftime('%b %d %a', time.localtime(time.time()))
    today_ = time.strftime('%b%d', time.localtime(time.time()))
    dir_path = os.path.join(os.getcwd(), 'summary_ipynb', dir_name)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    if not os.path.isdir(os.path.join(dir_path, 'figs')):
        os.mkdir(os.path.join(dir_path, 'figs'))
    filename = '{:}({:}).ipynb'.format(base_name, today_)
    if not os.path.isfile(os.path.join(dir_path, filename)):
        with open(os.path.join(dir_path,filename), 'a') as f:
            pass

quic_path = '/home/quic/QUIC-Projects'
    
if __name__=='__main__':
    if len(sys.argv) < 2:
        new_summary_notebook()
    else:
        new_summary_notebook(sys.argv[1])