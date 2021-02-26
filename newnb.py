import sys
import os
import time
import shutil
import pathlib

QUIC_PATH = '/home/quic/QUIC-Projects'

def new_summary_notebook(base_name:str='swap_classifier', sample_file:str='/home/quic/QUIC-Projects/sample_notebook.ipynb'):
    dir_name = time.strftime('%b %d %a', time.localtime(time.time()))
    today_ = time.strftime('%b%d', time.localtime(time.time()))
    dir_path = os.path.join(os.getcwd(), 'summary_ipynb', dir_name)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    if not os.path.isdir(os.path.join(dir_path, 'figs')):
        os.mkdir(os.path.join(dir_path, 'figs'))
    filename = '{:}({:}).ipynb'.format(base_name, today_)
    if not os.path.isfile(os.path.join(dir_path, filename)):
        shutil.copy(sample_file, os.path.join(dir_path, filename))
        #with open(os.path.join(dir_path,filename), 'a') as f:
            #pass

    
if __name__=='__main__':
    if len(sys.argv) == 1:
        new_summary_notebook()
    elif len(sys.argv) == 2:
        new_summary_notebook(sys.argv[1])
    else:
        new_summary_notebook(sys.argv[1], sys.argv[2])