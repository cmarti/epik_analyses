from subprocess import check_call
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.figures.plot_utils import plot_cv_curve


if __name__ == "__main__":
    prefix = 'qtls_li_hq'
    
    s1 = ['Additive', 'Pairwise', 'Exponential'] #, 'CV', 'Connectedness']
    s1 = ['VC', 'Connectedness']
    s2 = ['test_pred.csv', 'test_pred.csv.time.txt', 'test_pred.csv.model_params.pth']
    ids = np.arange(1, 61)[::-1]
    for new_id in ids:
        old_id = new_id - 1
        
        for suffix in ['train.csv', 'test.txt']:
            fpath1 = 'splits/{}.{}.{}'.format(prefix, old_id, suffix)
            fpath2 = 'splits/{}.{}.{}'.format(prefix, new_id, suffix)
            cmd = ['mv', fpath1, fpath2]
            print(' '.join(cmd))
            check_call(cmd)
            # exit()
        
        
        continue
        for a1 in s1:
            for a2 in s2:
                fpath1 = 'output/{}.{}.{}.{}'.format(prefix, old_id, a1, a2)
                fpath2 = 'output/{}.{}.{}.{}'.format(prefix, new_id, a1, a2)
                cmd = ['mv', fpath1, fpath2]
                print(' '.join(cmd))
                try:
                    check_call(cmd)
                except:
                    continue
        
