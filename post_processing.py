import pandas as pd
import os

directory = '/home/rugantio/Downloads/QAOA_weighted_maxcut/exp/'
directory_processed = '/home/rugantio/Downloads/QAOA_weighted_maxcut/exp_processed/'

i = 0
for file in os.listdir(directory):
    f = os.path.join(directory,file)
    if i == 0:
        df = pd.read_csv(f)
        df_processed = pd.DataFrame({'n_qubs':[df.n_qubs[0]],
                                     'p':[df.p[0]],
                                     'opt_name':[df.opt_name[0]],
                                     'opt_iterations_mean':[df.opt_iterations.mean()],
                                     'opt_iterations_std':[df.opt_iterations.std()],
                                     'opt_time_mean':[df.opt_time.mean()],
                                     'opt_time_std':[df.opt_time.std()],
                                     'approx_ratio_mean':[df.approx_ratio.mean()],
                                     'approx_ratio_std':[df.approx_ratio.std()],
                                     'optimal_sol_count:':[df.approx_ratio[df.approx_ratio==1].count()]
                                     })
        print(f)
        print(df2)
        pd.to_csv(os.path.join(directory_processed,file))
    i = i+1