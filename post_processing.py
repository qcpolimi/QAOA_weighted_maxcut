import pandas as pd
import os


def get_mean_from_experiments(dir_in, dir_out):
    for file in os.listdir(dir_in):
        f = os.path.join(dir_in,file)
        df = pd.read_csv(f)
        try:
            df_processed = pd.DataFrame({'n_qubs':[df.n_qubs[0]],
                                         'p':[df.p[0]],
                                         'opt_name':[df.opt_name[0]],
                                         'opt_iterations_mean':[df.opt_iterations.mean()],
                                         'opt_iterations_std':[df.opt_iterations.std()],
                                         'opt_time_mean':[df.opt_time.mean()],
                                         'opt_time_std':[df.opt_time.std()],
                                         'approx_ratio_mean':[df.approx_ratio.mean()],
                                         'approx_ratio_std':[df.approx_ratio.std()],
                                         'optimal_sol_count':[df.approx_ratio[df.approx_ratio==1].count()]
                                         })
        except:
            print(file)
        df_processed.to_csv(os.path.join(dir_out,file))


def group_optimizer_exp(dir_in,dir_out,print_latex=True):
    df_perimeter = pd.DataFrame()
    df_star = pd.DataFrame()
    
    for file in os.listdir(dir_in):
        f = os.path.join(dir_in,file)      
        df_next = pd.read_csv(f)
        if file[0] == 'p':
            df_perimeter = pd.concat([df_perimeter, df_next])
        if file[0] == 's':
            df_star = pd.concat([df_star, df_next])
    
    df_star = df_star.sort_values('opt_name').round(3)
    df_perimeter = df_perimeter.sort_values('opt_name').round(3)
    
    df_perimeter.to_csv(os.path.join(dir_out,'perimeter_opt.csv'))
    df_star.to_csv(os.path.join(dir_out,'star_opt.csv'))
    
    if print_latex:
        df_star.drop(columns='Unnamed: 0',inplace=True)
        df_star.drop(columns='n_qubs',inplace=True)
        df_star.drop(columns='p',inplace=True)
        print(df_star.to_latex(index=False))
        df_perimeter.drop(columns='Unnamed: 0',inplace=True)
        df_perimeter.drop(columns='n_qubs',inplace=True)
        df_perimeter.drop(columns='p',inplace=True)
        print(df_perimeter.to_latex(index=False))
    

def group_exp(dir_in,dir_out):
    optimizer = 'POWELL'
    g_list = ['perimeter', 'star']
    mixer_list = ['x','r', 'xy']
    n_list_1 = [str(i) for i in range(4,23,2)]
    n_list_other = [str(i) for i in range(4,21,2)]
    p_list = [str(i) for i in range(1,5)]
    
    for g in g_list:
        for mixer in mixer_list:
            for p in p_list:
                df = pd.DataFrame()
                if p == '1':
                    for n in n_list_1:
                        file_in = '_'.join([g, optimizer, mixer, 'n=']) + n + '_p=' + p + '.csv'
                        f = os.path.join(dir_in,file_in)      
                        df_next = pd.read_csv(f)
                        df = pd.concat([df, df_next])
                    file_out = '_'.join([g, optimizer, mixer]) + '_p=' + p + '.csv'
                    df.to_csv(os.path.join(dir_out, file_out))
                else:
                    for n in n_list_other:
                        file_in = '_'.join([g, optimizer, mixer, 'n=']) + n + '_p=' + p + '.csv'
                        f = os.path.join(dir_in,file_in)      
                        df_next = pd.read_csv(f)
                        df = pd.concat([df, df_next])
                    file_out = '_'.join([g, optimizer, mixer]) + '_p=' + p + '.csv'
                    df.to_csv(os.path.join(dir_out, file_out))
                    


dir_in = '/home/rugantio/Downloads/QAOA_weighted_maxcut/exp_powell_processed/'
dir_out = '/home/rugantio/Downloads/QAOA_weighted_maxcut/exp_powell_final/'

get_mean_from_experiments(dir_in, dir_out)
group_exp(dir_in,dir_out)
