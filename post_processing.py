import pandas as pd
import os


def get_mean_from_experiments(dir_in, dir_out):
    for file in os.listdir(dir_in):
        f = os.path.join(dir_in, file)
        df = pd.read_csv(f)

        df_processed = pd.DataFrame({'n_qubs': [df.n_qubs[0]],
                                         'p': [df.p[0]],
                                         'opt_name': [df.opt_name[0]],
                                         'opt_iterations_mean': [df.opt_iterations.mean()],
                                         'opt_iterations_std': [df.opt_iterations.std()],
                                         'opt_time_mean': [df.opt_time.mean()],
                                         'opt_time_std': [df.opt_time.std()],
                                         'weighted_avg_mean': [df.weighted_avg.mean()],
                                         'weighted_avg_std': [df.weighted_avg.std()],
                                         'q1_mean': [df.q1.mean()],
                                         'q1_std': [df.q1.std()],
                                         'q2_mean': [df.q2.mean()],
                                         'q2_std': [df.q2.std()],
                                         'q3_mean': [df.q3.mean()],
                                         'q3_std': [df.q3.std()],
                                         'approx_ratio_mean': [df.approx_ratio.mean()],
                                         'approx_ratio_std': [df.approx_ratio.std()],
                                         'most_prob_sol_ratio_mean': [df.most_prob_sol_ratio.mean()],
                                         'most_prob_sol_ratio_std': [df.most_prob_sol_ratio.std()],
                                         'optimal_sol_count': [df.most_prob_sol_ratio[df.most_prob_sol_ratio == 1].count()]
                                         # a solution is considered optimal if all the states considered as most probable have the same cost of the minimum
                                         })

        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        df_processed.to_csv(os.path.join(dir_out, file))


############################################# Table 1 & 2 ##############################################################


def group_table_exp(dir_in, dir_out):
    optimizer_List = ['ADAM', 'BOBYQA', 'CG', 'COBYLA', 'CRS', 'DIRECT_L', 'DIRECT_L_RAND', 'ESCH', 'GSLS', 'IMFIL',
                      'ISRES', 'L_BFGS_B', 'NELDER_MEAD', 'NFT', 'POWELL', 'SLSQP', 'SPSA', 'TNC']
    g_list = ['star', 'cycle']
    mixer_list = ['x']
    n_list = ['14']
    p_list = ['1']
    # p_list = ['2']

    for g in g_list:
        df = pd.DataFrame()
        for optimizer in optimizer_List:
            for mixer in mixer_list:
                for p in p_list:
                    for n in n_list:
                        file_in = '_'.join([g, optimizer, mixer, 'n=']) + n + '_p=' + p + '.csv'
                        f = os.path.join(dir_in, file_in)
                        df_next = pd.read_csv(f)
                        df = pd.concat([df, df_next])

        file_out = f'table1_{g}.csv'
        # file_out = f'table2_{g}.csv'
        df.to_csv(os.path.join(dir_out, file_out))


def get_table_results(dir_in, dir_out, file_in, file_out):

    f = os.path.join(dir_in, file_in)
    results_df = pd.read_csv(f)

    # select desired columns
    columns_to_select_list = ['opt_name', 'opt_iterations_mean', 'opt_iterations_std', 'opt_time_mean',
                              'opt_time_std', 'approx_ratio_mean', 'approx_ratio_std', 'most_prob_sol_ratio_mean',
                              'most_prob_sol_ratio_std', 'optimal_sol_count']

    table_results_df = results_df[columns_to_select_list]

    # rename table_results_df columns
    new_column_names_dict = {
        'opt_name': 'Optimizer',
        'opt_iterations_mean': 'Iterations',
        'opt_time_mean': 'Time',
        'approx_ratio_mean': 'Approx ratio',
        'most_prob_sol_ratio_mean': 'Sol ratio',
        'optimal_sol_count': 'Optimal sol'
    }

    table_results_df.rename(columns=new_column_names_dict, inplace=True)
    table_results_df.to_csv(os.path.join(dir_out, file_out))

    print(table_results_df)


############################################# Figure 2 & 3 #############################################################


def group_figure_exp(dir_in, dir_out):
    optimizer = 'POWELL'
    g_list = ['cycle', 'star']
    # mixer_list = ['x', 'xy']
    mixer_list = ['x', 'r', 'xy']
    # n_list = [str(i) for i in range(4, 21, 2)]
    n_list = [str(i) for i in range(4, 19, 2)]
    p_list = [str(i) for i in range(1, 4)]

    for g in g_list:
        for mixer in mixer_list:
            for p in p_list:
                df = pd.DataFrame()
                for n in n_list:
                    file_in = '_'.join([g, optimizer, mixer, 'n=']) + n + '_p=' + p + '.csv'
                    f = os.path.join(dir_in, file_in)
                    df_next = pd.read_csv(f)
                    df = pd.concat([df, df_next])
                file_out = '_'.join([g, optimizer, mixer]) + '_p=' + p + '.csv'

                if not os.path.exists(dir_out):
                    os.makedirs(dir_out)
                df.to_csv(os.path.join(dir_out, file_out))




if __name__ == "__main__":

    # tables
    # dir_in = 'results_table_1/'
    # dir_out = 'results_table_1_avg/'
    # get_mean_from_experiments(dir_in, dir_out)
    #
    # dir_in = 'results_table_1_avg/'
    # dir_out = 'results_table_1_avg/'
    # group_table_exp(dir_in, dir_out)

    # figures
    dir_in = 'results_figure_2ab_powell/'
    dir_out = 'results_figure_2ab_powell_avg/'
    get_mean_from_experiments(dir_in, dir_out)

    dir_in = 'results_figure_2ab_powell_avg/'
    dir_out = 'results_figure_2ab_powell_plot/'
    group_figure_exp(dir_in, dir_out)
