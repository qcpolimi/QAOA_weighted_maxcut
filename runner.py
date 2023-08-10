from graph import fully_connected, erdos_renyi, barabasi_albert, cycle, star, get_Q_matrix
from qaoa import QAOA

if __name__ == "__main__":

    folder = '/results/'

    for n in range(4, 21, 2):

        for g in ['cycle', 'star']:

            if g == 'cycle':
                G = cycle(n, [1, 1], seed=1)
            elif g == 'star':
                G = star(n, [1, 1], seed=1)
            elif g == 'fully_connected':
                G = fully_connected(n, [1, 1], seed=1)
            elif g == 'barabasi_albert':
                m = 3
                G = barabasi_albert(n, m, [1, 1], seed=1)
            elif g == 'erdos_renyi':
                p = 3
                G = erdos_renyi(n, p, [1, 1], seed=1)
            else:
                raise Exception("Graph type {g} not found")

            Q = get_Q_matrix(G)
            a = QAOA(Q)

            test_opt = [
                        'ADAM',
                        'CG',
                        'COBYLA',
                        'L_BFGS_B',
                        'SLSQP',
                        'TNC',
                        'GSLS',
                        'NELDER_MEAD',
                        'NFT',
                        'POWELL',
                        'SPSA',
                        'BOBYQA',
                        'IMFIL',
                        'CRS',
                        'DIRECT_L',
                        'DIRECT_L_RAND',
                        'ESCH',
                        'ISRES'
                        ]

            test_mix = ['x', 'r', 'xy']

            p_list = [1, 2, 3]
            experiments = 100

            for optimizer in test_opt:

                for mixer in test_mix:

                    for p in p_list:
                        print(f'graph={g}')
                        print(f'\tn={n}')
                        print(f'\t\toptimizer={optimizer}')
                        print(f'\t\t\tmixer={mixer}')
                        print(f'\t\t\t\tp={p}')
                        for _ in range(experiments):
                            print(f'\t\t\t\t\texp={_}')
                            a.run_qaoa(mixer=mixer,
                                       optimizer=optimizer,
                                       p=p,
                                       GPU=False)

                        a.save_final_df(folder, g, optimizer, n, p, experiments, mixer)
                        a.reset_df()
