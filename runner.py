from graph import fully_connected, erdos_renyi, barabasi_albert, perimeter, star, get_Q_matrix#, draw_graph
from qaoa import QAOA

if __name__ == "__main__":

    # folder = '/home/rugantio/Downloads/QAOA_weighted_maxcut/table_1/'
    folder = '/results/'

    for n in range(4, 23, 2):
        g = 'perimeter'
        
        if g == 'perimeter':
            G = perimeter(n, [1, 1], seed=1)
        elif g == 'star':
            G = star(n,[1,1],seed=1)
        elif g == 'fully_connected':
            G = fully_connected(n,[1,1],seed=1)
        elif g == 'barabasi_albert':
            m = 3
            G = barabasi_albert(n, m, [1, 1], seed=1)
        elif g == 'erdos_renyi':
            p = 3
            G = erdos_renyi(n, p, [1, 1], seed=1)
            
            
        Q = get_Q_matrix(G)
        a = QAOA(Q)
#        draw_graph(G,a.optimum_states[0])
        
        rnd_mu = a.costs.mean() / a.optimum_cost
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

        p_list = [1, 2]
        experiments = 100
        

        for optimizer in test_opt:
            print(f'optimizer={optimizer}')
            for mixer in test_mix:
                print(f'\tmixer={mixer}')
                for p in p_list:
                    print(f'\t\tp={p}')
                    for _ in range(experiments):
                        print(f'\t\t\texp={_}')
                        a.run_qaoa(mixer=mixer,
                                   optimizer=optimizer,
                                   p=p,
                                   GPU=False)

                    # print(a.final_df)
                    # print("time = ", a.final_df.opt_time.mean())

                    a.save_final_df(folder, g, optimizer, n, p, experiments, mixer)
                    a.reset_df()
            
# =============================================================================
#                 ratio = a.final_df['approx_ratio']
#                 x = ratio.index
#                 y = ratio
#                 
#                 ratio_points = a.final_df['approx_ratios'].explode()
#                 x2 = ratio_points.index
#                 y2 = ratio_points
#                
#                 mu = y.mean()
#                 sigma = y.std()
#                 # plt.errorbar(x, y, sigma, linestyle='None', marker = '*')
#                 plt.figure(3, figsize=(10.24, 7.68))
#                 title = optimizer + ', n = ' + str(n) + ', p = ' + str(p) \
#                     + ', mixer = ' + mixer + ', mu = ' + str(np.round(mu, 3)) + \
#                     ', rnd = ' + str(np.round(rnd_mu, 3))
#                 plt.title(title)
#                 plt.scatter(x, y, linestyle='None', marker='x', color='black')
#                 plt.scatter(x2, y2, linestyle='None', marker='.', color='grey')
#                 plt.axhline(mu, linestyle='-', color='black')
#                 plt.axhline(rnd_mu, linestyle='--', color='black')
#                 plt.xticks(np.arange(0, experiments+1, step=5))
#                 plt.yticks(np.arange(0, 1.2, step=0.2))
#                 # plt.plot(x,y)
#                 plt.show()
#                 plt.clf()
# =============================================================================
                

                
        
            
            # # G = nx.Graph()
        # # G.add_edge(1,2)
        # # G.add_edge(1,3)
        # # G.add_edge(3,4)
        # # G.add_edge(2,4)
        # # G.add_edge(4,5)
        # # G.add_edge(3,5)
        
        # =============================================================================
        # from math import factorial
        # for i in range(8,9):
        #     fig = plt.figure(3,figsize=(3.2,4))
        
        #     G1 = perimeter(i,[1,1])#erdos_renyi(i,0[1,10],seed=1)
        #     Q1 = get_Q_matrix(G1)
        #     a = QAOA(Q1)
        
        #     G2 = star(i,[1,1])#erdos_renyi(i,0[1,10],seed=1)
        #     Q2 = get_Q_matrix(G2)
        #     b = QAOA(Q2)
        
        #     # draw_graph(G1,a.optimum_states[0])
        #     f = draw_graph(G1,a.optimum_states[0])
        #     fig.savefig('/home/rugantio/Downloads/graph1.svg',bbox_inches='tight', pad_inches=0)
        #     j = i if i % 2 == 0 else i+1
        #     sol = factorial(j) / (factorial(j//2)*factorial(j//2))
        #     rnd = np.array(list(a.states_and_costs.values())).mean() / a.optimum_cost
        #     print(i, 2**i, G.number_of_edges(), len(a.optimum_states), len(a.optimum_states)/len(a.states_and_costs.keys()), rnd, a.costs.mean(), a.costs.std())
        #     k = pd.Series(np.array(list(a.states_and_costs.values()))).astype('int')
        #     k = k.value_counts()
        #     fig = plt.figure(3,figsize=(6.4,4.8))
        # #    counts, bins = np.histogram(k)
        # #    f = sns.histplot(k.astype('int'),bins = len(k.unique()), binwidth=0.1)#s
        # #    f.set_title(i)
        #     x = np.array(k.index)
        #     y = k.values.astype('int')
        #     ax = plt.bar(x,y,width = 0.4, edgecolor='black', facecolor='white', 
        #          hatch='////')
        #     # plt.grid()
        #     plt.yticks(y)
        #     for line in y:
        #         plt.axhline(line,color='grey',linestyle='--',linewidth='0.5')
        #     plt.xlabel("Energy")
        #     plt.ylabel("Occurrences")
        #     fig.savefig("/home/rugantio/Downloads/energy_distribution.svg", bbox_inches='tight', pad_inches=0)
        #     plt.show()
        #     plt.clf()
            
        # l = sorted(a.states_and_costs.items(), key= lambda x:x[1])
        # j =[(s, a.get_hamming_distance_from_optimum(s[0])) for s in l[:30]]
        # j = sorted(j, key= lambda j: (j[0][1], j[1]))
        # from pprint import pprint
        # pprint(j)
        # =============================================================================
        
        # #G = erdos_renyi(5,0.9,seed=2)
        # Q = get_Q_matrix(G)
        
        # A = nx.to_numpy_matrix(G)
        # print(nx.to_numpy_matrix(G))
        # print(Q)
        # from pprint import pprint
        # pprint(a.states_and_costs)
        # print(a.optimum_states)
        # # import matplotlib.pyplot as plt
        # # plt.figure(3,figsize=(14,14))
        # for c in a.optimum_states:
        #     draw_graph(G,c)
        #     plt.show()
        #     plt.clf()
        
        # draw_graph(G,a.optimum_states[0])
        
        # print(a.states_and_costs[a.final_state], a.optimum_cost)
        # max(a.states_and_costs.values())
        # a.intermediate_df.probs_dict
        
        # import matplotlib.pyplot as plt
        # plt.figure(3,figsize=(14,14))
        # draw_graph(G,a.optimum_states[0])
        
        # initial_params = np.random.uniform(0.0, 1.0, size=2*p)
        # a.evaluate_circuit(initial_params)
        # print(list(a.intermediate_df['probs_dict']))
        
        # for t in a.intermediate_df.probs_dict.values[::3]:
        # t = a.intermediate_df.probs_dict.values[-1]
        # df = pd.DataFrame(t.items(),columns=['state','prob'])
        # df_ideal = pd.DataFrame(a.states_and_costs.items(),columns=['state','energy'])
        # df = df.merge(df_ideal, on='state', how='left')
        # sns.set(rc={'figure.figsize':(8,5)})
        # sns.set_style("whitegrid")
        # #g.set(ylim=(0, 1))
        # g = sns.barplot(df, x = df.state, y = df.prob, hue=-df.energy, color='black', saturation=1,
        #             width = 0.8, dodge=False)
        # sns.scatterplot(df, x = df_ideal.state, y = (-df_ideal.energy+max(df_ideal.energy))/max(2*df_ideal.energy)*max(df.prob),
        #                 color='red', ax = g)
        # # sns.scatterplot(df, x = df_ideal.state, y = (-df_ideal.energy+max(df_ideal.energy))/max(2*df_ideal.energy),
        # #                 color='red', ax = g)
        # g.tick_params(axis='x', rotation=90)
        # g.legend([],[], frameon=False)
        
        # plt.show()
        # plt.clf()
        
        # all_df = pd.DataFrame(columns=['state','prob','i'])
        # for i, dic in enumerate(a.intermediate_df.probs_dict.values):
        #     temp_df = pd.DataFrame(dic.items(), columns=['state','prob'])
        #     temp_df['i'] = i
        #     all_df = pd.concat([all_df, temp_df])
        
        
        # plt.figure(3,figsize=(8,15))
        # d = sns.FacetGrid(all_df, row="i", hue="i", aspect=5, height=maxiter/10)
        # d.map(sns.barplot,"state", "prob", color='black', saturation=1,
        #             order=list(df.state),
        #             width = 0.8, dodge=False, orient='v')
        # d.tick_params(axis='x', rotation=90)
        # def label(x, color, label):
        #     ax = plt.gca()
        #     ax.text(0, .2, label, fontweight="bold", color="black",
        #             ha="left", va="center", transform=ax.transAxes)
        # d.map(label, "i")
        # d.set(yticks=[], ylabel="", xlabel="")
        # d.set_titles("")
        # #d.despine(bottom=True, left=True)
