import matplotlib.pyplot as plt
import pandas as pd

folder = 'results_figure_2ab_powell_plot/'

# cycle
c_x_p1 = pd.read_csv(folder + "cycle_POWELL_x_p=1.csv", index_col='n_qubs')
c_r_p1 = pd.read_csv(folder + "cycle_POWELL_r_p=1.csv", index_col='n_qubs')
c_xy_p1 = pd.read_csv(folder + "cycle_POWELL_xy_p=1.csv", index_col='n_qubs')

# star
s_x_p1 = pd.read_csv(folder + "star_POWELL_x_p=1.csv", index_col='n_qubs')
s_r_p1 = pd.read_csv(folder + "star_POWELL_r_p=1.csv", index_col='n_qubs')
s_xy_p1 = pd.read_csv(folder + "star_POWELL_xy_p=1.csv", index_col='n_qubs')


# 1
# all mixers, p=1, slice along graph-mixer, approx ratio vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = c_x_p1.index

# plt.errorbar(x_n, c_x_p1.approx_ratio_mean, yerr=c_x_p1.approx_ratio_std, marker='s', color='#9200ff', label='cycle x')
# plt.errorbar(x_n, c_r_p1.approx_ratio_mean, yerr=c_r_p1.approx_ratio_std, marker='o', color='#0000ff', label='cycle r')
# plt.errorbar(x_n, c_xy_p1.approx_ratio_mean, yerr=c_xy_p1.approx_ratio_std, marker='*', color='#00ccff', label='cycle xy')
# plt.errorbar(x_n, s_x_p1.approx_ratio_mean, marker='s', color='#ff0095', label='star x')
# plt.errorbar(x_n, s_r_p1.approx_ratio_mean, yerr=s_r_p1.approx_ratio_std, marker='o', color='#ff0000', label='star r')
# plt.errorbar(x_n, s_xy_p1.approx_ratio_mean, yerr=s_xy_p1.approx_ratio_std, marker='*', color='#ff9100', label='star xy')

# plt.fill_between(x_n, c_x_p1.approx_ratio_mean - c_x_p1.approx_ratio_std, c_x_p1.approx_ratio_mean + c_x_p1.approx_ratio_std, color='#9200ff', alpha=0.1)
# plt.fill_between(x_n, c_r_p1.approx_ratio_mean - c_r_p1.approx_ratio_std, c_r_p1.approx_ratio_mean + c_r_p1.approx_ratio_std, color='#0000ff', alpha=0.1)
# plt.fill_between(x_n, c_xy_p1.approx_ratio_mean - c_xy_p1.approx_ratio_std, c_xy_p1.approx_ratio_mean + c_xy_p1.approx_ratio_std, color='#00ccff', alpha=0.1)
# plt.fill_between(x_n, s_x_p1.approx_ratio_mean - s_x_p1.approx_ratio_std, s_x_p1.approx_ratio_mean + s_x_p1.approx_ratio_std, color='#ff0095', alpha=0.1)
# plt.fill_between(x_n, s_r_p1.approx_ratio_mean - s_r_p1.approx_ratio_std, s_r_p1.approx_ratio_mean + s_r_p1.approx_ratio_std, color='#ff0000', alpha=0.1)
# plt.fill_between(x_n, s_xy_p1.approx_ratio_mean - s_xy_p1.approx_ratio_std, s_xy_p1.approx_ratio_mean + s_xy_p1.approx_ratio_std, color='#ff9100', alpha=0.1)

plt.plot(x_n, c_x_p1.approx_ratio_mean, marker='o', color='#9200ff', label='cycle x')
plt.plot(x_n, c_r_p1.approx_ratio_mean, marker='o', color='#0000ff', label='cycle r')
plt.plot(x_n, c_xy_p1.approx_ratio_mean, marker='o', color='#00ccff', label='cycle xy')
plt.plot(x_n, s_x_p1.approx_ratio_mean, marker='o', color='#ff0095', label='star x')
plt.plot(x_n, s_r_p1.approx_ratio_mean, marker='o', color='#ff0000', label='star r')
plt.plot(x_n, s_xy_p1.approx_ratio_mean, marker='o', color='#ff9100', label='star xy')
plt.axhline(0.5, linestyle='--', color='black', label='average')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")
fig.savefig('results_figure_2ab_powell_plot/all_mixer_p=1_graph-mixer_approx_ratio_n.svg', bbox_inches='tight', pad_inches=0)
plt.show()


# 2
# all mixers, p=1, slice along graph-mixer, time vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = c_x_p1.index

# plt.errorbar(x_n, c_x_p1.opt_time_mean, yerr=c_x_p1.opt_time_std, marker='s', color='#9200ff', label='cycle x')
# plt.errorbar(x_n, c_r_p1.opt_time_mean, yerr=c_r_p1.opt_time_std, marker='o', color='#0000ff', label='cycle r')
# plt.errorbar(x_n, c_xy_p1.opt_time_mean, yerr=c_xy_p1.opt_time_std, marker='*', color='#00ccff', label='cycle xy')
# plt.errorbar(x_n, s_x_p1.opt_time_mean, marker='s', color='#ff0095', label='star x')
# plt.errorbar(x_n, s_r_p1.opt_time_mean, yerr=s_r_p1.opt_time_std, marker='o', color='#ff0000', label='star r')
# plt.errorbar(x_n, s_xy_p1.opt_time_mean, yerr=s_xy_p1.opt_time_std, marker='*', color='#ff9100', label='star xy')

# plt.fill_between(x_n, c_x_p1.opt_time_mean - c_x_p1.opt_time_std, c_x_p1.opt_time_mean + c_x_p1.opt_time_std, color='#9200ff', alpha=0.1)
# plt.fill_between(x_n, c_r_p1.opt_time_mean - c_r_p1.opt_time_std, c_r_p1.opt_time_mean + c_r_p1.opt_time_std, color='#0000ff', alpha=0.1)
# plt.fill_between(x_n, c_xy_p1.opt_time_mean - c_xy_p1.opt_time_std, c_xy_p1.opt_time_mean + c_xy_p1.opt_time_std, color='#00ccff', alpha=0.1)
# plt.fill_between(x_n, s_x_p1.opt_time_mean - s_x_p1.opt_time_std, s_x_p1.opt_time_mean + s_x_p1.opt_time_std, color='#ff0095', alpha=0.1)
# plt.fill_between(x_n, s_r_p1.opt_time_mean - s_r_p1.opt_time_std, s_r_p1.opt_time_mean + s_r_p1.opt_time_std, color='#ff0000', alpha=0.1)
# plt.fill_between(x_n, s_xy_p1.opt_time_mean - s_xy_p1.opt_time_std, s_xy_p1.opt_time_mean + s_xy_p1.opt_time_std, color='#ff9100', alpha=0.1)

plt.semilogy(x_n, c_x_p1.opt_time_mean, marker='o', color='#9200ff', label='cycle x')
plt.semilogy(x_n, c_r_p1.opt_time_mean, marker='o', color='#0000ff', label='cycle r')
plt.semilogy(x_n, c_xy_p1.opt_time_mean, marker='o', color='#00ccff', label='cycle xy')
plt.semilogy(x_n, s_x_p1.opt_time_mean, marker='o', color='#ff0095', label='star x')
plt.semilogy(x_n, s_r_p1.opt_time_mean, marker='o', color='#ff0000', label='star r')
plt.semilogy(x_n, s_xy_p1.opt_time_mean, marker='o', color='#ff9100', label='star xy')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Optimization time")
fig.savefig('results_figure_2ab_powell_plot/all_mixer_p=1_graph-mixer_opt_time_n.svg', bbox_inches='tight', pad_inches=0)
plt.show()


# 3
# all mixers, p=1, slice along graph-mixer, sol ratio vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = c_x_p1.index

# plt.errorbar(x_n, c_x_p1.most_prob_sol_ratio_mean, yerr=c_x_p1.most_prob_sol_ratio_std, marker='s', color='#9200ff', label='cycle x')
# plt.errorbar(x_n, c_r_p1.most_prob_sol_ratio_mean, yerr=c_r_p1.most_prob_sol_ratio_std, marker='o', color='#0000ff', label='cycle r')
# plt.errorbar(x_n, c_xy_p1.most_prob_sol_ratio_mean, yerr=c_xy_p1.most_prob_sol_ratio_std, marker='*', color='#00ccff', label='cycle xy')
# plt.errorbar(x_n, s_x_p1.most_prob_sol_ratio_mean, marker='s', color='#ff0095', label='star x')
# plt.errorbar(x_n, s_r_p1.most_prob_sol_ratio_mean, yerr=s_r_p1.most_prob_sol_ratio_std, marker='o', color='#ff0000', label='star r')
# plt.errorbar(x_n, s_xy_p1.most_prob_sol_ratio_mean, yerr=s_xy_p1.most_prob_sol_ratio_std, marker='*', color='#ff9100', label='star xy')

# plt.fill_between(x_n, c_x_p1.most_prob_sol_ratio_mean - c_x_p1.most_prob_sol_ratio_std, c_x_p1.most_prob_sol_ratio_mean + c_x_p1.most_prob_sol_ratio_std, color='#9200ff', alpha=0.1)
# plt.fill_between(x_n, c_r_p1.most_prob_sol_ratio_mean - c_r_p1.most_prob_sol_ratio_std, c_r_p1.most_prob_sol_ratio_mean + c_r_p1.most_prob_sol_ratio_std, color='#0000ff', alpha=0.1)
# plt.fill_between(x_n, c_xy_p1.most_prob_sol_ratio_mean - c_xy_p1.most_prob_sol_ratio_std, c_xy_p1.most_prob_sol_ratio_mean + c_xy_p1.most_prob_sol_ratio_std, color='#00ccff', alpha=0.1)
# plt.fill_between(x_n, s_x_p1.most_prob_sol_ratio_mean - s_x_p1.most_prob_sol_ratio_std, s_x_p1.most_prob_sol_ratio_mean + s_x_p1.most_prob_sol_ratio_std, color='#ff0095', alpha=0.1)
# plt.fill_between(x_n, s_r_p1.most_prob_sol_ratio_mean - s_r_p1.most_prob_sol_ratio_std, s_r_p1.most_prob_sol_ratio_mean + s_r_p1.most_prob_sol_ratio_std, color='#ff0000', alpha=0.1)
# plt.fill_between(x_n, s_xy_p1.most_prob_sol_ratio_mean - s_xy_p1.most_prob_sol_ratio_std, s_xy_p1.most_prob_sol_ratio_mean + s_xy_p1.most_prob_sol_ratio_std, color='#ff9100', alpha=0.1)

plt.plot(x_n, c_x_p1.most_prob_sol_ratio_mean, marker='o', color='#9200ff', label='cycle x')
plt.plot(x_n, c_r_p1.most_prob_sol_ratio_mean, marker='o', color='#0000ff', label='cycle r')
plt.plot(x_n, c_xy_p1.most_prob_sol_ratio_mean, marker='o', color='#00ccff', label='cycle xy')
plt.plot(x_n, s_x_p1.most_prob_sol_ratio_mean, marker='o', color='#ff0095', label='star x')
plt.plot(x_n, s_r_p1.most_prob_sol_ratio_mean, marker='o', color='#ff0000', label='star r')
plt.plot(x_n, s_xy_p1.most_prob_sol_ratio_mean, marker='o', color='#ff9100', label='star xy')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Sol ratio")
fig.savefig('results_figure_2ab_powell_plot/all_mixer_p=1_graph-mixer_sol_ratio_n.svg', bbox_inches='tight', pad_inches=0)
plt.show()


# 4
# all mixers, p=1, slice along graph-mixer, sol count vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = c_x_p1.index

plt.plot(x_n, c_x_p1.optimal_sol_count/100, marker='o', color='#9200ff', label='cycle x')
plt.plot(x_n, c_r_p1.optimal_sol_count/100, marker='o', color='#0000ff', label='cycle r')
plt.plot(x_n, c_xy_p1.optimal_sol_count/100, marker='o', color='#00ccff', label='cycle xy')
plt.plot(x_n, s_x_p1.optimal_sol_count/100, marker='o', color='#ff0095', label='star x')
plt.plot(x_n, s_r_p1.optimal_sol_count/100, marker='o', color='#ff0000', label='star r')
plt.plot(x_n, s_xy_p1.optimal_sol_count/100, marker='o', color='#ff9100', label='star xy')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Optimal sol")
fig.savefig('results_figure_2ab_powell_plot/all_mixer_p=1_graph-mixer_sol_count_n.svg', bbox_inches='tight', pad_inches=0)

