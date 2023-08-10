import matplotlib.pyplot as plt
import pandas as pd

folder = 'results_figure_3ab_powell_plot/'

# cycle
c_x_p1 = pd.read_csv(folder + "cycle_POWELL_x_p=1.csv", index_col='n_qubs')
# c_r_p1 = pd.read_csv(folder + "cycle_POWELL_r_p=1.csv", index_col='n_qubs')
c_xy_p1 = pd.read_csv(folder + "cycle_POWELL_xy_p=1.csv", index_col='n_qubs')

c_x_p2 = pd.read_csv(folder + "cycle_POWELL_x_p=2.csv", index_col='n_qubs')
# c_r_p2 = pd.read_csv(folder + "cycle_POWELL_r_p=2.csv", index_col='n_qubs')
c_xy_p2 = pd.read_csv(folder + "cycle_POWELL_xy_p=2.csv", index_col='n_qubs')

c_x_p3 = pd.read_csv(folder + "cycle_POWELL_x_p=3.csv", index_col='n_qubs')
# c_r_p3 = pd.read_csv(folder + "cycle_POWELL_r_p=3.csv", index_col='n_qubs')
c_xy_p3 = pd.read_csv(folder + "cycle_POWELL_xy_p=3.csv", index_col='n_qubs')

# star
s_x_p1 = pd.read_csv(folder + "star_POWELL_x_p=1.csv", index_col='n_qubs')
# s_r_p1 = pd.read_csv(folder + "star_POWELL_r_p=1.csv", index_col='n_qubs')
s_xy_p1 = pd.read_csv(folder + "star_POWELL_xy_p=1.csv", index_col='n_qubs')

s_x_p2 = pd.read_csv(folder + "star_POWELL_x_p=2.csv", index_col='n_qubs')
# s_r_p2 = pd.read_csv(folder + "star_POWELL_r_p=2.csv", index_col='n_qubs')
s_xy_p2 = pd.read_csv(folder + "star_POWELL_xy_p=2.csv", index_col='n_qubs')

s_x_p3 = pd.read_csv(folder + "star_POWELL_x_p=3.csv", index_col='n_qubs')
# s_r_p3 = pd.read_csv(folder + "star_POWELL_r_p=3.csv", index_col='n_qubs')
s_xy_p3 = pd.read_csv(folder + "star_POWELL_xy_p=3.csv", index_col='n_qubs')


# 1a
# cycle x, slice along p, approx ratio vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = c_x_p1.index

# plt.errorbar(x_n, c_x_p1.approx_ratio_mean, yerr=c_x_p1.approx_ratio_std, capsize=1, marker='o', color='#ff0095', label='cycle x p=1')
# plt.errorbar(x_n, c_x_p2.approx_ratio_mean, yerr=c_x_p2.approx_ratio_std, capsize=1, marker='o', color='#0D00C3', label='cycle x p=2')
# plt.errorbar(x_n, c_x_p3.approx_ratio_mean, yerr=c_x_p3.approx_ratio_std, capsize=1, marker='o', color='#00C30D', label='cycle x p=3')

plt.fill_between(x_n, c_x_p1.approx_ratio_mean - c_x_p1.approx_ratio_std, c_x_p1.approx_ratio_mean + c_x_p1.approx_ratio_std, color='#ff0095', alpha=0.1)
plt.fill_between(x_n, c_x_p2.approx_ratio_mean - c_x_p2.approx_ratio_std, c_x_p2.approx_ratio_mean + c_x_p2.approx_ratio_std, color='#0D00C3', alpha=0.1)
plt.fill_between(x_n, c_x_p3.approx_ratio_mean - c_x_p3.approx_ratio_std, c_x_p3.approx_ratio_mean + c_x_p3.approx_ratio_std, color='#00C30D', alpha=0.1)

plt.plot(x_n, c_x_p1.approx_ratio_mean, marker='o', color='#ff0095', label='cycle x p=1')
plt.plot(x_n, c_x_p2.approx_ratio_mean, marker='o', color='#0D00C3', label='cycle x p=2')
plt.plot(x_n, c_x_p3.approx_ratio_mean, marker='o', color='#00C30D', label='cycle x p=3')
plt.axhline(0.5, linestyle='--', color='black', label='average')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")
fig.savefig('results_figure_3ab_powell_plot/cycle_x_p_approx_ratio_n.svg', bbox_inches='tight', pad_inches=0)
# plt.clf()
plt.show()


# 1b
# star x, slice along p, approx ratio vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = s_x_p1.index

# plt.errorbar(x_n, s_x_p1.approx_ratio_mean, yerr=s_x_p1.approx_ratio_std, capsize=1, marker='o', color='#ff0095', label='star x p=1')
# plt.errorbar(x_n, s_x_p2.approx_ratio_mean, yerr=s_x_p2.approx_ratio_std, capsize=1, marker='o', color='#0D00C3', label='star x p=2')
# plt.errorbar(x_n, s_x_p3.approx_ratio_mean, yerr=s_x_p3.approx_ratio_std, capsize=1, marker='o', color='#00C30D', label='star x p=3')

plt.fill_between(x_n, s_x_p1.approx_ratio_mean - s_x_p1.approx_ratio_std, s_x_p1.approx_ratio_mean + s_x_p1.approx_ratio_std, color='#ff0095', alpha=0.1)
plt.fill_between(x_n, s_x_p2.approx_ratio_mean - s_x_p2.approx_ratio_std, s_x_p2.approx_ratio_mean + s_x_p2.approx_ratio_std, color='#0D00C3', alpha=0.1)
plt.fill_between(x_n, s_x_p3.approx_ratio_mean - s_x_p3.approx_ratio_std, s_x_p3.approx_ratio_mean + s_x_p3.approx_ratio_std, color='#00C30D', alpha=0.1)

plt.plot(x_n, s_x_p1.approx_ratio_mean, marker='o', color='#ff0095', label='star x p=1')
plt.plot(x_n, s_x_p2.approx_ratio_mean, marker='o', color='#0D00C3', label='star x p=2')
plt.plot(x_n, s_x_p3.approx_ratio_mean, marker='o', color='#00C30D', label='star x p=3')
plt.axhline(0.5, linestyle='--', color='black', label='average')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")
fig.savefig('results_figure_3ab_powell_plot/star_x_p_approx_ratio_n.svg', bbox_inches='tight', pad_inches=0)
# plt.clf()
plt.show()


# 1c
# cycle xy, slice along p, approx ratio vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = c_xy_p1.index

# plt.errorbar(x_n, c_xy_p1.approx_ratio_mean, yerr=c_xy_p1.approx_ratio_std, capsize=1, marker='o', color='#ff0095', label='cycle xy p=1')
# plt.errorbar(x_n, c_xy_p2.approx_ratio_mean, yerr=c_xy_p2.approx_ratio_std, capsize=1, marker='o', color='#0D00C3', label='cycle xy p=2')
# plt.errorbar(x_n, c_xy_p3.approx_ratio_mean, yerr=c_xy_p3.approx_ratio_std, capsize=1, marker='o', color='#00C30D', label='cycle xy p=3')

plt.fill_between(x_n, c_xy_p1.approx_ratio_mean - c_xy_p1.approx_ratio_std, c_xy_p1.approx_ratio_mean + c_xy_p1.approx_ratio_std, color='#ff0095', alpha=0.1)
plt.fill_between(x_n, c_xy_p2.approx_ratio_mean - c_xy_p2.approx_ratio_std, c_xy_p2.approx_ratio_mean + c_xy_p2.approx_ratio_std, color='#0D00C3', alpha=0.1)
plt.fill_between(x_n, c_xy_p3.approx_ratio_mean - c_xy_p3.approx_ratio_std, c_xy_p3.approx_ratio_mean + c_xy_p3.approx_ratio_std, color='#00C30D', alpha=0.1)

plt.plot(x_n, c_xy_p1.approx_ratio_mean, marker='o', color='#ff0095', label='cycle xy p=1')
plt.plot(x_n, c_xy_p2.approx_ratio_mean, marker='o', color='#0D00C3', label='cycle xy p=2')
plt.plot(x_n, c_xy_p3.approx_ratio_mean, marker='o', color='#00C30D', label='cycle xy p=3')
plt.axhline(0.5, linestyle='--', color='black', label='average')

plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")
fig.savefig('results_figure_3ab_powell_plot/cycle_xy_p_approx_ratio_n.svg', bbox_inches='tight', pad_inches=0)
# plt.clf()
plt.show()


# 1d
# star xy, slice along p, approx ratio vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = s_xy_p1.index

# plt.errorbar(x_n, s_xy_p1.approx_ratio_mean, yerr=s_xy_p1.approx_ratio_std, capsize=1, marker='o', color='#ff0095', label='star xy p=1')
# plt.errorbar(x_n, s_xy_p2.approx_ratio_mean, yerr=s_xy_p2.approx_ratio_std, capsize=1, marker='o', color='#0D00C3', label='star xy p=2')
# plt.errorbar(x_n, s_xy_p3.approx_ratio_mean, yerr=s_xy_p3.approx_ratio_std, capsize=1, marker='o', color='#00C30D', label='star xy p=3')

plt.fill_between(x_n, s_xy_p1.approx_ratio_mean - s_xy_p1.approx_ratio_std, s_xy_p1.approx_ratio_mean + s_xy_p1.approx_ratio_std, color='#ff0095', alpha=0.1)
plt.fill_between(x_n, s_xy_p2.approx_ratio_mean - s_xy_p2.approx_ratio_std, s_xy_p2.approx_ratio_mean + s_xy_p2.approx_ratio_std, color='#0D00C3', alpha=0.1)
plt.fill_between(x_n, s_xy_p3.approx_ratio_mean - s_xy_p3.approx_ratio_std, s_xy_p3.approx_ratio_mean + s_xy_p3.approx_ratio_std, color='#00C30D', alpha=0.1)

plt.plot(x_n, s_xy_p1.approx_ratio_mean, marker='o', color='#ff0095', label='star xy p=1')
plt.plot(x_n, s_xy_p2.approx_ratio_mean, marker='o', color='#0D00C3', label='star xy p=2')
plt.plot(x_n, s_xy_p3.approx_ratio_mean, marker='o', color='#00C30D', label='star xy p=3')
plt.axhline(0.5, linestyle='--', color='black', label='average')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")
fig.savefig("results_figure_3ab_powell_plot/star_xy_p_approx_ratio_n.svg", bbox_inches='tight', pad_inches=0)
# plt.clf()
plt.show()


# 2a
# cycle x, slice along p, most prob sol ratio vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = c_x_p1.index

# plt.errorbar(x_n, c_x_p1.most_prob_sol_ratio_mean, yerr=c_x_p1.most_prob_sol_ratio_std, capsize=1, marker='o', color='#ff0095', label='cycle x p=1')
# plt.errorbar(x_n, c_x_p2.most_prob_sol_ratio_mean, yerr=c_x_p2.most_prob_sol_ratio_std, capsize=1, marker='o', color='#0D00C3', label='cycle x p=2')
# plt.errorbar(x_n, c_x_p3.most_prob_sol_ratio_mean, yerr=c_x_p3.most_prob_sol_ratio_std, capsize=1, marker='o', color='#00C30D', label='cycle x p=3')

plt.fill_between(x_n, c_x_p1.most_prob_sol_ratio_mean - c_x_p1.most_prob_sol_ratio_std, c_x_p1.most_prob_sol_ratio_mean + c_x_p1.most_prob_sol_ratio_std, color='#ff0095', alpha=0.1)
plt.fill_between(x_n, c_x_p2.most_prob_sol_ratio_mean - c_x_p2.most_prob_sol_ratio_std, c_x_p2.most_prob_sol_ratio_mean + c_x_p2.most_prob_sol_ratio_std, color='#0D00C3', alpha=0.1)
plt.fill_between(x_n, c_x_p3.most_prob_sol_ratio_mean - c_x_p3.most_prob_sol_ratio_std, c_x_p3.most_prob_sol_ratio_mean + c_x_p3.most_prob_sol_ratio_std, color='#00C30D', alpha=0.1)

plt.plot(x_n, c_x_p1.most_prob_sol_ratio_mean, marker='o', color='#ff0095', label='cycle x p=1')
plt.plot(x_n, c_x_p2.most_prob_sol_ratio_mean, marker='o', color='#0D00C3', label='cycle x p=2')
plt.plot(x_n, c_x_p3.most_prob_sol_ratio_mean, marker='o', color='#00C30D', label='cycle x p=3')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Sol ratio")
fig.savefig('results_figure_3ab_powell_plot/cycle_x_p_most_prob_sol _ratio_n.svg', bbox_inches='tight', pad_inches=0)
# plt.clf()
plt.show()


# 2b
# star x, slice along p, most prob sol ratio vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = s_x_p1.index

# plt.errorbar(x_n, s_x_p1.most_prob_sol_ratio_mean, yerr=s_x_p1.most_prob_sol_ratio_std, capsize=1, marker='o', color='#ff0095', label='star x p=1')
# plt.errorbar(x_n, s_x_p2.most_prob_sol_ratio_mean, yerr=s_x_p2.most_prob_sol_ratio_std, capsize=1, marker='o', color='#0D00C3', label='star x p=2')
# plt.errorbar(x_n, s_x_p3.most_prob_sol_ratio_mean, yerr=s_x_p3.most_prob_sol_ratio_std, capsize=1, marker='o', color='#00C30D', label='star x p=3')

plt.fill_between(x_n, s_x_p1.most_prob_sol_ratio_mean - s_x_p1.most_prob_sol_ratio_std, s_x_p1.most_prob_sol_ratio_mean + s_x_p1.most_prob_sol_ratio_std, color='#ff0095', alpha=0.1)
plt.fill_between(x_n, s_x_p2.most_prob_sol_ratio_mean - s_x_p2.most_prob_sol_ratio_std, s_x_p2.most_prob_sol_ratio_mean + s_x_p2.most_prob_sol_ratio_std, color='#0D00C3', alpha=0.1)
plt.fill_between(x_n, s_x_p3.most_prob_sol_ratio_mean - s_x_p3.most_prob_sol_ratio_std, s_x_p3.most_prob_sol_ratio_mean + s_x_p3.most_prob_sol_ratio_std, color='#00C30D', alpha=0.1)

plt.plot(x_n, s_x_p1.most_prob_sol_ratio_mean, marker='o', color='#ff0095', label='star x p=1')
plt.plot(x_n, s_x_p2.most_prob_sol_ratio_mean, marker='o', color='#0D00C3', label='star x p=2')
plt.plot(x_n, s_x_p3.most_prob_sol_ratio_mean, marker='o', color='#00C30D', label='star x p=3')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Sol ratio")
fig.savefig('results_figure_3ab_powell_plot/star_x_p_most_prob_sol _ratio_n.svg', bbox_inches='tight', pad_inches=0)
# plt.clf()
plt.show()


# 2c
# cycle xy, slice along p, most prob sol ratio vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = c_xy_p1.index

# plt.errorbar(x_n, c_xy_p1.most_prob_sol_ratio_mean, yerr=c_xy_p1.most_prob_sol_ratio_std, capsize=1, marker='o', color='#ff0095', label='cycle xy p=1')
# plt.errorbar(x_n, c_xy_p2.most_prob_sol_ratio_mean, yerr=c_xy_p2.most_prob_sol_ratio_std, capsize=1, marker='o', color='#0D00C3', label='cycle xy p=2')
# plt.errorbar(x_n, c_xy_p3.most_prob_sol_ratio_mean, yerr=c_xy_p3.most_prob_sol_ratio_std, capsize=1, marker='o', color='#00C30D', label='cycle xy p=3')

plt.fill_between(x_n, c_xy_p1.most_prob_sol_ratio_mean - c_xy_p1.most_prob_sol_ratio_std, c_xy_p1.most_prob_sol_ratio_mean + c_xy_p1.most_prob_sol_ratio_std, color='#ff0095', alpha=0.1)
plt.fill_between(x_n, c_xy_p2.most_prob_sol_ratio_mean - c_xy_p2.most_prob_sol_ratio_std, c_xy_p2.most_prob_sol_ratio_mean + c_xy_p2.most_prob_sol_ratio_std, color='#0D00C3', alpha=0.1)
plt.fill_between(x_n, c_xy_p3.most_prob_sol_ratio_mean - c_xy_p3.most_prob_sol_ratio_std, c_xy_p3.most_prob_sol_ratio_mean + c_xy_p3.most_prob_sol_ratio_std, color='#00C30D', alpha=0.1)

plt.plot(x_n, c_xy_p1.most_prob_sol_ratio_mean, marker='o', color='#ff0095', label='cycle xy p=1')
plt.plot(x_n, c_xy_p2.most_prob_sol_ratio_mean, marker='o', color='#0D00C3', label='cycle xy p=2')
plt.plot(x_n, c_xy_p3.most_prob_sol_ratio_mean, marker='o', color='#00C30D', label='cycle xy p=3')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Sol ratio")
fig.savefig('results_figure_3ab_powell_plot/cycle_xy_p_most_prob_sol _ratio_n.svg', bbox_inches='tight', pad_inches=0)
# plt.clf()
plt.show()


# 2d
# star xy, slice along p, most prob sol ratio vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = s_xy_p1.index

# plt.errorbar(x_n, s_xy_p1.most_prob_sol_ratio_mean, yerr=s_xy_p1.most_prob_sol_ratio_std, capsize=1, marker='o', color='#ff0095', label='star xy p=1')
# plt.errorbar(x_n, s_xy_p2.most_prob_sol_ratio_mean, yerr=s_xy_p2.most_prob_sol_ratio_std, capsize=1, marker='o', color='#0D00C3', label='star xy p=2')
# plt.errorbar(x_n, s_xy_p3.most_prob_sol_ratio_mean, yerr=s_xy_p3.most_prob_sol_ratio_std, capsize=1, marker='o', color='#00C30D', label='star xy p=3')

plt.fill_between(x_n, s_xy_p1.most_prob_sol_ratio_mean - s_xy_p1.most_prob_sol_ratio_std, s_xy_p1.most_prob_sol_ratio_mean + s_xy_p1.most_prob_sol_ratio_std, color='#ff0095', alpha=0.1)
plt.fill_between(x_n, s_xy_p2.most_prob_sol_ratio_mean - s_xy_p2.most_prob_sol_ratio_std, s_xy_p2.most_prob_sol_ratio_mean + s_xy_p2.most_prob_sol_ratio_std, color='#0D00C3', alpha=0.1)
plt.fill_between(x_n, s_xy_p3.most_prob_sol_ratio_mean - s_xy_p3.most_prob_sol_ratio_std, s_xy_p3.most_prob_sol_ratio_mean + s_xy_p3.most_prob_sol_ratio_std, color='#00C30D', alpha=0.1)

plt.plot(x_n, s_xy_p1.most_prob_sol_ratio_mean, marker='o', color='#ff0095', label='star xy p=1')
plt.plot(x_n, s_xy_p2.most_prob_sol_ratio_mean, marker='o', color='#0D00C3', label='star xy p=2')
plt.plot(x_n, s_xy_p3.most_prob_sol_ratio_mean, marker='o', color='#00C30D', label='star xy p=3')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Sol ratio")
fig.savefig("results_figure_3ab_powell_plot/star_xy_p_most_prob_sol _ratio_n.svg", bbox_inches='tight', pad_inches=0)
# plt.clf()
plt.show()


# 3a
# cycle x, slice along p, optimal sol count vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = c_x_p1.index

plt.plot(x_n, c_x_p1.optimal_sol_count/100, marker='o', color='#ff0095', label='cycle x p=1')
plt.plot(x_n, c_x_p2.optimal_sol_count/100, marker='o', color='#0D00C3', label='cycle x p=2')
plt.plot(x_n, c_x_p3.optimal_sol_count/100, marker='o', color='#00C30D', label='cycle x p=3')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Optimal sol")
fig.savefig('results_figure_3ab_powell_plot/cycle_x_p_optimal_sol_count_n.svg', bbox_inches='tight', pad_inches=0)
# plt.clf()
plt.show()


# 3b
# star x, slice along p, optimal sol count vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = s_x_p1.index

plt.plot(x_n, s_x_p1.optimal_sol_count/100, marker='o', color='#ff0095', label='star x p=1')
plt.plot(x_n, s_x_p2.optimal_sol_count/100, marker='o', color='#0D00C3', label='star x p=2')
plt.plot(x_n, s_x_p3.optimal_sol_count/100, marker='o', color='#00C30D', label='star x p=3')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Optimal sol")
fig.savefig('results_figure_3ab_powell_plot/star_x_p_optimal_sol_count_n.svg', bbox_inches='tight', pad_inches=0)
# plt.clf()
plt.show()


# 3c
# cycle xy, slice along p, optimal sol count vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = c_xy_p1.index

plt.plot(x_n, c_xy_p1.optimal_sol_count/100, marker='o', color='#ff0095', label='cycle xy p=1')
plt.plot(x_n, c_xy_p2.optimal_sol_count/100, marker='o', color='#0D00C3', label='cycle xy p=2')
plt.plot(x_n, c_xy_p3.optimal_sol_count/100, marker='o', color='#00C30D', label='cycle xy p=3')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Optimal sol")
fig.savefig('results_figure_3ab_powell_plot/cycle_xy_p_optimal_sol_count_n.svg', bbox_inches='tight', pad_inches=0)
# plt.clf()
plt.show()


# 3d
# star xy, slice along p, optimal sol count vs n
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = s_xy_p1.index

plt.plot(x_n, s_xy_p1.optimal_sol_count/100, marker='o', color='#ff0095', label='star xy p=1')
plt.plot(x_n, s_xy_p2.optimal_sol_count/100, marker='o', color='#0D00C3', label='star xy p=2')
plt.plot(x_n, s_xy_p3.optimal_sol_count/100, marker='o', color='#00C30D', label='star xy p=3')

plt.xticks(x_n)
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Optimal sol")
fig.savefig("results_figure_3ab_powell_plot/star_xy_p_optimal_sol_count_n.svg", bbox_inches='tight', pad_inches=0)
# plt.clf()
plt.show()