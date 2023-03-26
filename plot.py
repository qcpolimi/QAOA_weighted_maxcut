import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

folder = '/home/rugantio/Downloads/QAOA_weighted_maxcut/exp_cobyla_final/'

p_x_p1 = pd.read_csv(folder + "perimeter_COBYLA_x_p=1.csv", index_col='n_qubs')
p_r_p1 = pd.read_csv(folder + "perimeter_COBYLA_r_p=1.csv", index_col='n_qubs')
p_xy_p1 = pd.read_csv(folder + "perimeter_COBYLA_xy_p=1.csv", index_col='n_qubs')
p_x_p2 = pd.read_csv(folder + "perimeter_COBYLA_x_p=2.csv", index_col='n_qubs')
p_r_p2 = pd.read_csv(folder + "perimeter_COBYLA_r_p=2.csv", index_col='n_qubs')
p_xy_p2 = pd.read_csv(folder + "perimeter_COBYLA_xy_p=2.csv", index_col='n_qubs')
p_x_p3 = pd.read_csv(folder + "perimeter_COBYLA_x_p=3.csv", index_col='n_qubs')
p_r_p3 = pd.read_csv(folder + "perimeter_COBYLA_r_p=3.csv", index_col='n_qubs')
p_xy_p3 = pd.read_csv(folder + "perimeter_COBYLA_xy_p=3.csv", index_col='n_qubs')
p_x_p4 = pd.read_csv(folder + "perimeter_COBYLA_x_p=4.csv", index_col='n_qubs')
p_r_p4 = pd.read_csv(folder + "perimeter_COBYLA_r_p=4.csv", index_col='n_qubs')
p_xy_p4 = pd.read_csv(folder + "perimeter_COBYLA_xy_p=4.csv", index_col='n_qubs')
s_x_p1 = pd.read_csv(folder + "star_COBYLA_x_p=1.csv", index_col='n_qubs')
s_r_p1 = pd.read_csv(folder + "star_COBYLA_r_p=1.csv", index_col='n_qubs')
s_xy_p1 = pd.read_csv(folder + "star_COBYLA_xy_p=1.csv", index_col='n_qubs')
s_x_p2 = pd.read_csv(folder + "star_COBYLA_x_p=2.csv", index_col='n_qubs')
s_r_p2 = pd.read_csv(folder + "star_COBYLA_r_p=2.csv", index_col='n_qubs')
s_xy_p2 = pd.read_csv(folder + "star_COBYLA_xy_p=2.csv", index_col='n_qubs')
s_x_p3 = pd.read_csv(folder + "star_COBYLA_x_p=3.csv", index_col='n_qubs')
s_r_p3 = pd.read_csv(folder + "star_COBYLA_r_p=3.csv", index_col='n_qubs')
s_xy_p3 = pd.read_csv(folder + "star_COBYLA_xy_p=3.csv", index_col='n_qubs')
s_x_p4 = pd.read_csv(folder + "star_COBYLA_x_p=4.csv", index_col='n_qubs')
s_r_p4 = pd.read_csv(folder + "star_COBYLA_r_p=4.csv", index_col='n_qubs')
s_xy_p4 = pd.read_csv(folder + "star_COBYLA_xy_p=4.csv", index_col='n_qubs')


# perimeter p 1 all mixers
fig = plt.figure(3, figsize=(10.24, 7.68))
x_1 = p_x_p1.index
x_n = p_x_p2.index
plt.errorbar(x_1, p_x_p1.approx_ratio_mean, yerr= p_x_p1.approx_ratio_std, marker='s', color='#9200ff', label='cycle x')
plt.errorbar(x_1, p_r_p1.approx_ratio_mean,  yerr= p_r_p1.approx_ratio_std, marker='o', color='#0000ff', label='cycle r')
plt.errorbar(x_1, p_xy_p1.approx_ratio_mean,  yerr= p_xy_p1.approx_ratio_std,marker='*', color='#00ccff', label = 'cycle xy')
# plt.axhline(0.5, linestyle='--', color='black', label='average')
#plt.xticks(x_1)#np.arange(0, experiments+1, step=5))
#plt.grid()
#plt.legend()
#plt.xlabel("Qubits")
#plt.ylabel("Approx ratio")

# star p 1 all mixers
#fig = plt.figure(3, figsize=(10.24, 7.68))
x_1 = p_x_p1.index
x_n = p_x_p2.index
#plt.errorbar(x_1, s_x_p1.approx_ratio_mean, marker='s', color='#ff0095', label='s x')
plt.errorbar(x_1, s_r_p1.approx_ratio_mean,  yerr= s_r_p1.approx_ratio_std, marker='o', color='#ff0000', label='star r and x')
plt.errorbar(x_1, s_xy_p1.approx_ratio_mean,  yerr= s_xy_p1.approx_ratio_std,marker='*', color='#ff9100', label = 'star xy')
plt.axhline(0.5, linestyle='--', color='black', label='average')
plt.xticks(x_1)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")
fig.savefig("/home/rugantio/Downloads/approx.svg", bbox_inches='tight', pad_inches=0)



# perimeter p 1 all mixers
fig = plt.figure(3, figsize=(10.24, 7.68))
x_1 = p_x_p1.index
x_n = p_x_p2.index
plt.plot(x_n, p_x_p2.approx_ratio_mean, marker='s', color='#9200ff', label='cycle x')
plt.plot(x_n, p_r_p2.approx_ratio_mean, marker='o', color='#0000ff', label='cycle r')
plt.plot(x_n, p_xy_p2.approx_ratio_mean, marker='*', color='#00ccff', label = 'cycle xy')
#plt.axhline(0.5, linestyle='--', color='black', label='average')
# plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
#plt.grid()
#plt.legend()
#plt.xlabel("Qubits")
#plt.ylabel("Approx ratio")

# star p 1 all mixers
#fig = plt.figure(3, figsize=(10.24, 7.68))
x_1 = p_x_p1.index
x_n = p_x_p2.index
#plt.plot(x_n, s_x_p2.approx_ratio_mean, marker='s', color='#ff0095', label='s x')
plt.plot(x_n, s_r_p2.approx_ratio_mean, marker='o', color='#ff0000', label='star x and r')
plt.plot(x_n, s_xy_p2.approx_ratio_mean, marker='*', color='#ff9100', label = 'star xy')
plt.axhline(0.5, linestyle='--', color='black', label='average')
plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")


# perimeter x, slice along p
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = p_x_p2.index
plt.plot(x_n, p_x_p1.approx_ratio_mean[:9], marker='o', color='#ff0095', label='cycle p=1')
plt.plot(x_n, p_x_p2.approx_ratio_mean, marker='o', color='#0D00C3', label='cycle p=2')
plt.plot(x_n, p_x_p3.approx_ratio_mean, marker='o', color='#00C30D', label = 'cycle p=3')
plt.plot(x_n, p_x_p4.approx_ratio_mean, marker='o', color='#9200ff', label='cycle p=4')
plt.axhline(0.5, linestyle='--', color='black', label='average')
plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")
fig.savefig("/home/rugantio/Downloads/slice_cycle.svg", bbox_inches='tight', pad_inches=0)


# star x, slice along p
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = p_x_p2.index
plt.plot(x_n, s_x_p1.approx_ratio_mean[:9], marker='o', color='#ff0095', label='s 1')
plt.plot(x_n, s_x_p2.approx_ratio_mean, marker='o', color='#ff0000', label='s 2')
plt.plot(x_n, s_x_p3.approx_ratio_mean, marker='o', color='#ff9100', label = 's 3')
plt.plot(x_n, s_x_p4.approx_ratio_mean, marker='o', color='#9200ff', label='s 4')
plt.axhline(0.5, linestyle='--', color='black', label='average')
plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")

# =============================================================================
# # perimeter r, slice along p
# fig = plt.figure(3, figsize=(10.24, 7.68))
# x_n = p_x_p2.index
# plt.plot(x_n, p_r_p1.approx_ratio_mean[:9], marker='o', color='#ff0095', label='p 1')
# plt.plot(x_n, p_r_p2.approx_ratio_mean, marker='o', color='#ff0000', label='p 2')
# plt.plot(x_n, p_r_p3.approx_ratio_mean, marker='o', color='#ff9100', label = 'p 3')
# plt.plot(x_n, p_r_p4.approx_ratio_mean, marker='o', color='#9200ff', label='p 4')
# plt.axhline(0.5, linestyle='--', color='black', label='average')
# plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
# plt.grid()
# plt.legend()
# plt.xlabel("Qubits")
# plt.ylabel("Approx ratio") 
# =============================================================================


# perimeter xy, slice along p
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = p_x_p2.index
plt.plot(x_n, p_xy_p1.approx_ratio_mean[:9], marker='o', color='#ff0095', label='p 1')
plt.plot(x_n, p_xy_p2.approx_ratio_mean, marker='o', color='#ff0000', label='p 2')
plt.plot(x_n, p_xy_p3.approx_ratio_mean, marker='o', color='#ff9100', label = 'p 3')
plt.plot(x_n, p_xy_p4.approx_ratio_mean, marker='o', color='#9200ff', label='p 4')
plt.axhline(0.5, linestyle='--', color='black', label='average')
plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")

# star xy, slice along p
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = p_x_p2.index
plt.plot(x_n, s_xy_p1.approx_ratio_mean[:9], marker='o', color='#ff0095', label='star p=1')
plt.plot(x_n, s_xy_p2.approx_ratio_mean, marker='o', color='#0D00C3', label='star p=2')
plt.plot(x_n, s_xy_p3.approx_ratio_mean, marker='o', color='#00C30D', label = 'star p=3')
plt.plot(x_n, s_xy_p4.approx_ratio_mean, marker='o', color='#9200ff', label='star p=4')
plt.axhline(0.5, linestyle='--', color='black', label='average')
plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")
fig.savefig("/home/rugantio/Downloads/slice_star.svg", bbox_inches='tight', pad_inches=0)


# =============================================================================
# # optimal sol count
# # perimeter x, slice along p
# fig = plt.figure(3, figsize=(10.24, 7.68))
# x_n = p_x_p2.index
# plt.plot(x_n, p_x_p1.optimal_sol_count[:9], marker='o', color='#ff0095', label='p 1')
# plt.plot(x_n, p_x_p2.optimal_sol_count, marker='o', color='#ff0000', label='p 2')
# plt.plot(x_n, p_x_p3.optimal_sol_count, marker='o', color='#ff9100', label = 'p 3')
# plt.plot(x_n, p_x_p4.optimal_sol_count, marker='o', color='#9200ff', label='p 4')
# plt.axhline(0.5, linestyle='--', color='black', label='average')
# plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
# plt.grid()
# plt.legend()
# plt.xlabel("Qubits")
# plt.ylabel("Optimal solution count")
# 
# =============================================================================

# OPTIMIZATION TIME
# perimeter p 1 all mixers
fig = plt.figure(3, figsize=(10.24, 7.68))
x_1 = p_x_p1.index
x_n = p_x_p2.index
plt.semilogy(x_1, p_x_p1.opt_time_mean, marker='s', color='#9200ff', label='cycle x')
plt.semilogy(x_1, p_r_p1.opt_time_mean, marker='o', color='#0000ff', label='cycle r')
plt.semilogy(x_1, p_xy_p1.opt_time_mean, marker='*', color='#00ccff', label = 'cycle xy')
# plt.xticks(x_1)#np.arange(0, experiments+1, step=5))
# plt.grid()
# plt.legend()
# plt.xlabel("Qubits")
# plt.ylabel("Optimization time")

# star p 1 all mixers
fig = plt.figure(3, figsize=(10.24, 7.68))
x_1 = p_x_p1.index
x_n = p_x_p2.index
plt.semilogy(x_1, s_x_p1.opt_time_mean, marker='s', color='#ff0095', label='star x')
plt.semilogy(x_1, s_r_p1.opt_time_mean, marker='o', color='#ff0000', label='star r')
plt.semilogy(x_1, s_xy_p1.opt_time_mean, marker='*', color='#ff9100', label = 'star xy')
plt.xticks(x_1)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Optimization time")

fig.savefig("/home/rugantio/Downloads/opt_time.svg", bbox_inches='tight', pad_inches=0)


# perimeter r, slice along p
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = p_x_p2.index
plt.semilogy(x_n, p_r_p1.opt_time_mean[:9], marker='o', color='#ff0095', label='p 1')
plt.semilogy(x_n, p_r_p2.opt_time_mean, marker='o', color='#ff0000', label='p 2')
plt.semilogy(x_n, p_r_p3.opt_time_mean, marker='o', color='#ff9100', label = 'p 3')
plt.semilogy(x_n, p_r_p4.opt_time_mean, marker='o', color='#9200ff', label='p 4')
plt.axhline(0.5, linestyle='--', color='black', label='average')
plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Optimization time")

# star r, slice along p
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = p_x_p2.index
plt.semilogy(x_n, s_r_p1.opt_time_mean[:9], marker='o', color='#ff0095', label='s 1')
plt.semilogy(x_n, s_r_p2.opt_time_mean, marker='o', color='#ff0000', label='s 2')
plt.semilogy(x_n, s_r_p3.opt_time_mean, marker='o', color='#ff9100', label = 's 3')
plt.semilogy(x_n, s_r_p4.opt_time_mean, marker='o', color='#9200ff', label='s 4')
# plt.axhline(0.5, linestyle='--', color='black', label='average')
plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Optimization time")

# OPT ITERATIONS 
# perimeter xy, slice along p
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = p_x_p2.index
plt.plot(x_n, p_xy_p1.opt_iterations_mean[:9], marker='o', color='#ff0095', label='p 1')
plt.plot(x_n, p_xy_p2.opt_iterations_mean, marker='o', color='#ff0000', label='p 2')
plt.plot(x_n, p_xy_p3.opt_iterations_mean, marker='o', color='#ff9100', label = 'p 3')
plt.plot(x_n, p_xy_p4.opt_iterations_mean, marker='o', color='#9200ff', label='p 4')
plt.axhline(0.5, linestyle='--', color='black', label='average')
plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")

# star xy, slice along p
fig = plt.figure(3, figsize=(10.24, 7.68))
x_n = p_x_p2.index
plt.plot(x_n, s_xy_p1.opt_iterations_mean[:9], marker='o', color='#ff0095', label='s 1')
plt.plot(x_n, s_xy_p2.opt_iterations_mean, marker='o', color='#ff0000', label='s 2')
plt.plot(x_n, s_xy_p3.opt_iterations_mean, marker='o', color='#ff9100', label = 's 3')
plt.plot(x_n, s_xy_p4.opt_iterations_mean, marker='o', color='#9200ff', label='s 4')
plt.axhline(0.5, linestyle='--', color='black', label='average')
plt.xticks(x_n)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")

# perimeter p 1 all mixers
fig = plt.figure(3, figsize=(10.24, 7.68))
x_1 = p_x_p1.index
x_n = p_x_p2.index
plt.plot(x_1, p_x_p1.opt_iterations_mean, marker='s', color='#9200ff', label='p x')
plt.plot(x_1, p_r_p1.opt_iterations_mean, marker='o', color='#0000ff', label='p r')
plt.plot(x_1, p_xy_p1.opt_iterations_mean, marker='*', color='#00ccff', label = 'p xy')
plt.xticks(x_1)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")

# star p 1 all mixers
fig = plt.figure(3, figsize=(10.24, 7.68))
x_1 = p_x_p1.index
x_n = p_x_p2.index
plt.plot(x_1, s_x_p1.opt_iterations_mean, marker='s', color='#ff0095', label='s x')
plt.plot(x_1, s_r_p1.opt_iterations_mean, marker='o', color='#ff0000', label='s r')
plt.plot(x_1, s_xy_p1.opt_iterations_mean, marker='*', color='#ff9100', label = 's xy')
plt.xticks(x_1)#np.arange(0, experiments+1, step=5))
plt.grid()
plt.legend()
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")

# # fig.savefig("/home/rugantio/Downloads/approx.svg", bbox_inches='tight', pad_inches=0)
# plt.clf()