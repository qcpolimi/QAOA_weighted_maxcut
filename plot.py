#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:55:45 2023

@author: rugantio
"""

import seaborn as sns
import matplotlib.pyplot as plt

# x = range(4,21,2)
# y_perimeter = (0.80, 0.82, 0.86, 0.911, 0.945, 0.908, 0.973, 0.941, 0.927)
# y_star = (1, 1, 1, 1, 1, 1, 1, 1, 1)
# y_star_xy = (1, 0.84, 0.8, 0.678, 0.59,0.64, 
#              0.667, 0.629, 0.65)   

# fig = plt.figure(3, figsize=(10.24, 7.68))
# # title = optimizer + ', n = ' + str(n) + ', depth = ' + str(depth) \
# #     + ', mixer = ' + mixer + ', mu = ' + str(np.round(mu, 3)) + \
# #     ', rnd = ' + str(np.round(rnd_mu, 3))
# # plt.title(title)
# plt.plot(x, y_star, linestyle='None', marker='o', color='black', label=None)
# plt.plot(x, y_star, marker='o', color='black', label='x and r optimizer')
# plt.plot(x, y_star_xy, linestyle='None', marker='o', color='black', label=None)
# plt.plot(x, y_star_xy, marker='o', color='black', label='xy optimizer')
# plt.xlabel("Qubits")
# plt.ylabel("Approx ratio")


# #plt.axhline(0.5, linestyle='-', color='black', label='average')
# #plt.axhline(1, linestyle='--', color='black', label='max')
# plt.xticks(range(2,21,2))#np.arange(0, experiments+1, step=5))

# plt.legend()# plt.yticks(np.arange(0, 1.2, step=0.2))
# plt.grid()
# # plt.plot(x,y)
# fig.savefig("/home/rugantio/Downloads/approx2.svg", bbox_inches='tight', pad_inches=0)
# plt.show()
# plt.clf()


x = range(4,21,2)
y_perimeter = (0.80, 0.82, 0.86, 0.911, 0.945, 0.908, 0.973, 0.941, 0.927)
y_perimeter2 = (1, 0.96, 0.943, 0.933, 0.964, 0.954, 0.987 , 1, 1)
y_star = (1, 1, 1, 1, 1, 1, 1, 1, 1) 

fig = plt.figure(3, figsize=(10.24, 7.68))
# title = optimizer + ', n = ' + str(n) + ', depth = ' + str(depth) \
#     + ', mixer = ' + mixer + ', mu = ' + str(np.round(mu, 3)) + \
#     ', rnd = ' + str(np.round(rnd_mu, 3))
# plt.title(title)
plt.plot(x, y_perimeter, linestyle='None', marker='x', color='#3371ff', label=None)
plt.plot(x, y_perimeter, marker='o', color='#3371ff', label='cycle p=1')
plt.plot(x, y_perimeter2, linestyle='None', marker='x', color='#a3bfff', label=None)
plt.plot(x, y_perimeter2, marker='o', color='#a3bfff', label='cycle p=2')
plt.plot(x, y_star, linestyle='None', marker='o', color='#ff3636', label=None)
plt.plot(x, y_star, marker='o', color='#ff3636', label='star')
plt.xlabel("Qubits")
plt.ylabel("Approx ratio")


#plt.axhline(0.5, linestyle='-', color='black', label='average')
#plt.axhline(1, linestyle='--', color='black', label='max')
plt.xticks(range(2,21,2))#np.arange(0, experiments+1, step=5))

plt.legend()# plt.yticks(np.arange(0, 1.2, step=0.2))
plt.grid()
# plt.plot(x,y)
fig.savefig("/home/rugantio/Downloads/approx.svg", bbox_inches='tight', pad_inches=0)
plt.show()
# plt.clf()