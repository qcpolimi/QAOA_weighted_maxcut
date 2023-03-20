# %cd Downloads/QAOA_weighted_maxcut
import numpy as np
import pandas as pd

import qiskit
#from qiskit.circuit.library import XXPlusYYGate  # NOT WORKING ON GPU
from qiskit import Aer, QuantumCircuit
from qiskit.algorithms.optimizers import ADAM, CG, COBYLA, L_BFGS_B, SLSQP, TNC, BOBYQA, IMFIL, GSLS, NELDER_MEAD, NFT, POWELL, SPSA, CRS, DIRECT_L, DIRECT_L_RAND, ESCH, ISRES, QNSPSA

from time import time


pd.set_option('display.max_columns', None)


class QAOA():
    def __init__(self, Q):
        self.Q = self.check_Q_matrix(Q)
        self.n_qubs = len(Q)
        self.states_and_costs = self.eval_states()
        self.states_reversed = [s[::-1] for s in sorted(self.states_and_costs.keys())]

        self.optimum_cost = min(self.states_and_costs.values())
        self.costs = np.array(list(self.states_and_costs.values()))
        self.optimum_states = [
            k for k, v in self.states_and_costs.items() if v == self.optimum_cost]
        self.backend = Aer.get_backend('statevector_simulator')
        self.intermediate_df = pd.DataFrame(columns=['mean_cost', 'probs_dict', 'params'])
        self.final_df = pd.DataFrame(columns = ['n_qubs', 'p', 'opt_name', 'opt_iterations', 'opt_time', 'final_sols', 'opt_sol', 'approx_ratio', 'approx_ratios', 'bit_diffs'])


    @staticmethod
    def check_Q_matrix(Q):
        assert Q.shape[0] == Q.shape[1], "The Q matrix needs to be squared"
        assert np.allclose(Q, Q.T), "The Q matrix needs to be symmetric"
        return Q


    def get_cost_from_state(self, state):
        x = np.array(list(state), dtype=np.byte).reshape((self.n_qubs, 1))
        cost = x.T @ self.Q @ x
        cost = cost
        return cost[0][0]


    def eval_states(self):
        basis_states = dict()
        for i in range(2 ** self.n_qubs):
            state = np.binary_repr(i, width=self.n_qubs)
            basis_states[state] = self.get_cost_from_state(state)
        return basis_states


    def get_cost_circuit(self, gamma):
        qc_cost = QuantumCircuit(self.n_qubs)
        for i in range(self.n_qubs):
            for j in range(i+1, self.n_qubs):
                if self.Q[i, j] != 0:
                    qc_cost.rzz(gamma * 0.5 * self.Q[i, j], i, j)
        return qc_cost
 

    def get_mixer_circuit_one(self, beta):
        qc_mixer = QuantumCircuit(self.n_qubs)
        if self.mixer == 'x':
            for i in range(self.n_qubs):
                qc_mixer.rx(2*beta, i)
        elif self.mixer == 'xy':
            for i in range(self.n_qubs):
                for j in range(i+1,self.n_qubs):
                    if self.Q[i,j] != 0:
                        qc_mixer.rxx(2*beta, i, j)
                        qc_mixer.ryy(2*beta, i, j)
        return qc_mixer

    
    def get_mixer_circuit_two(self, beta, beta2):
        qc_mixer = QuantumCircuit(self.n_qubs)
        for i in range(self.n_qubs):
            qc_mixer.r(2*beta, 2*beta2, i)
        return qc_mixer


    def get_qaoa_circuit(self, params):
        gammas = params[:self.p]
        betas = params[self.p:2*self.p]
        
        qc = QuantumCircuit(self.n_qubs)
        qc.h(range(self.n_qubs))
        
        if self.mixer == 'r':
            betas2 = params[2*self.p:3*self.p]
            for i in range(self.p):
                qc.compose(self.get_cost_circuit(gammas[i]), inplace=True)
                qc.compose(self.get_mixer_circuit_two(betas[i], betas2[i]), inplace=True)
        else:
            for i in range(self.p):
                qc.compose(self.get_cost_circuit(gammas[i]), inplace=True)
                qc.compose(self.get_mixer_circuit_one(betas[i]), inplace=True)
        return qc


    def get_expected_cost_from_probs(self, probs):
        cost = 0
        for state, prob in probs.items():
            cost += self.states_and_costs[state] * prob
        return cost


    # def save_intermediate_df(self, mean_cost, probs_dict, params):
    #     df = pd.DataFrame([[mean_cost, probs_dict, params]], columns=[
    #                       'mean_cost', 'probs_dict', 'params'])
    #     self.intermediate_df = pd.concat([self.intermediate_df, df], ignore_index=True)


    def evaluate_circuit(self, params):
        qc = self.get_qaoa_circuit(params)
        res = self.backend.run(qc).result()
        output_state = res.get_statevector()
        output_probs = np.abs(np.array(output_state))**2
        probs_dict = {k: v for k, v in zip(self.states_reversed, output_probs)}
        mean_cost = self.get_expected_cost_from_probs(probs_dict)
        self.intermediate_df = pd.DataFrame([[mean_cost, probs_dict, params]], columns=[
                          'mean_cost', 'probs_dict', 'params'])
        # self.save_intermediate_df(mean_cost, probs_dict, params)
        self.opt_iterations += 1
        return mean_cost


    def get_gamma_max(self):
        return 8 * np.pi / self.Q.max()


    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        qiskit.utils.algorithm_globals.random_seed = seed
        return seed


    def set_optimizer(self, optimizer):
        # ok ADAM, CG, COBYLA, L_BFGS_B, SLSQP, TNC
        # slow GSLS, NELDER_MEAD, NFT, POWELL, SPSA
        # not working AQGD, GradientDescent, P_BFGS, QNSPSA
        # slow BOBYQA, not working IMFIL, SNOBFIT
        # slow CRS, DIRECT_L, DIRECT_L_RAND, ESCH, ISRES
        
        #UMDA 
        supported_opt = ['ADAM', 
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
                         'ISRES']
        if isinstance(optimizer, str):
            if optimizer.upper() in supported_opt:
                return eval(optimizer.upper()+'()')
            else:
                raise ValueError('optimizer not recognized')
        else:
            return optimizer


    def run_qaoa(self,
                 mixer = 'xy',
                 p = 1,
                 optimizer = 'COBYLA',
                 seed = None,
                 initial_params = None,
                 GPU = True):
        
        if GPU:
            self.backend.set_options(device='GPU')
            
        self.mixer = mixer.lower()
        self.opt_iterations = 0
        self.seed = self.set_seed(seed)
        self.p = p
        opt = self.set_optimizer(optimizer)

        gamma_max = self.get_gamma_max()
        beta_max = np.pi
        gamma_bound = [(0, gamma_max) for _ in range(p)]
        beta_bound = [(0, beta_max) for _ in range(p)]

        if initial_params is None:
            gammas = np.random.uniform(0.0, gamma_max, size=self.p)
            betas = np.random.uniform(0.0, beta_max, size=self.p)
            if self.mixer == 'r':
                betas2 = np.random.uniform(0.0, beta_max, size=self.p)
                bounds = gamma_bound + beta_bound + beta_bound
                initial_params = np.concatenate((gammas, betas, betas2))
            else:
                bounds = gamma_bound + beta_bound
                initial_params = np.concatenate((gammas, betas))

        start = time()
        res = opt.minimize(fun = self.evaluate_circuit,
                           x0 = initial_params,
                           bounds = bounds)
        end = time()
        opt_time = np.round(end-start,3)
        
        if self.opt_iterations > 998:
            print(f"[WARNING] OPTIMIZATION TERMINATED DUE TO MAX_ITER: {self.opt_iterations}")
        
        final_probs = self.intermediate_df.probs_dict.values[-1]

        max_prob = max(final_probs.values())
        final_states = [k for k, v in final_probs.items() if np.isclose(v,max_prob,rtol=0,atol=1e-4)]
        final_costs = [self.states_and_costs[final_state] for final_state in final_states]
        bit_diffs = [self.get_hamming_distance_from_optimum(final_state) for final_state in final_states]
        approx_ratio = np.round(np.array(final_costs).mean() / self.optimum_cost, 3)
        approx_ratios = [np.round(final_cost / self.optimum_cost, 3) for final_cost in final_costs]
        
        if not isinstance(optimizer, str):
            optimizer = repr(optimizer).split()[0].split('.')[-1]
        final_df = pd.DataFrame([[self.n_qubs, self.p, optimizer, self.opt_iterations, opt_time, final_states, self.optimum_states, approx_ratio, approx_ratios, bit_diffs]],
                                columns=['n_qubs', 'p', 'opt_name', 'opt_iterations', 'opt_time', 'final_sols', 'opt_sol', 'approx_ratio', 'approx_ratios', 'bit_diffs'])

        self.final_df = pd.concat([self.final_df, final_df], ignore_index=True)


    @staticmethod
    def hamming_distance(s1, s2):
        s1 = np.array([int(i) for i in s1])
        s2 = np.array([int(i) for i in s2])
        return np.sum(np.bitwise_xor(s1, s2))


    def get_hamming_distance_from_optimum(self, state):
        hamming_distances = [self.hamming_distance(
            state, opt_state) for opt_state in self.optimum_states]
        return min(hamming_distances)


    def save_final_df(self, folder, g, optimizer, n, p, experiments, mixer):
        filename = folder + g + '_' + optimizer + '_' + mixer + '_n=' + str(n) + '_p=' + str(p) + '.csv' #'_exp=' + str(experiments) + '.csv'
        self.final_df.to_csv(filename)
    
    
    def reset_df(self):
        self.intermediate_df = pd.DataFrame(columns=['mean_cost', 'probs_dict', 'params'])
        self.final_df = pd.DataFrame(columns = ['n_qubs', 'p', 'opt_name', 'opt_iterations', 'opt_time', 'final_sols', 'opt_sol', 'approx_ratio', 'approx_ratios', 'bit_diffs'])

        
