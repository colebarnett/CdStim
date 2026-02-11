# -*- coding: utf-8 -*-
"""
ValueModelingClass - Clean interface for Q-learning value extraction

This class provides a simple interface for extracting Q-values from HDF behavioral data
using various Q-learning models. Designed for easy integration into analysis pipelines.

Usage:
    from ValueModelingClass import ValueModelingClass
    
    # Get Q-values from HDF file
    vmc = ValueModelingClass()
    value_dict = vmc.get_values(hdf_file, num_trials_A, num_trials_B, method='PersDecay')
    
    # Returns: {'Q_low': array, 'Q_high': array}
    
    # Compare multiple models
    vmc.plot_model_comparison(hdf_file, num_trials_A, num_trials_B, 
                             models=['Baseline', 'PersDecay', 'KernelDecay'])

Author: HeyEVA (for Cole Barnett)
Date: December 2024
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op
from DecisionMakingBehavior import ChoiceBehavior_TwoTargets_Stimulation

# Import all Q-learning model variants
from TwoStageQLearningModels.TwoStageQLearning_Baseline import Calc_DistrQlearning_2Targs_Baseline
from TwoStageQLearningModels.TwoStageQLearning_BaselineDecay import Calc_DistrQlearning_2Targs_BaselineDecay
from TwoStageQLearningModels.TwoStageQLearning_Pers import Calc_DistrQlearning_2Targs_Pers
from TwoStageQLearningModels.TwoStageQLearning_PersDecay import Calc_DistrQlearning_2Targs_PersDecay
from TwoStageQLearningModels.TwoStageQLearning_PersSingle import Calc_DistrQlearning_2Targs_PersSingle
from TwoStageQLearningModels.TwoStageQLearning_PersSingleDecay import Calc_DistrQlearning_2Targs_PersSingleDecay
from TwoStageQLearningModels.TwoStageQLearning_Kernel import Calc_DistrQlearning_2Targs_Kernel
from TwoStageQLearningModels.TwoStageQLearning_KernelDecay import Calc_DistrQlearning_2Targs_KernelDecay
from TwoStageQLearningModels.TwoStageQLearning_KernelSingle import Calc_DistrQlearning_2Targs_KernelSingle
from TwoStageQLearningModels.TwoStageQLearning_KernelSingleDecay import Calc_DistrQlearning_2Targs_KernelSingleDecay
from TwoStageQLearningModels.TwoStageQLearning_AdaptiveOnly import Calc_DistrQlearning_2Targs_AdaptiveOnly
from TwoStageQLearningModels.TwoStageQLearning_AdaptiveDecay import Calc_DistrQlearning_2Targs_AdaptiveDecay
from TwoStageQLearningModels.TwoStageQLearning_PersAdaptive import Calc_DistrQlearning_2Targs_PersAdaptive
from TwoStageQLearningModels.TwoStageQLearning_PersAdaptiveDecay import Calc_DistrQlearning_2Targs_PersAdaptiveDecay
from TwoStageQLearningModels.TwoStageQLearning_KernelAdaptive import Calc_DistrQlearning_2Targs_KernelAdaptive
from TwoStageQLearningModels.TwoStageQLearning_KernelAdaptiveDecay import Calc_DistrQlearning_2Targs_KernelAdaptiveDecay
from TwoStageQLearningModels.TwoStageQLearning_DualBaseline import Calc_DistrQlearning_2Targs_DualBaseline
from TwoStageQLearningModels.TwoStageQLearning_DualBaselineDecay import Calc_DistrQlearning_2Targs_DualBaselineDecay

from TwoStageQLearningModels.TwoStageQLearning_Baseline import loglikelihood_DistrQlearning_2Targs_Baseline
from TwoStageQLearningModels.TwoStageQLearning_BaselineDecay import loglikelihood_DistrQlearning_2Targs_BaselineDecay
from TwoStageQLearningModels.TwoStageQLearning_Pers import loglikelihood_DistrQlearning_2Targs_Pers
from TwoStageQLearningModels.TwoStageQLearning_PersDecay import loglikelihood_DistrQlearning_2Targs_PersDecay
from TwoStageQLearningModels.TwoStageQLearning_PersSingle import loglikelihood_DistrQlearning_2Targs_PersSingle
from TwoStageQLearningModels.TwoStageQLearning_PersSingleDecay import loglikelihood_DistrQlearning_2Targs_PersSingleDecay
from TwoStageQLearningModels.TwoStageQLearning_Kernel import loglikelihood_DistrQlearning_2Targs_Kernel
from TwoStageQLearningModels.TwoStageQLearning_KernelDecay import loglikelihood_DistrQlearning_2Targs_KernelDecay
from TwoStageQLearningModels.TwoStageQLearning_KernelSingle import loglikelihood_DistrQlearning_2Targs_KernelSingle
from TwoStageQLearningModels.TwoStageQLearning_KernelSingleDecay import loglikelihood_DistrQlearning_2Targs_KernelSingleDecay
from TwoStageQLearningModels.TwoStageQLearning_AdaptiveOnly import loglikelihood_DistrQlearning_2Targs_AdaptiveOnly
from TwoStageQLearningModels.TwoStageQLearning_AdaptiveDecay import loglikelihood_DistrQlearning_2Targs_AdaptiveDecay
from TwoStageQLearningModels.TwoStageQLearning_PersAdaptive import loglikelihood_DistrQlearning_2Targs_PersAdaptive
from TwoStageQLearningModels.TwoStageQLearning_PersAdaptiveDecay import loglikelihood_DistrQlearning_2Targs_PersAdaptiveDecay
from TwoStageQLearningModels.TwoStageQLearning_KernelAdaptive import loglikelihood_DistrQlearning_2Targs_KernelAdaptive
from TwoStageQLearningModels.TwoStageQLearning_KernelAdaptiveDecay import loglikelihood_DistrQlearning_2Targs_KernelAdaptiveDecay
from TwoStageQLearningModels.TwoStageQLearning_DualBaseline import loglikelihood_DistrQlearning_2Targs_DualBaseline
from TwoStageQLearningModels.TwoStageQLearning_DualBaselineDecay import loglikelihood_DistrQlearning_2Targs_DualBaselineDecay

class ValueModelingClass:
    """
    Main class for extracting Q-values from HDF behavioral data.
    
    This class wraps the Q-learning model pipeline to provide a clean interface
    for getting Q-values and comparing different models.
    """
    
    def __init__(self):
        """Initialize the ValueModelingClass with available model methods."""
        
        # Dictionary mapping model names to their functions
        self.model_functions = {
            'Baseline': Calc_DistrQlearning_2Targs_Baseline,
            'BaselineDecay': Calc_DistrQlearning_2Targs_BaselineDecay,
            'Pers': Calc_DistrQlearning_2Targs_Pers,
            'PersDecay': Calc_DistrQlearning_2Targs_PersDecay,
            'PersSingle': Calc_DistrQlearning_2Targs_PersSingle,
            'PersSingleDecay': Calc_DistrQlearning_2Targs_PersSingleDecay,
            'Kernel': Calc_DistrQlearning_2Targs_Kernel,
            'KernelDecay': Calc_DistrQlearning_2Targs_KernelDecay,
            'KernelSingle': Calc_DistrQlearning_2Targs_KernelSingle,
            'KernelSingleDecay': Calc_DistrQlearning_2Targs_KernelSingleDecay,
            'AdaptiveOnly': Calc_DistrQlearning_2Targs_AdaptiveOnly,
            'AdaptiveDecay': Calc_DistrQlearning_2Targs_AdaptiveDecay,
            'PersAdaptive': Calc_DistrQlearning_2Targs_PersAdaptive,
            'PersAdaptiveDecay': Calc_DistrQlearning_2Targs_PersAdaptiveDecay,
            'KernelAdaptive': Calc_DistrQlearning_2Targs_KernelAdaptive,
            'KernelAdaptiveDecay': Calc_DistrQlearning_2Targs_KernelAdaptiveDecay,
            'DualBaseline': Calc_DistrQlearning_2Targs_DualBaseline,
            'DualBaselineDecay': Calc_DistrQlearning_2Targs_DualBaselineDecay,
        }
        
        # Dictionary mapping model log likelihood names to their functions
        self.model_ll_functions = {
            'Baseline': loglikelihood_DistrQlearning_2Targs_Baseline,
            'BaselineDecay': loglikelihood_DistrQlearning_2Targs_BaselineDecay,
            'Pers': loglikelihood_DistrQlearning_2Targs_Pers,
            'PersDecay': loglikelihood_DistrQlearning_2Targs_PersDecay,
            'PersSingle': loglikelihood_DistrQlearning_2Targs_PersSingle,
            'PersSingleDecay': loglikelihood_DistrQlearning_2Targs_PersSingleDecay,
            'Kernel': loglikelihood_DistrQlearning_2Targs_Kernel,
            'KernelDecay': loglikelihood_DistrQlearning_2Targs_KernelDecay,
            'KernelSingle': loglikelihood_DistrQlearning_2Targs_KernelSingle,
            'KernelSingleDecay': loglikelihood_DistrQlearning_2Targs_KernelSingleDecay,
            'AdaptiveOnly': loglikelihood_DistrQlearning_2Targs_AdaptiveOnly,
            'AdaptiveDecay': loglikelihood_DistrQlearning_2Targs_AdaptiveDecay,
            'PersAdaptive': loglikelihood_DistrQlearning_2Targs_PersAdaptive,
            'PersAdaptiveDecay': loglikelihood_DistrQlearning_2Targs_PersAdaptiveDecay,
            'KernelAdaptive': loglikelihood_DistrQlearning_2Targs_KernelAdaptive,
            'KernelAdaptiveDecay': loglikelihood_DistrQlearning_2Targs_KernelAdaptiveDecay,
            'DualBaseline': loglikelihood_DistrQlearning_2Targs_DualBaseline,
            'DualBaselineDecay': loglikelihood_DistrQlearning_2Targs_DualBaselineDecay,
        }
        
        # Model display properties for plotting
        self.model_properties = {
            'Baseline': {'color': 'blue', 'linestyle': '-', 'name': 'Baseline'},
            'BaselineDecay': {'color': 'cyan', 'linestyle': '-', 'name': 'Baseline+Decay'},
            'Pers': {'color': 'green', 'linestyle': '-', 'name': 'Perseverance'},
            'PersDecay': {'color': 'lime', 'linestyle': '-', 'name': 'Pers+Decay'},
            'PersSingle': {'color': 'darkgreen', 'linestyle': '-', 'name': 'Pers (Single-α)'},
            'PersSingleDecay': {'color': 'lightgreen', 'linestyle': '-', 'name': 'Pers+Decay (Single-α)'},
            'Kernel': {'color': 'orange', 'linestyle': '-', 'name': 'Choice Kernel'},
            'KernelDecay': {'color': 'red', 'linestyle': '-', 'name': 'Kernel+Decay'},
            'KernelSingle': {'color': 'darkorange', 'linestyle': '-', 'name': 'Kernel (Single-α)'},
            'KernelSingleDecay': {'color': 'salmon', 'linestyle': '-', 'name': 'Kernel+Decay (Single-α)'},
            'AdaptiveOnly': {'color': 'purple', 'linestyle': '--', 'name': 'Adaptive Only'},
            'AdaptiveDecay': {'color': 'violet', 'linestyle': '--', 'name': 'Adaptive+Decay'},
            'PersAdaptive': {'color': 'magenta', 'linestyle': '--', 'name': 'Pers+Adaptive'},
            'PersAdaptiveDecay': {'color': 'orchid', 'linestyle': '--', 'name': 'Pers+Adaptive+Decay'},
            'KernelAdaptive': {'color': 'brown', 'linestyle': '--', 'name': 'Kernel+Adaptive'},
            'KernelAdaptiveDecay': {'color': 'tan', 'linestyle': '--', 'name': 'Kernel+Adaptive+Decay'},
            'DualBaseline': {'color': 'navy', 'linestyle': ':', 'name': 'Dual Baseline'},
            'DualBaselineDecay': {'color': 'skyblue', 'linestyle': ':', 'name': 'Dual Baseline+Decay'},
        }
        
        # # Parameter counts for AIC calculation (two-stage models have params for Block A and A')
        # self.model_param_counts = {
        #     'Baseline': 6,  # 2 alphas + beta for A, same for A'
        #     'BaselineDecay': 8,  # 2 alphas + decay + beta for A, same for A'
        #     'Pers': 8,  # 2 alphas + rho + beta for A, same for A'
        #     'PersDecay': 10,  # 2 alphas + decay + rho + beta for A, same for A'
        #     'PersSingle': 6,  # alpha + rho + beta for A, same for A'
        #     'PersSingleDecay': 8,  # alpha + decay + rho + beta for A, same for A'
        #     'Kernel': 10,  # 2 alphas + ck_weight + ck_decay + beta for A, same for A'
        #     'KernelDecay': 12,  # 2 alphas + decay + ck_weight + ck_decay + beta for A, same for A'
        #     'KernelSingle': 8,  # alpha + ck_weight + ck_decay + beta for A, same for A'
        #     'KernelSingleDecay': 10,  # alpha + decay + ck_weight + ck_decay + beta for A, same for A'
        #     'AdaptiveOnly': 4,  # Just adaptive alpha + beta for A, same for A'
        #     'AdaptiveDecay': 6,  # Adaptive alpha + decay + beta for A, same for A'
        #     'PersAdaptive': 6,  # Adaptive alpha + rho + beta for A, same for A'
        #     'PersAdaptiveDecay': 8,  # Adaptive alpha + decay + rho + beta for A, same for A'
        #     'KernelAdaptive': 8,  # Adaptive alpha + ck_weight + ck_decay + beta for A, same for A'
        #     'KernelAdaptiveDecay': 10,  # Adaptive alpha + decay + ck_weight + ck_decay + beta for A, same for A'
        #     'DualBaseline': 6,  # 2 alphas + beta for A, same for A'
        #     'DualBaselineDecay': 8,  # 2 alphas + decay + beta for A, same for A'
        # }
        
        # Parameters used by each model
        self.model_params = {
            'Baseline': ['alpha','beta'],
            'BaselineDecay': ['alpha','decay','beta'],
            'Pers': ['pos_alpha','neg_alpha','rho','beta'],
            'PersDecay': ['pos_alpha','neg_alpha','decay','rho','beta'],
            'PersSingle': ['alpha','rho','beta'],
            'PersSingleDecay': ['alpha','decay','rho','beta'],
            'Kernel': ['pos_alpha','neg_alpha', 'ck_weight', 'ck_decay', 'beta'],
            'KernelDecay': ['pos_alpha','neg_alpha', 'decay', 'ck_weight', 'ck_decay', 'beta'],
            'KernelSingle': ['alpha', 'ck_weight', 'ck_decay', 'beta'],
            'KernelSingleDecay': ['alpha', 'decay', 'ck_weight', 'ck_decay', 'beta'],
            'AdaptiveOnly': ['alpha','beta'],
            'AdaptiveDecay': ['alpha','decay','beta'],
            'PersAdaptive': ['alpha','rho','beta'],
            'PersAdaptiveDecay': ['alpha','decay','rho','beta'],
            'KernelAdaptive': ['alpha', 'ck_weight', 'ck_decay', 'beta'],
            'KernelAdaptiveDecay': ['alpha', 'decay', 'ck_weight', 'ck_decay', 'beta'],
            'DualBaseline': ['pos_alpha','neg_alpha','beta'],
            'DualBaselineDecay': ['pos_alpha','neg_alpha','decay','beta'],
        }
        
        # Short Description of each model
        self.model_descriptions = {
            'Baseline': "Standard Q-learning: Q(t+1) = Q(t) + α×(r - Q(t))",
            'BaselineDecay': "Standard Q-learning with forgetting",
            'Pers': "Dual-Alpha, No Decay, No Lapse",
            'PersDecay': "Dual-Alpha + Decay, No Lapse",
            'PersSingle': "Single-Alpha, No Decay, No Lapse",
            'PersSingleDecay': 'Single-Alpha + Decay, No Lapse',
            'Kernel': "Dual-Alpha, No Decay, No Lapse",
            'KernelDecay': "Dual-Alpha + Decay, No Lapse",
            'KernelSingle': "Single-Alpha, No Decay, No Lapse",
            'KernelSingleDecay': "Single-Alpha + Decay, No Lapse",
            'AdaptiveOnly': "α_effective = α_base × |prediction_error|",
            'AdaptiveDecay': "α_effective = α_base × |prediction_error|",
            'PersAdaptive': "Adaptive Learning, No Decay, No Lapse",
            'PersAdaptiveDecay': "Adaptive + Decay, No Lapse",
            'KernelAdaptive': "Adaptive Learning, No Decay, No Lapse",
            'KernelAdaptiveDecay': "Adaptive + Decay, No Lapse",
            'DualBaseline': "Asymmetric learning: pos_alpha (wins) ≠ neg_alpha (losses)",
            'DualBaselineDecay': "Asymmetric learning + forgetting",
        }
        
        # Superset of all parameters and initial values used by all models 
        self.master_params_list = ['pos_alpha', 'neg_alpha', 'alpha', 'decay', 'ck_weight', 'ck_decay', 'beta', 'rho']
        self.master_x0 =          [   0.5,         0.5,        0.5,    0.95,       0.0,        0.5,       3.0,     0.0 ]
        self.master_bounds =      [  (0,1),       (0,1),      (0,1),  (0.9,1),   (-2,2),      (0,1),   (0.1,50), (-2,2)]
        
        # Superset of all parameters which will get saved out when the model finishes running
        self.all_model_params_A = {'pos_alpha':None, 'neg_alpha':None, 'alpha':None, 
                              'decay':None, 'ck_weight':None, 'ck_decay':None, 
                              'beta':None, 'rho':None}
        self.all_model_params_Ap = {'pos_alpha':None, 'neg_alpha':None, 'alpha':None, 
                              'decay':None, 'ck_weight':None, 'ck_decay':None, 
                              'beta':None, 'rho':None}

    
    def DistrQlearning_2Targs_TwoStage(self, method, chosen_target, rewards, instructed_or_freechoice, num_trials_A, num_trials_B):
        
        model_func = self.model_functions[method] # calculator function
        model_ll_func = self.model_ll_functions[method] #log likelihood function
        model_func_params = self.model_params[method] #params used by method
        
        """Fit Block A and A' separately"""
        if num_trials_A>0 and num_trials_B>0:
            
            block_A_end = num_trials_A
            block_B_end = num_trials_A + num_trials_B
            
            chosen_target_A = chosen_target[:block_A_end]
            rewards_A = rewards[:block_A_end]
            instructed_A = instructed_or_freechoice[:block_A_end]
            
            chosen_target_B = chosen_target[block_A_end:block_B_end]
            rewards_B = rewards[block_A_end:block_B_end]
            instructed_B = instructed_or_freechoice[block_A_end:block_B_end]
            
            chosen_target_Ap = chosen_target[block_B_end:]
            rewards_Ap = rewards[block_B_end:]
            instructed_Ap = instructed_or_freechoice[block_B_end:]
            
            print("\n" + "="*60)
            print(f"MODEL: {method} ({self.model_descriptions[method]})")
            print("STAGE 1: Fitting Block A")
            print("="*60)
            
            Q_initial = 0.5 * np.ones(2)
            nll = lambda *args: -model_ll_func(*args)
            x0 = [self.master_x0[self.master_params_list.index(model_func_params[i])] for i in range(len(model_func_params))]
            bounds = [self.master_bounds[self.master_params_list.index(model_func_params[i])] for i in range(len(model_func_params))]
            
            result_A = op.minimize(nll, x0, 
                                   args=(Q_initial, chosen_target_A, rewards_A, instructed_A),
                                   bounds=bounds, method='Nelder-Mead')

            [print(f"{model_func_params[i]}: {result_A['x'][i]:.4f}") for i in range(len(x0))]
            
            Q_low_A, Q_high_A, prob_choice_low_A, prob_choice_high_A, accuracy_A, log_prob_A = \
                model_func(result_A["x"], Q_initial, chosen_target_A, rewards_A, instructed_A)
            
            print("STAGE 2: Block B (no fitting)")
            Q_initial_B = np.array([Q_low_A[-1], Q_high_A[-1]])
            Q_low_B, Q_high_B, prob_choice_low_B, prob_choice_high_B, accuracy_B, log_prob_B = \
                model_func(result_A["x"], Q_initial_B, chosen_target_B, rewards_B, instructed_B)
            
            print("STAGE 3: Fitting Block A'")
            Q_initial_Ap = np.array([Q_low_B[-1], Q_high_B[-1]])
            x0_Ap = result_A["x"]
            
            result_Ap = op.minimize(nll, x0_Ap,
                                    args=(Q_initial_Ap, chosen_target_Ap, rewards_Ap, instructed_Ap),
                                    bounds=bounds, method='Nelder-Mead')
            
            [print(f"{model_func_params[i]}: {result_Ap['x'][i]:.4f}") for i in range(len(x0))]
            print("="*60 + "\n")
            
            Q_low_Ap, Q_high_Ap, prob_choice_low_Ap, prob_choice_high_Ap, accuracy_Ap, log_prob_Ap = \
                model_func(result_Ap["x"], Q_initial_Ap, chosen_target_Ap, rewards_Ap, instructed_Ap)
            
            Q_low = np.concatenate([Q_low_A, Q_low_B, Q_low_Ap])
            Q_high = np.concatenate([Q_high_A, Q_high_B, Q_high_Ap])
            prob_choice_low = np.concatenate([prob_choice_low_A, prob_choice_low_B, prob_choice_low_Ap])
            prob_choice_high = np.concatenate([prob_choice_high_A, prob_choice_high_B, prob_choice_high_Ap])
            accuracy = np.concatenate([accuracy_A, accuracy_B, accuracy_Ap])
            log_prob_total = log_prob_A + log_prob_B + log_prob_Ap
        
        
        ##Only fit using one block (for different task structure)##
        elif num_trials_A==0 and num_trials_B==0:
            
            print("\n" + "="*60)
            print(f"MODEL: {method} ({self.model_descriptions[method]})")
            print("Fitting All Trials")
            print("="*60)
            
            Q_initial = 0.5 * np.ones(2)
            nll = lambda *args: -model_ll_func(*args)
            x0 = [self.master_x0[self.master_params_list.index(model_func_params[i])] for i in range(len(model_func_params))]
            bounds = [self.master_bounds[self.master_params_list.index(model_func_params[i])] for i in range(len(model_func_params))]
            
            result = op.minimize(nll, x0, 
                                   args=(Q_initial, chosen_target, rewards, instructed_or_freechoice),
                                   bounds=bounds, method='Nelder-Mead')
            
            [print(f"{model_func_params[i]}: {result['x'][i]:.4f}") for i in range(len(x0))]
            
            Q_low, Q_high, prob_choice_low, prob_choice_high, accuracy, log_prob_total = \
                model_func(result["x"], Q_initial, chosen_target, rewards, instructed_or_freechoice)

            result_A = dict()
            result_Ap = dict()
            result_A["x"], result_Ap["x"] = result["x"], result["x"]


        # organize parameters used into a dictionary
        for param in self.master_params_list:
            if param in model_func_params:
                idx = model_func_params.index(param)
                self.all_model_params_A[param] = result_A["x"][idx]
                self.all_model_params_Ap[param] = result_Ap["x"][idx]
        
        return (Q_low, Q_high, prob_choice_low, prob_choice_high, accuracy, log_prob_total,
             self.all_model_params_A, self.all_model_params_Ap)
    
    
    def get_values(self, hdf_file, num_trials_A, num_trials_B, method='PersSingle'):
        """
        Extract Q-values from HDF file using specified Q-learning model.
        
        Parameters:
        -----------
        hdf_file : str or list of str
            Path to HDF file(s). Can be a single file or list of files for sessions
            split across multiple files.
        num_trials_A : int
            Number of trials in Block A (baseline block)
        num_trials_B : int
            Number of trials in Block B (stimulation block)
        method : str, optional (default='PersDecay')
            Q-learning model to use. Options include:
            - 'Baseline': Basic dual-alpha Q-learning
            - 'BaselineDecay': Baseline + decay/forgetting
            - 'Pers': Baseline + perseverance
            - 'PersDecay': Baseline + perseverance + decay (RECOMMENDED)
            - 'Kernel': Baseline + choice kernel
            - 'KernelDecay': Baseline + choice kernel + decay
            - 'AdaptiveOnly': Adaptive learning rates only
            - 'AdaptiveDecay': Adaptive learning rates + decay
            - 'PersAdaptive': Perseverance + adaptive learning
            - 'PersAdaptiveDecay': Perseverance + adaptive + decay
            And more (see self.model_functions for complete list)
        
        Returns:
        --------
        dict with keys:
            - 'Q_low': numpy array of Q-values for low-value target
            - 'Q_high': numpy array of Q-values for high-value target
            - 'prob_choice_low': numpy array of choice probabilities for low-value target
            - 'prob_choice_high': numpy array of choice probabilities for high-value target
            - 'chosen_target': numpy array of actual choices (1=low, 2=high)
            - 'rewards': numpy array of rewards received (0 or 1)
            - 'num_trials_A': number of Block A trials
            - 'num_trials_B': number of Block B trials
        """
        
        # Validate method
        if method not in self.model_functions:
            raise ValueError(f"Unknown method '{method}'. Available methods: {list(self.model_functions.keys())}")
        
        # Convert single file to list for consistency
        if isinstance(hdf_file, str):
            hdf_files = [hdf_file]
        else:
            hdf_files = hdf_file
        
        # Load behavioral data from HDF file(s)
        behavior = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, num_trials_A, num_trials_B)
        
        # Extract choices, rewards, and trial types
        chosen_target, rewards, instructed_or_freechoice = behavior.GetChoicesAndRewards()
        
        # Check if we got any trials
        if len(chosen_target) == 0:
            raise ValueError(
                f"No trials extracted from HDF file!"
            )
        
        # Run the Q-learning model
        # model_func = self.model_functions[method]
        results = self.DistrQlearning_2Targs_TwoStage(method, chosen_target, rewards, instructed_or_freechoice, 
                           num_trials_A, num_trials_B)
        
        # Unpack results (format varies slightly by model)
        Q_low = results[0]
        Q_high = results[1]
        prob_choice_low = results[2]
        prob_choice_high = results[3]
        accuracy = results[4]
        log_likelihood = results[5]  # Total log-likelihood across all blocks
        
        # Create output dictionary
        value_dict = {
            'Q_low': Q_low,
            'Q_high': Q_high,
            'prob_choice_low': prob_choice_low,
            'prob_choice_high': prob_choice_high,
            'chosen_target': chosen_target,
            'rewards': rewards,
            'instructed_or_freechoice': instructed_or_freechoice,
            'num_trials_A': num_trials_A,
            'num_trials_B': num_trials_B,
            'method': method,
            'log_likelihood': log_likelihood,
            'accuracy': accuracy,
        }
        
        
        print(f"Q_low range: [{np.min(Q_low):.3f}, {np.max(Q_low):.3f}]")
        print(f"Q_high range: [{np.min(Q_high):.3f}, {np.max(Q_high):.3f}]")
        
        
        return value_dict
    
    def plot_model_comparison(self, hdf_file, num_trials_A=100, num_trials_B=100, 
                             models=None, window_size=10, save_path=None):
        """
        Compare multiple Q-learning models against actual behavior WITH AIC comparison.
        
        Parameters:
        -----------
        hdf_file : str or list of str
            Path to HDF file(s)
        num_trials_A : int
            Number of trials in Block A
        num_trials_B : int
            Number of trials in Block B
        models : list of str, optional
            List of model names to compare. If None, uses ALL available models.
        window_size : int, optional (default=10)
            Sliding window size for smoothing actual behavior
        save_path : str, optional
            If provided, saves the figure to this path
        
        Returns:
        --------
        fig, axes : matplotlib figure and axes objects
        aic_values : dict of AIC values for each model
        """
        
        # Default to ALL models if not specified
        if models is None:
            models = list(self.model_functions.keys())
        # Get Q-values for each model
        model_results = {}
        aic_values = {}
        ll_values = {}
        acc_values = {}
        
        for model in models:
            print(f"\nRunning model: {model}")
            model_results[model] = self.get_values(hdf_file, num_trials_A, num_trials_B, method=model)
            
            # Calculate AIC: AIC = 2k - 2*log(L), where k is num_params and L is log-likelihood
            if num_trials_A>0 and num_trials_B>0:
                k = 2*len(self.model_params[model]) #set of params for block A and A', hence the x2
            elif num_trials_A==0 and num_trials_B==0:
                k = len(self.model_params[model])
            log_L = model_results[model]['log_likelihood']
            aic = 2 * k - 2 * log_L
            # aic=-log_L
            aic_values[model] = aic
            ll_values[model] = log_L 
            acc_values[model] = np.mean(model_results[model]['accuracy'])
            
            print(f"  Log-likelihood: {log_L:.2f}")
            print(f"  Parameters: {k}")
            print(f"  AIC: {aic:.2f}")
        
        # Extract actual behavior from first model (same for all)
        first_model = list(model_results.values())[0]
        chosen_target = first_model['chosen_target']
        
        # Convert choices to binary arrays for each target
        # chosen_target: 1=low, 2=high
        actual_choices_low = (chosen_target == 1).astype(float)
        actual_choices_high = (chosen_target == 2).astype(float)
        
        # Extract Block A' only (post-stimulation return)
        block_Ap_start = num_trials_A + num_trials_B
        actual_choices_low_Ap = actual_choices_low[block_Ap_start:]
        actual_choices_high_Ap = actual_choices_high[block_Ap_start:]
        
        # Smooth actual behavior
        behavior_smoothed_low = self._sliding_average(actual_choices_low_Ap, window_size)
        behavior_smoothed_high = self._sliding_average(actual_choices_high_Ap, window_size)
        
        # Create figure with THREE subplots
        fig = plt.figure(figsize=(32, 7))
        ax1 = plt.subplot(1, 4, 1)
        ax2 = plt.subplot(1, 4, 2)
        ax3 = plt.subplot(1, 4, 3)
        ax4 = plt.subplot(1, 4, 4)
        
        trials = np.arange(len(actual_choices_low_Ap))
        
        
        # ========== HIGH VALUE TARGET ==========
        ax=ax1
        # Plot actual behavior (RED)
        ax.plot(trials, behavior_smoothed_high, 'r-', linewidth=3.0, label='Actual Behavior', zorder=10)
        
        # Plot each model
        for model in models:
            if model in model_results:
                prob_high = model_results[model]['prob_choice_high'][block_Ap_start:]
                
                # Get model properties or use defaults
                props = self.model_properties.get(model,
                    {'color': 'gray', 'linestyle': '-', 'name': model})
                
                ax.plot(trials, prob_high,
                        color=props['color'],
                        linestyle=props['linestyle'],
                        linewidth=2.0,
                        # label=props['name'],
                        label='',
                        alpha=0.8)
        
        ax.set_xlabel('Trial Number (Block A\' only)', fontsize=14)
        ax.set_ylabel('P(Choose High Value Target)', fontsize=14)
        ax.set_title('High Value Target Selection', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        
        # ========== ACCURACY COMPARISON ==========
        ax=ax2
        # Sort models by accuracy (hgiher is better)
        sorted_models = sorted(acc_values.items(), key=lambda x: x[1], reverse=True)
        model_names = [self.model_properties.get(m, {'name': m})['name'] for m, _ in sorted_models]
        acc_vals = [acc for _, acc in sorted_models]
        
        # Create bar chart
        bars = ax.barh(range(len(model_names)), acc_vals, 
                       color=[self.model_properties.get(m, {'color': 'gray'})['color'] 
                             for m, _ in sorted_models],
                       alpha=0.7)
        
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=10)
        ax.set_xlabel('Accuracy (higher is better)', fontsize=14)
        ax.set_title('Model Comparison (Accuracy)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Highlight best model (highest accuracy)
        best_idx = 0
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(3)
        
        # Add Acc values as text
        for i, (model, acc) in enumerate(sorted_models):
            ax.text(acc + (max(acc_vals) - min(acc_vals)) * 0.01, i, 
                    f'{acc:.3f}', 
                    va='center', fontsize=9)
            
        # ========== LOG LIKELIHOOD COMPARISON ==========
        ax=ax3
        # Sort models by log likelihood (hgiher is better)
        sorted_models = sorted(ll_values.items(), key=lambda x: x[1], reverse=True)
        model_names = [self.model_properties.get(m, {'name': m})['name'] for m, _ in sorted_models]
        ll_vals = [ll for _, ll in sorted_models]
        
        # Create bar chart
        bars = ax.barh(range(len(model_names)), ll_vals, 
                       color=[self.model_properties.get(m, {'color': 'gray'})['color'] 
                             for m, _ in sorted_models],
                       alpha=0.7)
        
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=10)
        ax.set_xlabel('Log Likelihood (higher is better)', fontsize=14)
        ax.set_title('Model Comparison (Log Likelihood)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Highlight best model (highest ll)
        best_idx = 0
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(3)
        
        # Add ll values as text
        for i, (model, ll) in enumerate(sorted_models):
            ax.text(ll + (max(ll_vals) - min(ll_vals)) * 0.01, i, 
                    f'{ll:.1f}', 
                    va='center', fontsize=9)

            
        # ========== AIC COMPARISON ==========
        ax=ax4
        # Sort models by aic (lower is better)
        sorted_models = sorted(aic_values.items(), key=lambda x: x[1])
        model_names = [self.model_properties.get(m, {'name': m})['name'] for m, _ in sorted_models]
        aic_vals = [aic for _, aic in sorted_models]
        
        # Create bar chart
        bars = ax.barh(range(len(model_names)), aic_vals, 
                       color=[self.model_properties.get(m, {'color': 'gray'})['color'] 
                             for m, _ in sorted_models],
                       alpha=0.7)
        
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=10)
        ax.set_xlabel('AIC (lower is better)', fontsize=14)
        ax.set_title('Model Comparison (AIC)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Highlight best model (lowest AIC)
        best_idx = 0
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(3)
        
        # Add AIC values as text
        for i, (model, aic) in enumerate(sorted_models):
            ax.text(aic + (max(aic_vals) - min(aic_vals)) * 0.01, i, 
                    f'{aic:.1f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n{'='*60}")
            print(f"Figure saved to: {save_path}")
            print(f"{'='*60}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Best model (lowest AIC): {sorted_models[0][0]} (AIC = {sorted_models[0][1]:.2f})")
        print(f"Worst model (highest AIC): {sorted_models[-1][0]} (AIC = {sorted_models[-1][1]:.2f})")
        print(f"{'='*60}\n")
        
        return fig, ax1,ax2,ax3,ax4, aic_values, ll_values, acc_values
    
    def _sliding_average(self, data, window):
        """Apply sliding window average for smoothing."""
        smoothed = np.zeros(len(data))
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            smoothed[i] = np.mean(data[start:end])
        return smoothed
    
    # def _get_num_parameters(self, model_name):
    #     """
    #     Get the number of free parameters for each model.
        
    #     Two-stage models fit Block A and A' separately, so they have
    #     roughly 2x the parameters (except beta which may be shared).
        
    #     Returns:
    #     --------
    #     int : number of free parameters
    #     """
    #     # Parameter counts for each model
    #     # Format: (params_per_block, shared_across_blocks)
    #     # Two-stage models fit A and A' separately
    #     param_counts = {
    #         # Baseline models: pos_alpha, neg_alpha, beta per block
    #         'Baseline': 3 * 2,  # 6 total (A and A' fitted separately)
    #         'BaselineDecay': 4 * 2,  # pos_alpha, neg_alpha, decay, beta per block
            
    #         # Perseverance models: add rho parameter
    #         'Pers': 4 * 2,  # pos_alpha, neg_alpha, rho, beta per block
    #         'PersDecay': 5 * 2,  # pos_alpha, neg_alpha, decay, rho, beta per block
    #         'PersSingle': 3 * 2,  # alpha, rho, beta per block (single alpha)
    #         'PersSingleDecay': 4 * 2,  # alpha, decay, rho, beta per block
            
    #         # Choice kernel models: add ck_weight, ck_decay
    #         'Kernel': 5 * 2,  # pos_alpha, neg_alpha, ck_weight, ck_decay, beta per block
    #         'KernelDecay': 6 * 2,  # pos_alpha, neg_alpha, decay, ck_weight, ck_decay, beta
    #         'KernelSingle': 4 * 2,  # alpha, ck_weight, ck_decay, beta per block
    #         'KernelSingleDecay': 5 * 2,  # alpha, decay, ck_weight, ck_decay, beta
            
    #         # Adaptive models: adaptive learning rates
    #         'AdaptiveOnly': 4 * 2,  # pos_alpha, neg_alpha, adaptive_param, beta per block
    #         'AdaptiveDecay': 5 * 2,  # pos_alpha, neg_alpha, decay, adaptive_param, beta
            
    #         # Combined models
    #         'PersAdaptive': 5 * 2,  # pos_alpha, neg_alpha, rho, adaptive_param, beta
    #         'PersAdaptiveDecay': 6 * 2,  # pos_alpha, neg_alpha, decay, rho, adaptive_param, beta
    #         'KernelAdaptive': 6 * 2,  # pos_alpha, neg_alpha, ck_weight, ck_decay, adaptive_param, beta
    #         'KernelAdaptiveDecay': 7 * 2,  # all of the above + decay
            
    #         # Dual baseline models
    #         'DualBaseline': 3 * 2,  # Same as Baseline
    #         'DualBaselineDecay': 4 * 2,  # Same as BaselineDecay
    #     }
        
    #     return param_counts.get(model_name, 6)  # Default to 6 if unknown
    
    # def plot_aic_comparison(self, hdf_file, num_trials_A, num_trials_B, 
    #                        models=None, save_path=None):
    #     """
    #     Create AIC comparison bar chart across different Q-learning models.
    #     AIC = 2k - 2·log(L)
    #     where k = number of parameters, L = likelihood
        
    #     Parameters:
    #     -----------
    #     hdf_file : str or list of str
    #         Path to HDF file(s)
    #     num_trials_A : int
    #         Number of trials in Block A
    #     num_trials_B : int
    #         Number of trials in Block B
    #     models : list of str, optional
    #         List of model names to compare. If None, uses all available models.
    #     save_path : str, optional
    #         If provided, saves the figure to this path
        
    #     Returns:
    #     --------
    #     fig, ax : matplotlib figure and axes objects
    #     aic_values : dict
    #         Dictionary mapping model names to their AIC values
    #     """
        
    #     if models is None:
    #         models = list(self.model_functions.keys())
    #     print(f"\n{'='*60}")
    #     print(f"Computing AIC for {len(models)} models...")
    #     print(f"{'='*60}")
        
    #     # Get results for each model
    #     aic_values = {}
    #     bic_values = {}
    #     log_likelihoods = {}
        
    #     for model in models:
    #         print(f"\nRunning model: {model}")
            
    #         # Get model results
    #         result = self.get_values(hdf_file, num_trials_A, num_trials_B, method=model)
            
    #         # Extract log likelihood from the full model results
    #         # We need to re-run to get the log_prob_total
    #         if isinstance(hdf_file, str):
    #             hdf_files = [hdf_file]
    #         else:
    #             hdf_files = hdf_file
            
    #         behavior = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, num_trials_A, num_trials_B)
    #         chosen_target, rewards, instructed_or_freechoice = behavior.GetChoicesAndRewards()
            
    #         model_func = self.model_functions[model]
    #         results = model_func(chosen_target, rewards, instructed_or_freechoice, 
    #                            num_trials_A, num_trials_B)
            
    #         # Extract log likelihood (usually index 5 in results tuple)
    #         log_likelihood = results[5]
            
    #         # Get number of parameters
    #         k = self._get_num_parameters(model)
            
    #         # Calculate AIC and BIC
    #         # AIC = 2k - 2·log(L)
    #         aic = 2 * k - 2 * log_likelihood
            
    #         # BIC = k·log(n) - 2·log(L)
    #         n = len(chosen_target)
    #         bic = k * np.log(n) - 2 * log_likelihood
            
    #         aic_values[model] = aic
    #         bic_values[model] = bic
    #         log_likelihoods[model] = log_likelihood
            
    #         print(f"  Log-likelihood: {log_likelihood:.2f}")
    #         print(f"  Parameters: {k}")
    #         print(f"  AIC: {aic:.2f}")
    #         print(f"  BIC: {bic:.2f}")
        
    #     # Create bar chart
    #     fig, ax = plt.subplots(figsize=(12, 6))
        
    #     # Sort models by AIC (best to worst)
    #     sorted_models = sorted(models, key=lambda m: aic_values[m])
    #     sorted_aic = [aic_values[m] for m in sorted_models]
        
    #     # Get colors from model properties
    #     colors = [self.model_properties.get(m, {'color': 'gray'})['color'] 
    #              for m in sorted_models]
        
    #     # Create bar chart
    #     x_pos = np.arange(len(sorted_models))
    #     bars = ax.bar(x_pos, sorted_aic, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
    #     # Highlight the best model (lowest AIC)
    #     bars[0].set_edgecolor('gold')
    #     bars[0].set_linewidth(3)
        
    #     # Add value labels on bars
    #     for i, (bar, aic) in enumerate(zip(bars, sorted_aic)):
    #         height = bar.get_height()
    #         ax.text(bar.get_x() + bar.get_width()/2., height,
    #                f'{aic:.1f}',
    #                ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    #     # Formatting
    #     ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    #     ax.set_ylabel('AIC (lower is better)', fontsize=14, fontweight='bold')
    #     ax.set_title('Model Comparison: Akaike Information Criterion', 
    #                 fontsize=16, fontweight='bold')
    #     ax.set_xticks(x_pos)
    #     ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=11)
    #     ax.grid(axis='y', alpha=0.3)
        
    #     # Add text box with best model
    #     best_model = sorted_models[0]
    #     best_aic = sorted_aic[0]
    #     textstr = f'Best Model: {best_model}\nAIC = {best_aic:.2f}'
    #     props = dict(boxstyle='round', facecolor='gold', alpha=0.3)
    #     ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
    #            verticalalignment='top', bbox=props, fontweight='bold')
        
    #     plt.tight_layout()
        
    #     # Save if requested
    #     if save_path:
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #         print(f"\nAIC comparison plot saved to: {save_path}")
        
    #     print(f"\n{'='*60}")
    #     print(f"Best model: {best_model} (AIC = {best_aic:.2f})")
    #     print(f"{'='*60}\n")
        
    #     return fig, ax, aic_values

if __name__ == "__main__":

    # Example: Extract Q-values from a session
    vmc = ValueModelingClass()
    
    # hdf_file = '/Users/rishichapati/Documents/SantaCruzLab/RishiMacData/Luigi/hdf/luig20170927_07_te361.hdf'
    hdf_file = r"C:\Users\coleb\Desktop\Santacruz Lab\Value Stimulation\luig20170927_07_te361.hdf"
    num_trials_A = 100
    num_trials_B = 100
    
    # hdf_file = r"C:\Users\coleb\Desktop\Santacruz Lab\Whitehall\Analysis\airp20250919_02_te2177\airp20250919_02_te2177.hdf"
    # num_trials_A = 0
    # num_trials_B = 0
    
    value_dict = vmc.get_values(hdf_file, num_trials_A, num_trials_B, method='PersDecay')
    print(f"\nQ-values extracted! Shape: {value_dict['Q_low'].shape}")
    
    vmc.plot_model_comparison(
        hdf_file, num_trials_A, num_trials_B,
        
        save_path='model_comparison.png'
    )
    plt.show()
    
    

