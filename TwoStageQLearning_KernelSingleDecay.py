# -*- coding: utf-8 -*-
"""
Choice Kernel + Decay Model - SINGLE-ALPHA, WITH DECAY, NO LAPSE
Parameters: alpha, decay, ck_weight, ck_decay, beta
Fits Block A and A' separately (different parameters for each)
"""

import numpy as np
from scipy import optimize as op


def Calc_DistrQlearning_2Targs_KernelSingleDecay(parameters, Q_initial, chosen_target, rewards, instructed_or_freechoice):
	'''Single-alpha Q-learning with choice kernel and decay, no lapse'''
	alpha = parameters[0]
	decay = parameters[1]
	ck_weight = parameters[2]
	ck_decay = parameters[3]
	beta = parameters[4]

	Q_low = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))
	Q_low[0] = Q_initial[0]
	Q_high[0] = Q_initial[1]

	CK_low = np.zeros(len(chosen_target))
	CK_high = np.zeros(len(chosen_target))

	prob_choice_low = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))
	prob_choice_low[0] = 0.5
	prob_choice_high[0] = 0.5

	log_prob_total = 0.
	accuracy = np.array([])

	for i in range(len(chosen_target)-1):
		# DECAY/FORGETTING
		Q_low[i] = decay * Q_low[i] + (1 - decay) * 0.5
		Q_high[i] = decay * Q_high[i] + (1 - decay) * 0.5
		
		# Update choice kernel
		CK_low[i+1] = ck_decay * CK_low[i] + (1 - ck_decay) * float(chosen_target[i]==1)
		CK_high[i+1] = ck_decay * CK_high[i] + (1 - ck_decay) * float(chosen_target[i]==2)
		
		# SINGLE-ALPHA Q-LEARNING UPDATE
		if chosen_target[i] == 1:
			delta = float(rewards[i]) - Q_low[i]
			Q_low[i+1] = Q_low[i] + alpha * delta
			Q_high[i+1] = Q_high[i]
		elif chosen_target[i] == 2:
			delta = float(rewards[i]) - Q_high[i]
			Q_high[i+1] = Q_high[i] + alpha * delta
			Q_low[i+1] = Q_low[i]

		if instructed_or_freechoice[i+1] == 2:
			ck_bonus = ck_weight * (CK_high[i+1] - CK_low[i+1])
			prob_choice_low[i+1] = 1./(1. + np.exp(beta*(Q_high[i+1] - Q_low[i+1]) + beta*ck_bonus))
			prob_choice_high[i+1] = 1. - prob_choice_low[i+1]

			accuracy = np.append(accuracy, (prob_choice_high[i+1] >= 0.5)&(chosen_target[i+1]==2) or 
			                               (prob_choice_high[i+1] < 0.5)&(chosen_target[i+1]==1))
			log_prob_total += np.log(prob_choice_low[i+1]*(chosen_target[i+1]==1) + 
			                         prob_choice_high[i+1]*(chosen_target[i+1]==2))
		else:
			prob_choice_low[i+1] = prob_choice_low[i]
			prob_choice_high[i+1] = prob_choice_high[i]

	return Q_low, Q_high, prob_choice_low, prob_choice_high, accuracy, log_prob_total


def loglikelihood_DistrQlearning_2Targs_KernelSingleDecay(parameters, Q_initial, chosen_target, rewards, instructed_or_freechoice):
	Q_low, Q_high, prob_choice_low, prob_choice_high, accuracy, log_prob_total = \
		Calc_DistrQlearning_2Targs_KernelSingleDecay(parameters, Q_initial, chosen_target, rewards, instructed_or_freechoice)
	return log_prob_total


def DistrQlearning_2Targs_TwoStage_KernelSingleDecay(chosen_target, rewards, instructed_or_freechoice, num_trials_A, num_trials_B):
	"""Fit Block A and A' separately"""
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
	print("MODEL: KernelSingleDecay (Single-Alpha + Decay, No Lapse)")
	print("STAGE 1: Fitting Block A")
	print("="*60)
	
	Q_initial = 0.5 * np.ones(2)
	nll = lambda *args: -loglikelihood_DistrQlearning_2Targs_KernelSingleDecay(*args)
	x0 = [0.5, 0.95, 0.0, 0.5, 3.0]
	
	result_A = op.minimize(nll, x0, 
						   args=(Q_initial, chosen_target_A, rewards_A, instructed_A),
						   bounds=[(0,1), (0.9,1), (-2,2), (0,1), (0.1,50)],
						   method='Nelder-Mead')
	alpha_A, decay_A, ck_weight_A, ck_decay_A, beta_A = result_A["x"]
	
	print(f"  alpha: {alpha_A:.4f}, decay: {decay_A:.4f}")
	print(f"  ck_weight: {ck_weight_A:.4f}, ck_decay: {ck_decay_A:.4f}, beta: {beta_A:.4f}")
	
	Q_low_A, Q_high_A, prob_choice_low_A, prob_choice_high_A, accuracy_A, log_prob_A = \
		Calc_DistrQlearning_2Targs_KernelSingleDecay([alpha_A, decay_A, ck_weight_A, ck_decay_A, beta_A], 
											Q_initial, chosen_target_A, rewards_A, instructed_A)
	
	print("STAGE 2: Block B (no fitting)")
	Q_initial_B = np.array([Q_low_A[-1], Q_high_A[-1]])
	Q_low_B, Q_high_B, prob_choice_low_B, prob_choice_high_B, accuracy_B, log_prob_B = \
		Calc_DistrQlearning_2Targs_KernelSingleDecay([alpha_A, decay_A, ck_weight_A, ck_decay_A, beta_A], 
											Q_initial_B, chosen_target_B, rewards_B, instructed_B)
	
	print("STAGE 3: Fitting Block A'")
	Q_initial_Ap = np.array([Q_low_B[-1], Q_high_B[-1]])
	x0_Ap = [alpha_A, decay_A, ck_weight_A, ck_decay_A, beta_A]
	
	result_Ap = op.minimize(nll, x0_Ap,
							args=(Q_initial_Ap, chosen_target_Ap, rewards_Ap, instructed_Ap),
							bounds=[(0,1), (0.9,1), (-2,2), (0,1), (0.1,50)],
							method='Nelder-Mead')
	alpha_Ap, decay_Ap, ck_weight_Ap, ck_decay_Ap, beta_Ap = result_Ap["x"]
	
	print(f"  alpha: {alpha_Ap:.4f}, decay: {decay_Ap:.4f}")
	print(f"  ck_weight: {ck_weight_Ap:.4f}, ck_decay: {ck_decay_Ap:.4f}, beta: {beta_Ap:.4f}")
	print("="*60 + "\n")
	
	Q_low_Ap, Q_high_Ap, prob_choice_low_Ap, prob_choice_high_Ap, accuracy_Ap, log_prob_Ap = \
		Calc_DistrQlearning_2Targs_KernelSingleDecay([alpha_Ap, decay_Ap, ck_weight_Ap, ck_decay_Ap, beta_Ap], 
											Q_initial_Ap, chosen_target_Ap, rewards_Ap, instructed_Ap)
	
	Q_low = np.concatenate([Q_low_A, Q_low_B, Q_low_Ap])
	Q_high = np.concatenate([Q_high_A, Q_high_B, Q_high_Ap])
	prob_choice_low = np.concatenate([prob_choice_low_A, prob_choice_low_B, prob_choice_low_Ap])
	prob_choice_high = np.concatenate([prob_choice_high_A, prob_choice_high_B, prob_choice_high_Ap])
	accuracy = np.concatenate([accuracy_A, accuracy_B, accuracy_Ap])
	log_prob_total = log_prob_A + log_prob_B + log_prob_Ap
	
	return (Q_low, Q_high, prob_choice_low, prob_choice_high, accuracy, log_prob_total,
			alpha_A, alpha_A, decay_A, ck_weight_A, ck_decay_A, None, beta_A,
			alpha_Ap, alpha_Ap, decay_Ap, ck_weight_Ap, ck_decay_Ap, None, beta_Ap)
