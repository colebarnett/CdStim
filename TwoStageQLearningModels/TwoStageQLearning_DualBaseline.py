# -*- coding: utf-8 -*-
"""
DUAL-ALPHA BASELINE Q-Learning Model
Parameters: pos_alpha, neg_alpha, beta (3 params)
NO stimulation, NO lapse, NO kernel, NO decay, NO adaptive learning

This is the simplest asymmetric learning model.
Tests whether separate learning rates for wins vs losses are needed.
"""

import numpy as np
from scipy import optimize as op


def Calc_DistrQlearning_2Targs_DualBaseline(parameters, Q_initial, chosen_target, rewards, instructed_or_freechoice):
	'''Dual-alpha Q-learning: different learning rates for wins (pos_alpha) vs losses (neg_alpha)'''
	pos_alpha = parameters[0]  # Learning rate for rewards
	neg_alpha = parameters[1]  # Learning rate for no rewards
	beta = parameters[2]       # Inverse temperature

	Q_low = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))
	Q_low[0] = Q_initial[0]
	Q_high[0] = Q_initial[1]

	prob_choice_low = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))
	prob_choice_low[0] = 0.5
	prob_choice_high[0] = 0.5

	log_prob_total = 0.
	accuracy = np.array([])

	for i in range(len(chosen_target)-1):
		# DUAL-ALPHA Q-LEARNING UPDATE
		# Use pos_alpha for rewards, neg_alpha for no rewards
		if chosen_target[i] == 1:  # LV chosen
			delta = float(rewards[i]) - Q_low[i]
			if rewards[i]:
				Q_low[i+1] = Q_low[i] + pos_alpha * delta
			else:
				Q_low[i+1] = Q_low[i] + neg_alpha * delta
			Q_high[i+1] = Q_high[i]  # Unchosen option unchanged
			
		elif chosen_target[i] == 2:  # HV chosen
			delta = float(rewards[i]) - Q_high[i]
			if rewards[i]:
				Q_high[i+1] = Q_high[i] + pos_alpha * delta
			else:
				Q_high[i+1] = Q_high[i] + neg_alpha * delta
			Q_low[i+1] = Q_low[i]  # Unchosen option unchanged

		# Softmax decision rule (free-choice only)
		if instructed_or_freechoice[i+1] == 2:
			prob_choice_low[i+1] = 1./(1. + np.exp(beta*(Q_high[i+1] - Q_low[i+1])))
			prob_choice_high[i+1] = 1. - prob_choice_low[i+1]

			accuracy = np.append(accuracy, (prob_choice_high[i+1] >= 0.5)&(chosen_target[i+1]==2) or 
			                               (prob_choice_high[i+1] < 0.5)&(chosen_target[i+1]==1))
			log_prob_total += np.log(prob_choice_low[i+1]*(chosen_target[i+1]==1) + 
			                         prob_choice_high[i+1]*(chosen_target[i+1]==2))
		else:
			prob_choice_low[i+1] = prob_choice_low[i]
			prob_choice_high[i+1] = prob_choice_high[i]

	return Q_low, Q_high, prob_choice_low, prob_choice_high, accuracy, log_prob_total


def loglikelihood_DistrQlearning_2Targs_DualBaseline(parameters, Q_initial, chosen_target, rewards, instructed_or_freechoice):
	Q_low, Q_high, prob_choice_low, prob_choice_high, accuracy, log_prob_total = \
		Calc_DistrQlearning_2Targs_DualBaseline(parameters, Q_initial, chosen_target, rewards, instructed_or_freechoice)
	return log_prob_total


def DistrQlearning_2Targs_TwoStage_DualBaseline(chosen_target, rewards, instructed_or_freechoice, num_trials_A, num_trials_B):
	"""
	Dual-alpha baseline model - fit Block A and A' separately
	Only 3 parameters: pos_alpha, neg_alpha, beta
	Tests asymmetric learning without other mechanisms
	"""
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
	print("MODEL: Dual-Alpha Baseline")
	print("Asymmetric learning: pos_alpha (wins) ≠ neg_alpha (losses)")
	print("Parameters: pos_alpha, neg_alpha, beta ONLY")
	print("STAGE 1: Fitting Block A")
	print("="*60)
	
	Q_initial = 0.5 * np.ones(2)
	nll = lambda *args: -loglikelihood_DistrQlearning_2Targs_DualBaseline(*args)
	x0 = [0.5, 0.5, 3.0]  # [pos_alpha, neg_alpha, beta]
	
	result_A = op.minimize(nll, x0, 
						   args=(Q_initial, chosen_target_A, rewards_A, instructed_A),
						   bounds=[(0,1), (0,1), (0.1,50)],
						   method='Nelder-Mead')
	pos_alpha_A, neg_alpha_A, beta_A = result_A["x"]
	
	print(f"  pos_alpha: {pos_alpha_A:.4f}, neg_alpha: {neg_alpha_A:.4f}, beta: {beta_A:.4f}")
	
	Q_low_A, Q_high_A, prob_choice_low_A, prob_choice_high_A, accuracy_A, log_prob_A = \
		Calc_DistrQlearning_2Targs_DualBaseline([pos_alpha_A, neg_alpha_A, beta_A], 
											Q_initial, chosen_target_A, rewards_A, instructed_A)
	
	print("STAGE 2: Block B (no fitting)")
	Q_initial_B = np.array([Q_low_A[-1], Q_high_A[-1]])
	Q_low_B, Q_high_B, prob_choice_low_B, prob_choice_high_B, accuracy_B, log_prob_B = \
		Calc_DistrQlearning_2Targs_DualBaseline([pos_alpha_A, neg_alpha_A, beta_A], 
											Q_initial_B, chosen_target_B, rewards_B, instructed_B)
	
	print("STAGE 3: Fitting Block A'")
	Q_initial_Ap = np.array([Q_low_B[-1], Q_high_B[-1]])
	x0_Ap = [pos_alpha_A, neg_alpha_A, beta_A]
	
	result_Ap = op.minimize(nll, x0_Ap,
							args=(Q_initial_Ap, chosen_target_Ap, rewards_Ap, instructed_Ap),
							bounds=[(0,1), (0,1), (0.1,50)],
							method='Nelder-Mead')
	pos_alpha_Ap, neg_alpha_Ap, beta_Ap = result_Ap["x"]
	
	print(f"  pos_alpha: {pos_alpha_Ap:.4f}, neg_alpha: {neg_alpha_Ap:.4f}, beta: {beta_Ap:.4f}")
	print("="*60 + "\n")
	
	Q_low_Ap, Q_high_Ap, prob_choice_low_Ap, prob_choice_high_Ap, accuracy_Ap, log_prob_Ap = \
		Calc_DistrQlearning_2Targs_DualBaseline([pos_alpha_Ap, neg_alpha_Ap, beta_Ap], 
											Q_initial_Ap, chosen_target_Ap, rewards_Ap, instructed_Ap)
	
	Q_low = np.concatenate([Q_low_A, Q_low_B, Q_low_Ap])
	Q_high = np.concatenate([Q_high_A, Q_high_B, Q_high_Ap])
	prob_choice_low = np.concatenate([prob_choice_low_A, prob_choice_low_B, prob_choice_low_Ap])
	prob_choice_high = np.concatenate([prob_choice_high_A, prob_choice_high_B, prob_choice_high_Ap])
	accuracy = np.concatenate([accuracy_A, accuracy_B, accuracy_Ap])
	log_prob_total = log_prob_A + log_prob_B + log_prob_Ap
	
	# Return format: set unused params to None
	return (Q_low, Q_high, prob_choice_low, prob_choice_high, accuracy, log_prob_total,
			pos_alpha_A, neg_alpha_A, None, None, None, beta_A,
			pos_alpha_Ap, neg_alpha_Ap, None, None, None, beta_Ap)
