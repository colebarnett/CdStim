# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:07:55 2023

@author: coleb
"""
import tables
# import os
# import sys
# import time
# import warnings
import matplotlib.pyplot as plt
import numpy as np
# import tdt
import pandas as pd
# from matplotlib.patches import Ellipse
import scipy as sp
spect = sp.signal.spectrogram
zscr = sp.stats.zscore
from scipy.interpolate import make_interp_spline as spline
from scipy import signal
from scipy.ndimage import filters
from scipy import optimize as op
import statsmodels.api as sm
# from logLikelihoodRLPerformance import logLikelihoodRLPerformance, RLPerformance
# from statsmodels.multivariate.manova import MANOVA


def trial_sliding_avg(trial_array, num_trials_slide):
	'''
	This method performs a simple sliding average of trial_array using a window
	of length num_trials_slide.
	'''
	num_trials = len(trial_array)
	slide_avg = np.zeros(num_trials)

	for i in range(num_trials):
		if i < num_trials_slide:
			slide_avg[i] = np.sum(trial_array[:i+1])/float(i+1)
		else:
			slide_avg[i] = np.sum(trial_array[i-num_trials_slide+1:i+1])/float(num_trials_slide)

	return slide_avg
	
	

def get_HDFstate_TDT_LFPsamples(ind_state,state_time,syncHDF_file):
	'''
	This method finds the TDT sample numbers that correspond to indicated task state using the syncHDF.mat file.

	Inputs:
		- ind_state: array with state numbers corresponding to which state we're interested in finding TDT sample numbers for, e.g. self.ind_hold_center_states
		- state_time: array of state times taken from corresponding hdf file
		- syncHDF_file: syncHDF.mat file path, e.g. '/home/srsummerson/storage/syncHDF/Mario20161104_b1_syncHDF.mat'
	Output:
		- lfp_state_row_ind: array of tdt sample numbers that correspond the the task state events in ind_state array
	'''
	# Load syncing data
	hdf_times = dict()
	sp.io.loadmat(syncHDF_file, hdf_times)
	#print(syncHDF_file)
	hdf_rows = np.ravel(hdf_times['row_number'])
	hdf_rows = [val for val in hdf_rows]
	#print(hdf_times['tdt_dio_samplerate'])
	dio_tdt_sample = np.ravel(hdf_times['tdt_samplenumber'])
	dio_freq = np.ravel(hdf_times['tdt_dio_samplerate'])

	lfp_dio_sample_num = dio_tdt_sample  # assumes DIOx and LFPx are saved using the same sampling rate

	state_row_ind = state_time[ind_state]		# gives the hdf row number sampled at 60 Hz
	lfp_state_row_ind = np.zeros(state_row_ind.size)

	for i in range(len(state_row_ind)):
		hdf_index = np.argmin(np.abs(hdf_rows - state_row_ind[i])) #find index of row we are on
		if np.abs(hdf_rows[hdf_index] - state_row_ind[i])==0: #if time of state matches hdf row timestamp
			lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]
		elif hdf_rows[hdf_index] > state_row_ind[i]: #if times don't match up, do linear interp with the prev timestamp
			hdf_row_diff = hdf_rows[hdf_index] - hdf_rows[hdf_index -1]  # distance of the interval of the two closest hdf_row_numbers
			m = (lfp_dio_sample_num[hdf_index]-lfp_dio_sample_num[hdf_index - 1])/hdf_row_diff
			b = lfp_dio_sample_num[hdf_index-1] - m*hdf_rows[hdf_index-1]
			lfp_state_row_ind[i] = int(m*state_row_ind[i] + b)
		elif (hdf_rows[hdf_index] < state_row_ind[i])&(hdf_index + 1 < len(hdf_rows)): #if times don't match up, do linear interp with the next timestamp
			hdf_row_diff = hdf_rows[hdf_index + 1] - hdf_rows[hdf_index]
			if (hdf_row_diff > 0):
				m = (lfp_dio_sample_num[hdf_index + 1] - lfp_dio_sample_num[hdf_index])/hdf_row_diff
				b = lfp_dio_sample_num[hdf_index] - m*hdf_rows[hdf_index]
				lfp_state_row_ind[i] = int(m*state_row_ind[i] + b)
			else:
				lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]
		else:
			lfp_state_row_ind[i] = lfp_dio_sample_num[hdf_index]

	return lfp_state_row_ind, dio_freq
	
	
class ChoiceBehavior_TwoTargets_Stimulation():
	'''
	Class for behavior taken from ABA' task, where there are two targets of different probabilities of reward
	and stimulation is paired with the middle-value target during the hold-period of instructed trials during
	blocks B and A'. Can pass in a list of hdf files when initially instantiated in the case that behavioral data
	is split across multiple hdf files. In this case, the files should be listed in the order in which they were saved.
	'''

	def __init__(self, hdf_files, num_trials_A, num_trials_B):
		for i, hdf_file in enumerate(hdf_files): 
			filename =  hdf_file
			table = tables.open_file(filename)
			if i == 0:
				self.state = table.root.task_msgs[:]['msg']
				self.state_time = table.root.task_msgs[:]['time']
				self.trial_type = table.root.task[:]['target_index']
				self.targetL = table.root.task[:]['targetL']
				self.targetH = table.root.task[:]['targetH']
# 				self.hdfs = [table]

			else:
				self.state = np.append(self.state, table.root.task_msgs[:]['msg'])
				self.state_time = np.append(self.state_time, self.state_time[-1] + table.root.task_msgs[:]['time'])
				self.trial_type = np.append(self.trial_type, table.root.task[:]['target_index'])
				self.targetL = np.vstack([self.targetL, table.root.task[:]['targetL']])
				self.targetH = np.vstack([self.targetH, table.root.task[:]['targetH']])
# 				self.hdfs.append(table)
				
		self.ind_wait_states = np.ravel(np.nonzero(self.state == b'wait'))   # total number of unique trials
		self.ind_center_states = np.ravel(np.nonzero(self.state == b'center'))   # total number of totals (includes repeats if trial was incomplete)
		self.ind_hold_center_states = np.ravel(np.nonzero(self.state == b'hold_center'))
		self.ind_hold_center_stimulate_states = np.ravel(np.nonzero(self.state == b'hold_center_and_stimulate'))
		self.ind_target_states = np.ravel(np.nonzero(self.state == b'target'))
		self.ind_hold_targetL_states = np.ravel(np.nonzero(self.state == b'hold_targetL'))
		self.ind_hold_targetH_states = np.ravel(np.nonzero(self.state == b'hold_targetH'))
		self.ind_check_reward_states = np.ravel(np.nonzero(self.state == b'check_reward'))
		
		self.fs_hdf = 60 #Hz
		
		self.num_trials = self.ind_center_states.size
		self.num_successful_trials = self.ind_check_reward_states.size
		self.num_trials_A = num_trials_A
		self.num_trials_B = num_trials_B
		#self.table=table


	def GetChoicesAndRewards(self):
		'''
		This method extracts which target was chosen for each trial, whether or not a reward was given, and if
		the trial was instructed or free-choice.

		Output:
		- chosen_target: array of length N, where N is the number of trials (instructed + freechoice), which contains 
						contains values indicating a low-value target choice was made (=1) or if a high-value target
						choice was made (=2)
		- rewards: array of length N, which contains 0s or 1s to indicate whether a reward was received at the end
							of the ith trial. 
		- instructed_or_freechoice: array of length N, which indicates whether a trial was instructed (=1) or free-choice (=2)

		'''

		ind_holds = self.ind_check_reward_states - 2
		ind_rewards = self.ind_check_reward_states + 1
		rewards = np.array([float(st==b'reward') for st in self.state[ind_rewards]])
		instructed_or_freechoice = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]])  # = 1: instructed, =2: free-choice
		chosen_target = np.array([(int(self.state[ind]==b'hold_targetH') + 1) for ind in ind_holds])

		
		return chosen_target, rewards, instructed_or_freechoice


	def ChoicesAfterStimulation(self):
		'''
		Method to extract information about the stimulation trials and the trial immediately following
		stimulation. 

		Output: 
		- choice: length n array with values indicating which target was selecting in the trial following
			stimulation (= 1 if LV, = 2 if HV)
		- stim_reward: length n array with Boolean values indicating whether reward was given (True) during
			the stimulation trial or not (False)
		- target_reward: length n array with Boolean values indicating whether reward was given (True) in
			the trial following the stimulation trial or not (False)
		- stim side: length n array with values indicating what side the MV target was on during the 
			stimulation trial (= 1 for left, -1 for right)
		- stim_trial_ind: length n array containing trial numbers during which stimulation was performed. 
			Since stimulation is during Block A' only, the minimum value should be num_trials_A + num_trials_B
			and the maximum value should be the total number of trials.
		'''

		# Get target selection information
		chosen_target, rewards, instructed_or_freechoice, _ = self.GetChoicesAndRewards()
		ind_targets = self.ind_check_reward_states - 3
		targetH_side = self.targetH[self.state_time[ind_targets]][:,2]
		targetL_side = self.targetL[self.state_time[ind_targets]][:,2]
		instructed_trials_inds = np.ravel(np.nonzero(2 - instructed_or_freechoice))
		stim_trial_inds = instructed_trials_inds[np.ravel(np.nonzero(np.greater(instructed_trials_inds, self.num_trials_A + self.num_trials_B)))]
		stim_trial_inds = np.array([ind for ind in stim_trial_inds if ((ind+1) not in stim_trial_inds)&(ind < (self.num_successful_trials-1))])
		fc_trial_inds = stim_trial_inds + 1

		choice = chosen_target[fc_trial_inds]		# what target was selected in trial following stim trial
		stim_reward = rewards[stim_trial_inds]		# was the stim trial rewarded
		target_reward = rewards[fc_trial_inds]		# was the target selected in trial after stimulation rewarded
		stim_side = targetL_side[stim_trial_inds] 		# side the MV target was on during stimulation
		choice_side = np.array([(targetH_side[ind]*(choice[i]==2) + targetL_side[ind]*(choice[i]==1)) for i,ind in enumerate(fc_trial_inds)])		# what side the selected target was on following the stim trial
		
		return choice, choice_side, stim_reward, target_reward, stim_side, stim_trial_inds

	def GetTargetSideSelection(self):
		'''
		Method to determine which side (left or right) the chosen target was on 
		for all successful reach trials.

		Output:
			- choice_side: 1 x num_trials vector
			Vector containing the side (left or right) chosen for each 
			successful reach trial. 0 or 1.

		'''
		choices, _, _ = self.GetChoicesAndRewards()

		# Get target side information
		ind_targets = self.ind_check_reward_states - 3
		targetL_side = self.targetL[self.state_time[ind_targets]][:,2]
		targetH_side = self.targetH[self.state_time[ind_targets]][:,2]
		
		# Get the side information for the chosen target
		choice_side = np.array([(targetH_side[i]*(choice==2) + targetL_side[i]*(choice==1)) for i,choice in enumerate(choices)])

		return choice_side
	
	
	
	def CalcValue_2Targs(self,choices,rewards,win_sz,smooth):
		'''
		Calculates values of the high value and low value target for each trial throughout a session by using a sliding window.
		This defines value as a number of rewards dispensed in past n choices of a particular target divided by past n choices to that target.
		Essentially empirical probability of reward as calculated by a sliding window of size n.
		
		Parameters
		----------
		choices : 1 x num_trials vector
			Vector containing the NHP's choice for each trial which had a successful reach. 
			1 = low value target, 2 = high value target.
		rewards : 1 x num_trials vector
			Vector containing if a reward was dispensed for each trial which had a successful reach. 
			0 = no reward, 1 = reward dispensed.
		win_sz : int
			Defines window size for sliding average
		smooth : bool
			Whether to perform a sliding average over the final result
			
		Returns
		-------
		values : num_targets x num_trials
			Matrix containing the value of each target during each trial which had a successful reach.
			Row 0  = low value target
			Row 1 = high value target
		win_sz : int
			Defines window size for sliding average.
		'''
	
		val_lv = np.full([len(choices)],0.5) #0.5 is default value
		val_hv = np.full([len(choices)],0.5)
		
		# Calculate empirical probability of reward using sliding window
		for i in range(len(choices)):
			
			if i !=0 : #skip first trial
				
				#LV
				counter=0 #counts number of trials to target
				j=0 #variable to increment to go backwards in time
				lv_rewards=[] #list to accumulate rewards of lv choice trials
				
				while counter < win_sz: #go backwards in time until you get win_sz number of trials
					
					if i-j == 0: #if we are at beginning, then exit while loop
						break
					j+=1 
					
					if choices[i-j] == 1: #lv choices
						lv_rewards.append(rewards[i-j]) #append rewards of lv choices
						counter+=1 #update counter of how many trials we have
						
				# calculate value
				if counter>0:
					val_lv[i] = np.sum(lv_rewards) / counter #sum rewards and divide by number of trials
					
				
				#HV
				counter=0 #counts number of trials to target
				j=0 #variable to increment to go backwards in time
				hv_rewards=[] #list to accumulate rewards of hv choice trials
				
				while counter < win_sz: #go backwards in time until you get win_sz number of trials
					
					if i-j == 0: #if we are at beginning, then exit while loop
						break
					j+=1 
					
					if choices[i-j] == 2: #hv choices
						hv_rewards.append(rewards[i-j]) #append rewards of hv choices
						counter+=1 #update counter of how many trials we have
						
				# calculate value
				if counter>0:
					val_hv[i] = np.sum(hv_rewards) / counter #sum rewards and divide by number of trials
			
# 		print(lv_rewards)

		if smooth:
			#val_lv = trial_sliding_avg(val_lv,num_trials_slide=5)
			#val_hv = trial_sliding_avg(val_hv,num_trials_slide=5)
			val_hv = sp.ndimage.gaussian_filter(val_hv,sigma=1)
			val_lv = sp.ndimage.gaussian_filter(val_lv,sigma=1)

		# Combine high and low values into a single matrix
		values = [np.transpose(val_lv),np.transpose(val_hv)]
		return values, win_sz

	
	def GetFreeChoicesRewardsAndValues(choices,rewards,values,instructed_or_freechoice):
		'''
		Takes the choices, rewards, and values and filters them by instructed_or_freechoice
		so that only the free choices remain.
	
		Parameters
		----------
		choices : 1 x num_trials vector
			Vector containing the NHP's choice for each trial which had a successful reach. 
			0 = low value target, 2 = high value target.
		rewards : 1 x num_trials vector
			Vector containing if a reward was dispensed for each trial which had a successful reach. 
			0 = no reward, 1 = reward dispensed.
		instructed_or_freechoice: array of length num_trials
			Indicates whether a trial was instructed (=1) or free-choice (=2)
		values : num_targets x num_trials
			Matrix containing the value of each target during each trial which had a successful reach.
			Row 0  = low value target
			Row 1 = high value target
	
		Returns
		-------
		free_choices : 1 x num_free_choices vector
			Vector containing the NHP's choice for free choice trials which had a successful reach. 
			1 = low value target, 2 = high value target.
		free_choice_rewards : 1 x num_free_choices vector
			Vector containing if a reward was dispensed for free choice trials which had a successful reach.
		free_choice_values: num_targets x num_free_choices
			Matrix containing the value of each target during free choice trials which had a successful reach.
			Row 0  = low value target
			Row 1 = high value target
	
		'''
		
		#Filter original vectors by instructed_or_freechoice to get only free choice trials
		free_choices = choices[instructed_or_freechoice==2]
		free_choice_rewards = rewards[instructed_or_freechoice==2]
		free_choice_low_values = values[0][instructed_or_freechoice==2]
		free_choice_high_values = values[1][instructed_or_freechoice==2]
		free_choice_values = [free_choice_low_values,free_choice_high_values]
		
		return free_choices, free_choice_rewards, free_choice_values
	
	
	
	def CalcMovtTime(self):
		
		#combine hold_targetL and hold_targetH
		ind_hold_target_states = np.sort(np.concatenate((self.ind_hold_targetH_states,self.ind_hold_targetL_states)))
				
		mts = []
		#we only want the successful reaches (where the hold_center precedes a check_reward) 
		for ind in ind_hold_target_states:
			if self.state[ind+2] == b'check_reward': #if the reach was successful
				mt = (self.state_time[ind] - self.state_time[ind-1]) / self.fs_hdf
 				#print(mt)
				mts.append(mt)
			
		return np.array(mts)


class ChoiceBehavior_ThreeTargets_Stimulation():
	'''
	Class for behavior taken from ABA' task, where there are three targets of different probabilities of reward
	and stimulation is paired with the middle-value target during the hold-period of instructed trials during
	blocks B and A'. Can pass in a list of hdf files when initially instantiated in the case that behavioral data
	is split across multiple hdf files. In this case, the files should be listed in the order in which they were saved.
	'''

	def __init__(self, hdf_files, num_trials_A, num_trials_B):
		for i, hdf_file in enumerate(hdf_files): 
			filename =  hdf_file
			table = tables.open_file(filename)
			if i == 0:
				self.state = table.root.task_msgs[:]['msg']
				self.state_time = table.root.task_msgs[:]['time']
				self.trial_type = table.root.task[:]['target_index']
				self.targets_on = table.root.task[:]['LHM_target_on']
				self.targetL = table.root.task[:]['targetL']
				self.targetH = table.root.task[:]['targetH']
				self.targetM = table.root.task[:]['targetM']
			else:
				self.state = np.append(self.state, table.root.task_msgs[:]['msg'])
				self.state_time = np.append(self.state_time, self.state_time[-1] + table.root.task_msgs[:]['time'])
				self.trial_type = np.append(self.trial_type, table.root.task[:]['target_index'])
				self.targets_on = np.append(self.targets_on, table.root.task[:]['LHM_target_on'])
				self.targetL = np.vstack([self.targetL, table.root.task[:]['targetL']])
				self.targetH = np.vstack([self.targetH, table.root.task[:]['targetH']])
				self.targetM = np.vstack([self.targetM, table.root.task[:]['targetM']])
		
		if len(hdf_files) > 1:
			self.targets_on = np.reshape(self.targets_on, (len(self.targets_on)//3,3))  				# this should contain triples indicating targets
		self.ind_wait_states = np.ravel(np.nonzero(self.state == b'wait'))   # total number of unique trials
		self.ind_center_states = np.ravel(np.nonzero(self.state == b'center'))   # total number of totals (includes repeats if trial was incomplete)
		self.ind_hold_center_states = np.ravel(np.nonzero(self.state == b'hold_center'))
		self.ind_hold_center_stimulate_states = np.ravel(np.nonzero(self.state == b'hold_center_and_stimulate'))
		self.ind_target_states = np.ravel(np.nonzero(self.state == b'target'))
		self.ind_hold_targetL_states = np.ravel(np.nonzero(self.state == b'hold_targetL'))
		self.ind_hold_targetM_states = np.ravel(np.nonzero(self.state == b'hold_targetM'))
		self.ind_hold_targetH_states = np.ravel(np.nonzero(self.state == b'hold_targetH'))
		self.ind_check_reward_states = np.ravel(np.nonzero(self.state == b'check_reward'))
		
		self.fs_hdf = 60 #Hz
		
		self.num_trials = self.ind_center_states.size
		self.num_successful_trials = self.ind_check_reward_states.size
		self.num_trials_A = num_trials_A
		self.num_trials_B = num_trials_B
		#self.table=table
		

	def TrialChoices(self, num_trials_slide, plot_results = False):
		'''
		This method computes the sliding average over num_trials_slide trials of the number of choices for the 
		optimal target choice. It looks at overall the liklihood of selecting the better choice, as well as the 
		choice behavior for the three different scenarios: L-H targets shown, L-M targets shown, and M-H targets
		shown. Choice behavior is split across the three blocks.
		'''

		# Get indices of free-choice trials for Blocks A and A', as well as the corresponding target selections.
		freechoice_trial = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]]) - 1
		freechoice_trial_ind_A = np.ravel(np.nonzero(freechoice_trial[:self.num_trials_A]))
		freechoice_trial_ind_Aprime = np.ravel(np.nonzero(freechoice_trial[self.num_trials_A+self.num_trials_B:])) + self.num_trials_A+self.num_trials_B
		
		target_choices_A = self.state[self.ind_check_reward_states - 2][freechoice_trial_ind_A]
		target_choices_Aprime = self.state[self.ind_check_reward_states - 2][freechoice_trial_ind_Aprime]
		targets_on = self.targets_on[self.state_time[self.ind_check_reward_states]]  # array of three boolean values: LHM

		# Initialize variables
		num_FC_trials_A = len(freechoice_trial_ind_A)
		num_FC_trials_Aprime = len(freechoice_trial_ind_Aprime)
		all_choices_A = np.zeros(num_FC_trials_A)
		all_choices_Aprime = np.zeros(num_FC_trials_Aprime)
		LM_choices_A = []
		LH_choices_A = []
		MH_choices_A = []
		LM_choices_Aprime = []
		LH_choices_Aprime = []
		MH_choices_Aprime = []
		

		for i, choice in enumerate(target_choices_A):
			# only look at freechoice trials
			targ_presented = targets_on[freechoice_trial_ind_A[i]]
			# L-M targets presented
			if (targ_presented[0]==1)&(targ_presented[2]==1):
				if choice=='hold_targetM':
					all_choices_A[i] = 1		# optimal choice was made
					LM_choices_A = np.append(LM_choices_A, 1)
				else:
					LM_choices_A = np.append(LM_choices_A, 0)

			# L-H targets presented
			if (targ_presented[0]==1)&(targ_presented[1]==1):
				if choice=='hold_targetH':
					all_choices_A[i] = 1
					LH_choices_A = np.append(LH_choices_A, 1)
				else:
					LH_choices_A = np.append(LH_choices_A, 0)

			# M-H targets presented
			if (targ_presented[1]==1)&(targ_presented[2]==1):
				if choice=='hold_targetH':
					all_choices_A[i] = 1
					MH_choices_A = np.append(MH_choices_A, 1)
				else:
					MH_choices_A = np.append(MH_choices_A, 0)

		for i, choice in enumerate(target_choices_Aprime):
			# only look at freechoice trials
			targ_presented = targets_on[freechoice_trial_ind_Aprime[i]]
			# L-M targets presented
			if (targ_presented[0]==1)&(targ_presented[2]==1):
				if choice=='hold_targetM':
					all_choices_Aprime[i] = 1		# optimal choice was made
					LM_choices_Aprime = np.append(LM_choices_Aprime, 1)
				else:
					LM_choices_Aprime = np.append(LM_choices_Aprime, 0)

			# L-H targets presented
			if (targ_presented[0]==1)&(targ_presented[1]==1):
				if choice=='hold_targetH':
					all_choices_Aprime[i] = 1
					LH_choices_Aprime = np.append(LH_choices_Aprime, 1)
				else:
					LH_choices_Aprime = np.append(LH_choices_Aprime, 0)

			# M-H targets presented
			if (targ_presented[1]==1)&(targ_presented[2]==1):
				if choice=='hold_targetH':
					all_choices_Aprime[i] = 1
					MH_choices_Aprime = np.append(MH_choices_Aprime, 1)
				else:
					MH_choices_Aprime = np.append(MH_choices_Aprime, 0)

		sliding_avg_all_choices_A = trial_sliding_avg(all_choices_A, num_trials_slide)
		sliding_avg_LM_choices_A = trial_sliding_avg(LM_choices_A, num_trials_slide)
		sliding_avg_LH_choices_A = trial_sliding_avg(LH_choices_A, num_trials_slide)
		sliding_avg_MH_choices_A = trial_sliding_avg(MH_choices_A, num_trials_slide)

		sliding_avg_all_choices_Aprime = trial_sliding_avg(all_choices_Aprime, num_trials_slide)
		sliding_avg_LM_choices_Aprime = trial_sliding_avg(LM_choices_Aprime, num_trials_slide)
		sliding_avg_LH_choices_Aprime = trial_sliding_avg(LH_choices_Aprime, num_trials_slide)
		sliding_avg_MH_choices_Aprime = trial_sliding_avg(MH_choices_Aprime, num_trials_slide)

		if plot_results:
			fig = plt.figure()
			ax11 = plt.subplot(221)
			plt.plot(sliding_avg_LM_choices_A, c = 'b', label = 'Block A')
			plt.plot(sliding_avg_LM_choices_Aprime, c = 'r', label = "Block A'")
			plt.xlabel('Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('Low vs. Mid')
			ax11.get_yaxis().set_tick_params(direction='out')
			ax11.get_xaxis().set_tick_params(direction='out')
			ax11.get_xaxis().tick_bottom()
			ax11.get_yaxis().tick_left()
			plt.legend()

			ax12 = plt.subplot(222)
			plt.plot(sliding_avg_LH_choices_A, c = 'b', label = 'Block A')
			plt.plot(sliding_avg_LH_choices_Aprime, c = 'r', label = "Block A'")
			plt.xlabel('Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('Low vs. High')
			ax12.get_yaxis().set_tick_params(direction='out')
			ax12.get_xaxis().set_tick_params(direction='out')
			ax12.get_xaxis().tick_bottom()
			ax12.get_yaxis().tick_left()
			plt.legend()

			ax21 = plt.subplot(223)
			plt.plot(sliding_avg_MH_choices_A, c = 'b', label = 'Block A')
			plt.plot(sliding_avg_MH_choices_Aprime, c = 'r', label = "Block A'")
			plt.xlabel('Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('Mid vs. High')
			ax21.get_yaxis().set_tick_params(direction='out')
			ax21.get_xaxis().set_tick_params(direction='out')
			ax21.get_xaxis().tick_bottom()
			ax21.get_yaxis().tick_left()
			plt.legend()

			ax22 = plt.subplot(224)
			plt.plot(sliding_avg_all_choices_A, c = 'b', label = 'Block A')
			plt.plot(sliding_avg_all_choices_Aprime, c = 'r', label = "Block A'")
			plt.xlabel('Trials')
			plt.ylabel('Probability Best Choice')
			plt.title('All Choices')
			ax22.get_yaxis().set_tick_params(direction='out')
			ax22.get_xaxis().set_tick_params(direction='out')
			ax22.get_xaxis().tick_bottom()
			ax22.get_yaxis().tick_left()
			plt.legend()

		return all_choices_A, LM_choices_A, LH_choices_A, MH_choices_A, all_choices_Aprime, LM_choices_Aprime, LH_choices_Aprime, MH_choices_Aprime

	def GetChoicesAndRewards(self):
		'''
		This method extracts which target was chosen for each trial, whether or not a reward was given, and if
		the trial was instructed or free-choice.

		Output:
		- chosen_target: array of length N, where N is the number of trials (instructed + freechoice), which contains 
						contains values indicating if choice was to low-value target (=0), to medium-value target (=1), 
						or to high-value target (=2)
		- rewards: array of length N, which contains 0s or 1s to indicate whether a reward was received at the end
							of the ith trial. 
		- instructed_or_freechoice: array of length N, which indicates whether a trial was instructed (=1) or free-choice (=2)

		'''
		
		ind_holds = self.ind_check_reward_states - 2
		ind_rewards = self.ind_check_reward_states + 1
		rewards = np.array([float(st==b'reward') for st in self.state[ind_rewards]])
		targets_on = self.targets_on[self.state_time[self.ind_check_reward_states]]  # array of three boolean values: LHM
		instructed_or_freechoice = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]])  # = 1: instructed, =2: free-choice
		chosen_target = np.zeros(len(ind_holds))

		for i, ind in enumerate(ind_holds):
			if self.state[ind] == b'hold_targetM':
				chosen_target[i] = 1
			elif self.state[ind] == b'hold_targetH':
				chosen_target[i] = 2

		return targets_on, chosen_target, rewards, instructed_or_freechoice


	def GetTargetSideSelection(self):
		'''
		Method to determine which side (left or right) the chosen target was on 
		for all successful reach trials.

		Output:
			- choice_side: 1 x num_trials vector
			Vector containing the side (left or right) chosen for each 
			successful reach trial. 0 or 1.

		'''
		_,choices, _, _ = self.GetChoicesAndRewards()

		# Get target side information
		ind_targets = self.ind_check_reward_states - 3
		targetL_side = self.targetL[self.state_time[ind_targets]][:,2]
		targetM_side = self.targetM[self.state_time[ind_targets]][:,2]
		targetH_side = self.targetH[self.state_time[ind_targets]][:,2]
		
		# Get the side information for the chosen target
		choice_side = np.array([(targetH_side[i]*(choice==2) + targetM_side[i]*(choice==1) + targetL_side[i]*(choice==0)) for i,choice in enumerate(choices)])

		return choice_side
	

	def ChoicesAfterStimulation(self):
		'''
		Method to extract information about the stimulation trials and the trial immediately following
		stimulation. 

		Output: 
		- targets_on_after: n x 3 array, where n is the number of stimulation trials, containing indicators
			of whether the LV, HV, or MV targets are shown, respectively. E.g. if after the first stimulation
			trial the LV and MV targets are shown, then targets_on_after[0] = [1,0,1].
		- choice: length n array with values indicating which target was selecting in the trial following
			stimulation (=0 if LV, = 1 if MV, = 2 if HV)
		- stim_reward: length n array with Boolean values indicating whether reward was given (True) during
			the stimulation trial or not (False)
		- target_reward: length n array with Boolean values indicating whether reward was given (True) in
			the trial following the stimulation trial or not (False)
		- stim side: length n array with values indicating what side the MV target was on during the 
			stimulation trial (= 1 for left, -1 for right)
		- stim_trial_ind: length n array containing trial numbers during which stimulation was performed. 
			Since stimulation is during Block A' only, the minimum value should be num_trials_A + num_trials_B
			and the maximum value should be the total number of trials.
		'''

		# Get target selection information
		targets_on, chosen_target, rewards, instructed_or_freechoice = self.GetChoicesAndRewards()
		ind_targets = self.ind_check_reward_states - 3
		targetM_side = self.targetM[self.state_time[ind_targets]][:,2]
		targetH_side = self.targetH[self.state_time[ind_targets]][:,2]
		targetL_side = self.targetL[self.state_time[ind_targets]][:,2]
		instructed_or_freechoice = np.ravel(self.trial_type[self.state_time[self.ind_check_reward_states]])  # = 1: instructed, =2: free-choice

		targets_on_after = []	# what targets are presented after stim trial
		choice = []				# what target was selected in trial following stim trial
		choice_side = []		# what side the selected target was on following the stim trial
		stim_reward = []		# was the stim trial rewarded
		target_reward = []		# was the target selected in trial after stimulation rewarded
		stim_side = [] 			# side the MV target was on during stimulation
		stim_trial_ind = []		# index of trial with stimulation

		counter = 0

		# Find trials only following a trial with stimulation
		for i in range(self.num_trials_A + self.num_trials_B,len(chosen_target)-1): # only consider trials in Block A'
			# only consider M is shown (stim trial) and next trial is not also a stim trial
			if np.array_equal(targets_on[i],[0,0,1])&(~np.array_equal(targets_on[i+1],[0,0,1])):  		
				if counter==0:
					targets_on_after = targets_on[i+1]
				else:
					targets_on_after = np.vstack([targets_on_after,targets_on[i+1]])
				
				choice = np.append(choice,chosen_target[i+1])					# choice = 0 if LV, choice = 1 if MV, choice = 2 if HV
				if chosen_target[i+1]==0:
					side = targetL_side[i+1]*(targetL_side[i+1]!=0) + (2*np.random.randint(2) - 1)*(targetL_side[i+1]==0)
					choice_side = np.append(choice_side, side)
				elif chosen_target[i+1]==1:
					side = targetM_side[i+1]*(targetM_side[i+1]!=0) + (2*np.random.randint(2) - 1)*(targetM_side[i+1]==0)
					choice_side = np.append(choice_side, side)
				else:
					side = targetH_side[i+1]*(targetH_side[i+1]!=0) + (2*np.random.randint(2) - 1)*(targetH_side[i+1]==0)
					choice_side = np.append(choice_side, side)

				stim_side_find = targetM_side[i]*(targetM_side[i]!=0) + (2*np.random.randint(2) - 1)*(targetM_side[i]==0)
				#stim_side_find = targetM_side[i]
				stim_reward = np.append(stim_reward, rewards[i])
				target_reward = np.append(target_reward, rewards[i+1])
				stim_side = np.append(stim_side, stim_side_find)
				stim_trial_ind = np.append(stim_trial_ind, i)
				counter += 1

		return targets_on_after, choice, choice_side, stim_reward, target_reward, stim_side, stim_trial_ind
	
	
	def CalcValue_3Targs(self,choices,rewards,win_sz,smooth):
		
		'''
		Calculates values of the high value and low value target for each trial throughout a session by using a sliding window.
		This defines value as a number of rewards dispensed in past n choices of a particular target divided by past n choices to that target.
		Essentially empirical probability of reward as calculated by a sliding window of size n.
		
		Parameters
		----------
		choices : 1 x num_trials vector
			Vector containing the NHP's choice for each trial which had a successful reach. 
			0 = low value target, 1 = med value target, 2 = high value target.
		rewards : 1 x num_trials vector
			Vector containing if a reward was dispensed for each trial which had a successful reach. 
			0 = no reward, 1 = reward dispensed.
		win_sz : int
			Defines window size for sliding average
		smooth : bool
			Whether to perform a sliding average over the final result
			
		Returns
		-------
		values : num_targets x num_trials
			Matrix containing the value of each target during each trial which had a successful reach.
			Row 0  = low value target
			Row 1 = med value target
			Row 2 = high value target
		win_sz : scalar
			Defines window size for sliding average.
		'''
		
		val_lv = np.full([len(choices)],0.5) #0.5 is default value
		val_mv = np.full([len(choices)],0.5)
		val_hv = np.full([len(choices)],0.5)
		
		# Calculate empirical probability of reward using sliding window
		for i in range(len(choices)):
			
			if i !=0 : #skip first trial
				
				#LV
				counter=0 #counts number of trials to target
				j=0 #variable to increment to go backwards in time
				lv_rewards=[] #list to accumulate rewards of lv choice trials
				
				while counter < win_sz: #go backwards in time until you get win_sz number of trials
					
					if i-j == 0: #if we are at beginning, then exit while loop
						break
					j+=1 
					
					if choices[i-j] == 0: #lv choices
						lv_rewards.append(rewards[i-j]) #append rewards of lv choices
						counter+=1 #update counter of how many trials we have
						
				# calculate value
				if counter>0:
					val_lv[i] = np.sum(lv_rewards) / counter #sum rewards and divide by number of trials
					
					
				#MV
				counter=0 #counts number of trials to target
				j=0 #variable to increment to go backwards in time
				mv_rewards=[] #list to accumulate rewards of mv choice trials
				
				while counter < win_sz: #go backwards in time until you get win_sz number of trials
					
					if i-j == 0: #if we are at beginning, then exit while loop
						break
					j+=1 
					
					if choices[i-j] == 1: #mv choices
						mv_rewards.append(rewards[i-j]) #append rewards of mv choices
						counter+=1 #update counter of how many trials we have
						
				# calculate value
				if counter>0:
					val_mv[i] = np.sum(mv_rewards) / counter #sum rewards and divide by number of trials
						
				
				#HV
				counter=0 #counts number of trials to target
				j=0 #variable to increment to go backwards in time
				hv_rewards=[] #list to accumulate rewards of hv choice trials
				
				while counter < win_sz: #go backwards in time until you get win_sz number of trials
					
					if i-j == 0: #if we are at beginning, then exit while loop
						break
					j+=1 
					
					if choices[i-j] == 2: #hv choices
						hv_rewards.append(rewards[i-j]) #append rewards of hv choices
						counter+=1 #update counter of how many trials we have
						
				# calculate value
				if counter>0:
					val_hv[i] = np.sum(hv_rewards) / counter #sum rewards and divide by number of trials
			
# 		print(lv_rewards)

		if smooth:
			#val_lv = trial_sliding_avg(val_lv,num_trials_slide=5)
			#val_mv = trial_sliding_avg(val_mv,num_trials_slide=5)
			#val_hv = trial_sliding_avg(val_hv,num_trials_slide=5)
			val_lv = sp.ndimage.gaussian_filter(val_lv,sigma=1)
			val_mv = sp.ndimage.gaussian_filter(val_mv,sigma=1)
			val_hv = sp.ndimage.gaussian_filter(val_hv,sigma=1)

		# Combine high and low values into a single matrix
		values = [np.transpose(val_lv),np.transpose(val_mv),np.transpose(val_hv)]
		return values, win_sz
	
	
	def GetFreeChoicesRewardsAndValues(choices,rewards,values,instructed_or_freechoice):
		'''
		Takes the choices, rewards, and values and filters them by instructed_or_freechoice
		so that only the free choices remain.
	
		Parameters
		----------
		choices : 1 x num_trials vector
			Vector containing the NHP's choice for each trial which had a successful reach. 
			0 = low value target, 1 = med value target, 2 = high value target.
		rewards : 1 x num_trials vector
			Vector containing if a reward was dispensed for each trial which had a successful reach. 
			0 = no reward, 1 = reward dispensed.
		instructed_or_freechoice: array of length num_trials
			Indicates whether a trial was instructed (=1) or free-choice (=2)
		values : num_targets x num_trials
			Matrix containing the value of each target during each trial which had a successful reach.
			Row 0  = low value target
			Row 1 = med value target
			Row 2 = high value target
	
		Returns
		-------
		free_choices : 1 x num_free_choices vector
			Vector containing the NHP's choice for free choice trials which had a successful reach. 
			0 = low value target, 1 = med value target, 2 = high value target.
		free_choice_rewards : 1 x num_free_choices vector
			Vector containing if a reward was dispensed for free choice trials which had a successful reach.
		free_choice_values: num_targets x num_free_choices
			Matrix containing the value of each target during free choice trials which had a successful reach.
			Row 0  = low value target
			Row 1 = med value target
			Row 2 = high value target
	
		'''
		
		#Filter original vectors by instructed_or_freechoice to get only free choice trials
		free_choices = choices[instructed_or_freechoice==2]
		free_choice_rewards = rewards[instructed_or_freechoice==2]
		free_choice_low_values = values[0][instructed_or_freechoice==2]
		free_choice_med_values = values[1][instructed_or_freechoice==2]
		free_choice_high_values = values[2][instructed_or_freechoice==2]
		free_choice_values = [free_choice_low_values,free_choice_med_values,free_choice_high_values]
		
		return free_choices, free_choice_rewards, free_choice_values
	
	
	def CalcMovtTime(self):
		
		#combine hold_targetL and hold_targetH
		ind_hold_target_states = np.sort(np.concatenate((self.ind_hold_targetH_states,self.ind_hold_targetM_states,self.ind_hold_targetL_states)))
				
		mts = []
		#we only want the successful reaches (where the hold_center precedes a check_reward) 
		for ind in ind_hold_target_states:
			if self.state[ind+2] == b'check_reward': #if the reach was successful
				mt = (self.state_time[ind] - self.state_time[ind-1]) / self.fs_hdf
 				#print(mt)
				mts.append(mt)
			
		return np.array(mts)
	
	

class CalcRxnTime:
	
	def __init__(self,hdf_file):
		self.hdf_file = hdf_file

	def compute_rt_per_trial_FreeChoiceTask(self):
		# Load HDF file
		hdf = tables.open_file(self.hdf_file)
		
		
		# Extract go_cue_indices in units of hdf file row number
		go_cue_ix = np.array([hdf.root.task_msgs[j - 3]['time']
							  for j, i in enumerate(hdf.root.task_msgs) if i['msg'] == b'check_reward'])
		
		# Calculate filtered velocity and 'velocity mag. in target direction'
		filt_vel, total_vel, vel_bins, skipped_indices = self.get_cursor_velocity(hdf,go_cue_ix, 0., 2., use_filt_vel=False)
		
		# Calculate 'RT' from vel_in_targ_direction: use with get_cusor_velocity_in_targ_dir
		#kin_feat = get_kin_sig_shenoy_method(vel_in_targ_dir.T, vel_bins, perc=.2, start_tm = .1)
		#kin_feat = get_rt(total_vel.T, vel_bins, vel_thres = 0.1)
		kin_feat = self.get_rt_change_deriv(
			total_vel.T, vel_bins, d_vel_thres=0.3, fs=60)
		

		#Plot first 5 trials in a row
# 		for n in range(0,5):
# 			plt.plot(total_vel[:, n], '.-')
# 			plt.plot(kin_feat[n, :][0], total_vel[int(kin_feat[n,:][0]), n], '.', markersize=10)
# 			plt.show()
# 			time.sleep(1.)

		hdf.close()
		
		return kin_feat[:, 1], total_vel
	
	
	def get_cursor_velocity(self, hdf, go_cue_ix, before_cue_time, after_cue_time, fs=60., use_filt_vel=True):
		'''
		hdf file -- task file generated from bmi3d
		go_cue_ix -- list of go cue indices (units of hdf file row numbers)
		before_cue_time -- time before go cue to inclue in trial (units of sec)
		after_cue_time -- time after go cue to include in trial (units of sec)
	
		returns a time x (x,y) x trials filtered velocity array
		'''

		ix = np.arange(-1 * before_cue_time * fs, after_cue_time * fs).astype(int)
		skipped_indices = np.array([])
	
		# Get trial trajectory:
		cursor = []
		for k, g in enumerate(go_cue_ix):
			try:
				# Get cursor
				cursor.append(hdf.root.task[ix + g]['cursor'][:, [0, 2]])
	
			except:
				print('skipping index: ', g,
					  ' -- too close to beginning or end of file')
				skipped_indices = np.append(skipped_indices, k)
	
		cursor = np.dstack((cursor))	# time x (x,y) x trial
	
		dt = 1. / fs
		vel = np.diff(cursor, axis=0) / dt
	
		# Filter velocity:
		if use_filt_vel:
			filt_vel = signal.savgol_filter(vel, 9, 5, axis=0)
		else:
			filt_vel = vel
		total_vel = np.zeros((int(filt_vel.shape[0]), int(filt_vel.shape[2])))
		for n in range(int(filt_vel.shape[2])):
			total_vel[:, n] = np.sqrt(filt_vel[:, 0, n]**2 + filt_vel[:, 1, n]**2)
	
		vel_bins = np.linspace(-1 * before_cue_time, after_cue_time, vel.shape[0])
	
		return filt_vel, total_vel, vel_bins, skipped_indices
	
	
	def get_cursor_velocity_in_targ_dir(self, hdf, go_cue_ix, before_cue_time, after_cue_time, fs=60., use_filt_vel=True):
		'''
		hdf file -- task file generated from bmi3d
		go_cue_ix -- list of go cue indices (units of hdf file row numbers)
		before_cue_time -- time before go cue to inclue in trial (units of sec)
		after_cue_time -- time after go cue to include in trial (units of sec)
	
		returns a time x (x,y) x trials filtered velocity array
		'''
	
		ix = np.arange(-1 * before_cue_time * fs, after_cue_time * fs).astype(int)
	
		# Get trial trajectory:
		cursor = []
		target = []
		for g in go_cue_ix:
			try:
				# Get cursor
				cursor.append(hdf.root.task[ix + g]['cursor'][:, [0, 2]])
				# plt.plot(hdf.root.task[ix+g]['cursor'][:, 0]), hdf.root.task[ix+g]['cursor'][:, 2])
				# plt.plot(hdf.root.task[ix+g]['cursor'][0, 0]), hdf.root.task[ix+g]['cursor'][0, 2], '.', markersize=20)
	
				# Get target:
				target.append(hdf.root.task[g + 4]['target'][[0, 2]])
	
			except:
				print('skipping index: ', g,
					  ' -- too close to beginning or end of file')
		cursor = np.dstack((cursor))	# time x (x,y) x trial
		target = np.vstack((target)).T  # (x,y) x trial
	
		dt = 1. / fs
		vel = np.diff(cursor, axis=0) / dt
	
		# Filter velocity:
		filt_vel = signal.savgol_filter(vel, 9, 5, axis=0)
		vel_bins = np.linspace(-1 * before_cue_time, after_cue_time, vel.shape[0])
	
		mag = np.linalg.norm(target, axis=0)
		mag_arr = np.vstack((np.array([mag]), np.array([mag])))
		unit_targ = target / mag_arr
		mag_mat = np.tile(unit_targ, (vel.shape[0], 1, 1))
	
		# Dot prod of velocity onto unitary v
		if use_filt_vel:
			vel_in_targ_dir = np.sum(np.multiply(mag_mat, filt_vel), axis=1)
		else:
			vel_in_targ_dir = np.sum(np.multiply(mag_mat, vel), axis=1)
	
		return filt_vel, vel_bins, vel_in_targ_dir
	
	
	def get_rt_change_deriv(self, kin_sig, bins, d_vel_thres=0., fs=60):
		'''
		input:
			kin_sig: trials x time array corresponding to velocity of the cursor
	
			start_tm: time from beginning of 'bins' of which to ignore any motion (e.g. if hold 
				time is 200 ms, and your kin_sig starts at the beginning of the hold time, set 
				start_tm = 0.2 to prevent micromovements in the hold time from being captured)
	
		output: 
			kin_feat : a trl x 3 array:
				column1 = RT in units of "bins" indices
				column2 = RT in units of time (bins[column1])
				column3 = index of max of kin_sig
	
		'''
		ntrials = kin_sig.shape[0]
		kin_feat = np.zeros((ntrials, 2))
	
		# Iterate through trials
		for trl in range(ntrials):
			spd = kin_sig[trl, :]
	
			dt = 1. / fs
			d_spd = np.diff(spd, axis=0) / dt
	
			if len(np.ravel(np.nonzero(np.greater(d_spd, d_vel_thres)))) == 0:
				bin_rt = 0
			else:
				bin_rt = np.ravel(np.nonzero(np.greater(d_spd, d_vel_thres)))[0]

			kin_feat[trl, 0] = bin_rt + 1  # Index of 'RT'
			kin_feat[trl, 1] = bins[int(kin_feat[trl, 0])]  # Actual time of 'RT'
		return kin_feat
	
	
	def get_kin_sig_shenoy_method(self, kin_sig, bins, perc=.2, start_tm=.1):
		'''
		input:
			kin_sig: trials x time array corresponding to velocity of the cursor
			perc: reaction time is calculated by determining the time point at which the 
				kin_sig crosses a specific threshold. This threshold is defined by finding all
				the local maxima, then picking the highest of those (not the same as taking the 
				global maxima, because I want points that have a derivative of zero). 
	
				Then I take the point of the highest local maxima and step backwards until I cross 
				perc*highest_local_max value. If I never cross that threshold then I print a 
				statement and just take the beginning index of the kin_sig as 'rt'. This happens once
				or twice in my entire kinarm dataset. 
	
			start_tm: time from beginning of 'bins' of which to ignore any motion (e.g. if hold 
				time is 200 ms, and your kin_sig starts at the beginning of the hold time, set 
				start_tm = 0.2 to prevent micromovements in the hold time from being captured)
	
		output: 
			kin_feat : a trl x 3 array:
				column1 = RT in units of "bins" indices
				column2 = RT in units of time (bins[column1])
				column3 = index of max of kin_sig
	
		'''
		ntrials = kin_sig.shape[0]
		kin_feat = np.zeros((ntrials, 3))
		start_ix = int(start_tm * 60)
		# Iterate through trials
		for trl in range(ntrials):
			spd = kin_sig[trl, :]
	
			d_spd = np.diff(spd)
	
			# Est. number of bins RT should come after:
	
			# Find first cross from + --> -
	
			local_max_ind = np.array([i for i, s in enumerate(d_spd[:-1]) if sp.logical_and(
				s > 0, d_spd[i + 1] < 0)])  # derivative crosses zero w/ negative slope
			local_max_ind = local_max_ind[local_max_ind > start_ix]
			# How to choose:
			if len(local_max_ind) > 0:
				local_ind = np.argmax(spd[local_max_ind + 1])  # choose the biggest
				bin_max = local_max_ind[local_ind] + 1  # write down the time
	
			else:
				print(' no local maxima found -- using maximum speed point as starting pt')
				bin_max = np.argmax(spd)
	
			percent0 = spd[bin_max] * perc  # Bottom Threshold
			ind = range(0, int(bin_max))  # ix: 0 - argmax_index
			rev_ind = ind[-1:0:-1]  # Reverse
			rev_spd = spd[rev_ind]
			try:
				bin_rt = np.nonzero(rev_spd < percent0)[0][0]
			except:
				print('never falls below percent of max speed')
				bin_rt = len(rev_spd) - 1
	
			kin_feat[trl, 0] = rev_ind[bin_rt]  # Index of 'RT'
			kin_feat[trl, 1] = bins[kin_feat[trl, 0]]  # Actual time of 'RT'
			kin_feat[trl, 2] = bin_max
		return kin_feat	
	
	
	
	
	
	
	
	
	
	





def Calc_DistQlearning_3Targs(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. 
	
	This uses distributional Q-learning (Muller et al. 2024), i.e. different learning rates (alpha) for rewarded vs unrewarded trials.

	Inputs:
	- parameters: length 2 array containing the learning rate, alpha (parameters[0]), and the inverse temperate, beta (parameters[1])
	- Q_initial: length 3 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = middle-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''
	# Set Q-learning parameters
	pos_alpha = parameters[0] #for delta>0
	neg_alpha = parameters[1] #for delta<0
# 	beta = parameters[2]
	beta = 3

	# Initialize Q values. Note: Q[i] is the value on trial i before reward feedback
	Q_low = np.zeros(len(chosen_target))
	Q_mid = np.zeros(len(chosen_target))
	Q_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	Q_low[0] = Q_initial[0]
	Q_mid[0] = Q_initial[1]
	Q_high[0] = Q_initial[2]

	# Initiaialize probability values. Note: prob[i] is the probability on trial i before reward feedback
	prob_choice_low = np.zeros(len(chosen_target))
	prob_choice_mid = np.zeros(len(chosen_target))
	prob_choice_high = np.zeros(len(chosen_target))

	# Set values for first trial (indexed as trial 0)
	prob_choice_low[0] = 0.5
	prob_choice_mid[0] = 0.5
	prob_choice_high[0] = 0.5

	prob_choice_opt_lvhv = np.array([])
	prob_choice_opt_mvhv = np.array([])
	prob_choice_opt_lvmv = np.array([])

	log_prob_total = 0.
	accuracy = np.array([])

	for i in range(len(chosen_target)-1):
		# Update Q values with temporal difference error
		delta_low = float(rewards[i]) - Q_low[i]
		delta_mid = float(rewards[i]) - Q_mid[i]
		delta_high = float(rewards[i]) - Q_high[i]
		
		if rewards[i]: #rewarded trials, delta>0
			Q_low[i+1] = Q_low[i] + pos_alpha*delta_low*float(chosen_target[i]==0)
			Q_mid[i+1] = Q_mid[i] + pos_alpha*delta_mid*float(chosen_target[i]==1)
			Q_high[i+1] = Q_high[i] + pos_alpha*delta_high*float(chosen_target[i]==2)
		else: #unrewarded trials, delta<0
			Q_low[i+1] = Q_low[i] + neg_alpha*delta_low*float(chosen_target[i]==0)
			Q_mid[i+1] = Q_mid[i] + neg_alpha*delta_mid*float(chosen_target[i]==1)
			Q_high[i+1] = Q_high[i] + neg_alpha*delta_high*float(chosen_target[i]==2)

		# Update probabilities with new Q-values
		if instructed_or_freechoice[i+1] == 2:
			if np.array_equal(targets_on[i+1], [1,1,0]):
				Q_opt = Q_high[i+1]
				Q_nonopt = Q_low[i+1]

				prob_choice_low[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_low[i+1])))
				prob_choice_high[i+1] = 1. - prob_choice_low[i+1]
				prob_choice_mid[i+1] = prob_choice_mid[i]

				prob_choice_opt = prob_choice_high[i+1]
				prob_choice_nonopt = prob_choice_low[i+1]
				prob_choice_opt_lvhv = np.append(prob_choice_opt_lvhv, prob_choice_opt)

				# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
				choice = 0.5*chosen_target[i+1]+1
				# Does the predicted choice for trial i+1 match the actual choice
				accuracy = np.append(accuracy, (prob_choice_high[i+1] >= 0.5)&(chosen_target[i+1]==2) or (prob_choice_high[i+1] < 0.5)&(chosen_target[i+1]==0))

			elif np.array_equal(targets_on[i+1],[1,0,1]):
				Q_opt = Q_mid[i+1]
				Q_nonopt = Q_low[i+1]

				prob_choice_low[i+1] = 1./(1 + np.exp(beta*(Q_mid[i+1] - Q_low[i+1])))
				prob_choice_high[i+1] = prob_choice_high[i]
				prob_choice_mid[i+1] = 1. - prob_choice_low[i+1]

				prob_choice_opt = prob_choice_mid[i+1]
				prob_choice_nonopt = prob_choice_low[i+1]
				prob_choice_opt_lvmv = np.append(prob_choice_opt_lvmv, prob_choice_opt)

				# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
				choice = chosen_target[i+1]+1
				# Does the predicted choice for trial i+1 match the actual choice
				accuracy = np.append(accuracy, (prob_choice_mid[i+1] >= 0.5)&(chosen_target[i+1]==1) or (prob_choice_mid[i+1] < 0.5)&(chosen_target[i+1]==0))


			else:
				Q_opt = Q_high[i+1]
				Q_nonopt = Q_mid[i+1]

				prob_choice_mid[i+1] = 1./(1 + np.exp(beta*(Q_high[i+1] - Q_mid[i+1])))
				prob_choice_low[i+1] = prob_choice_low[i]
				prob_choice_high[i+1] = 1. - prob_choice_mid[i+1]

				prob_choice_opt = prob_choice_high[i+1]
				prob_choice_nonopt = prob_choice_mid[i+1]
				prob_choice_opt_mvhv = np.append(prob_choice_opt_mvhv, prob_choice_opt)

				# The choice on trial i+1 as either optimal (choice = 2) or nonoptimal (choice = 1)
				choice = chosen_target[i+1]

				# Does the predicted choice for trial i+1 match the actual choice
				accuracy = np.append(accuracy, (prob_choice_high[i+1] >= 0.5)&(chosen_target[i+1]==2) or (prob_choice_high[i+1] < 0.5)&(chosen_target[i+1]==1))


			log_prob_total += np.log(prob_choice_nonopt*(choice==1) + prob_choice_opt*(choice==2))

		else:
			prob_choice_low[i+1] = prob_choice_low[i]
			prob_choice_mid[i+1] = prob_choice_mid[i]
			prob_choice_high[i+1] = prob_choice_high[i]

	return Q_low, Q_mid, Q_high, prob_choice_low, prob_choice_mid, prob_choice_high, accuracy, log_prob_total


def loglikelihood_DistQlearning_3Targs(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice):
	'''
	This method finds the Q-values associated with the three target options in the probabilistic reward free-choice task
	with three targets: low-value target, middle-value target, and high-value target. The Q-values are determined based
	on a Q-learning model with temporal difference error. 
	
	This uses distributional Q-learning (Muller et al. 2024), i.e. different learning rates (alphas) for rewarded vs unrewarded trials.

	Inputs:
	- parameters: length 3 array containing the learning rates, pos_alpha (parameters[0]), neg_alpha (parameters[1]),and the inverse temperate, beta (parameters[2])
	- Q_initial: length 3 array containing the initial Q-values set for trial 1
	- chosen_target: length N array of values in range [0,2] which indicate the selected target for the trial. 0 = low-value,
						1 = middle-value, 2 = high-value.
	- rewards: length N array of boolen values indicating whether a reward was given (i.e. = 1) or not (i.e. = 0)
	- targets_on: length N array of 3-tuples which are indicators of what targets are presented. The values are arranged
					as LHM.
	'''
	Q_low, Q_mid, Q_high, prob_choice_low, prob_choice_mid, prob_choice_high, accuracy, log_prob_total = Calc_DistQlearning_3Targs(parameters, Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice)

	return log_prob_total


def DistQlearning_3Targs(chosen_target, rewards, targets_on, instructed_or_freechoice):
	#This uses distributional Q-learning (Muller et al. 2024), i.e. different learning rates (alpha) for rewarded vs unrewarded trials.
	
	# Find ML fit of alpha and beta
	Q_initial = 0.5*np.ones(3)
	nll = lambda *args: -loglikelihood_DistQlearning_3Targs(*args)
# 	x0 = [0.5, 0.5, 1] #inital guesses for pos_alpha, neg_alpha, and beta, repsectively.
# 	result = op.minimize(nll, x0, args=(Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice), bounds=[(0,1),(0,1),(0,10)],method='Nelder-Mead')#,options={'maxiter':2000})
# 	pos_alpha, neg_alpha, beta = result["x"]
	beta=3
	x0 = [0.5,0.5]
	result = op.minimize(nll, x0, args=(Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice), bounds=[(0,1),(0,1)],method='Nelder-Mead')#,options={'maxiter':2000})
	pos_alpha, neg_alpha = result["x"]
	print (f"Best fitting parameters:\n +alpha: {pos_alpha},\n -alpha: {neg_alpha},\n beta: {beta}")
	# RL model fit for Q values
	Q_low, Q_mid, Q_high, prob_choice_low, prob_choice_mid, prob_choice_high, accuracy, log_prob_total = \
		Calc_DistQlearning_3Targs([pos_alpha, neg_alpha], Q_initial, chosen_target, rewards, targets_on, instructed_or_freechoice)

	return Q_low, Q_mid, Q_high, prob_choice_low, prob_choice_mid, prob_choice_high, accuracy, log_prob_total, pos_alpha, neg_alpha, beta