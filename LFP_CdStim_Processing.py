# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:46:07 2024

@author: coleb

Process LFP data from the caudate stimulation experiment (Santacruz et al. 2017) 

Methods which process raw HDF, TDT, and syncHDF files to yield time-aligned LFP
snippets, spectrograms, PSDs, and behavioral metrics.
"""


from SortedFiles import GetFileList
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore
from scipy.signal import iirnotch, filtfilt
import statsmodels.api as sm
# import SpikeAnalysis_OfflineSortedCSVs as SpikeAnalysis
import DecisionMakingBehavior as BehaviorAnalysis
import copy
import pandas as pd
import tables
import re
import scipy.signal as signal
spect = signal.spectrogram
import tdt
from tqdm import tqdm
import time
import sys
from pandas.core.common import flatten

# import neurodsp.spectral
# import neurodsp.utils
# import neurodsp.timefrequency

# import neptune

import os
import inspect

def GetLineNumber(): #to aid w/ debugging
    return inspect.currentframe().f_back.f_lineno


#% Loop Function
def RunAllFiles(subject, funct_to_run, epoch, t_before, t_after, saveFlag, plotFlag, savefigFlag = False, 
				file_list=[], mode=[], doNeptuneFlag=False, debugFlag=False, test=None, spect_mode='wavelet', downsampleFlag=True):
	
	'''
	Loops thru the list of sessions defined in GetFileList.py to get the 
	HDF, TDT, offlineSortedSpikes, and syncHDF files for each session.
	After getting the names of the data files, funct_to_run can process the data files
	and save out processed data which can then be used by LFP_CdStim_Analysis.py.
	
	
	Parameters
	----------
	subject : 'Mario' or 'Luigi'
	funct_to_run : One of the following strings: 'Behavior' , 'TimesAlign' ,
		'Encoding' , 'Check TimesAlign', 'Check Spikes HoldCenter'
		Choose which function to call and run. See rest of file
		for more details on each function
	epoch: 'HoldCenter', 'RewardPeriod', 'MvmtPeriod', or 'RxnTime'
		Choose which part of each trial will be processed in ProcessLFP.
		See GetSamplesAlign for more details.
	t_before : float. 
		Amount of time (in seconds) before each time-alignment sample to include 
		when getting LFP snippets and spectrograms.
	t_after : float. 
		Amount of time (in seconds) after each time-alignment sample to include 
		when getting LFP snippets and spectrograms.
	saveFlag : bool. 
		True = save results as pickle files. False = do not save results
	plotFlag : bool. 
		True = plot results. False = do not plot anything
	savefigFlag : bool.
		Choose whether to save the figure to the Reward Directionality Paper folder
	file_list : list of 3 digit strs. Default is [] which runs all files
		Used to specify subset of sessions to run. If you want to run all files then set file_list=[] or leave blank
		Entries in the list are the last 3 numbers from the first hdf filename for a session
		E.g. '133' would correspond to the session of luig20170822_07_te133.hdf
		E.g. '376' would correspond to the session of luig20170929_11_te376.hdf and luig20170929_13_te378.hdf
	mode : str. 'snips', 'spects', 'psds', or None. 
		Used to select the mode when running ProcessLFP.
		'snips'= processed LFP recording snippets (voltage traces) for each trial and each channel
		'spects'= spectrograms (power vs freq vs time) for each trial and each channel
		'psds'= PSD (power vs freq) for each trial and each channel
		None= all of the above
	doNeptuneFlag : bool.
		Choose whether to track code run progress on online Neptune GUI (app.neptune.ai).
		Usually only doNeptuneFlag=True for long (>1hr) runs which I want to keep an eye on remotely.
	debugFlag : bool.
		Choose to print out data as script progresses in order to aid with debugging.
	test : str. 'chirp', 'pulses', 'sines', or False
		Used to bypass using actual data to run tests on signal processing pipeline by passing through predefined test signals.
		'chirp' is a signal that increases frequency linearly over time. Used to test processing fidelity over all freqs
		'pulses' has square wave pulses of decreasing durations. Used to test temporal resolution
		'sines' compounds sine waves of various frequencies. Used to test spectral resolution
		See scratch_chirp.py for more details. 
	spect_mode : 'wavelet' or 'sfft'
		Choose which function will be used to calculate spectrograms in ProcessLFP.
	downsampleFlag : bool.
		Choose whether raw LFP signal will be downsampled from fs=3051Hz to fs=1000Hz.
		This is to facilitate faster processing times.

	'''
	
	if doNeptuneFlag:
		run = neptune.init_run(
	    project="ValueStimulation/LFP-dataproccessing",
	    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNzU1MmJkZC1lYzA3LTRhNWEtOGRlMC05N2MyOGViYTlkMGIifQ==",
	)
		
		run["time begin"] = time.ctime(time.time())
	
	
	if subject == 'Luigi':
		num_trials_A = num_trials_B = 100
		#FR_window_hc = [0.0,1.0]
		#FR_window_rp = [0.0,1.0]
	elif subject == 'Mario':
		num_trials_A = 150
		num_trials_B = 100
		#FR_window_hc = [0.0,1.0]
		#FR_window_rp = [0.0,1.0]

	win_stay_probs = []
	lose_shift_probs = []

# 	loop_counter = 0
	for cond_counter,stim_or_sham in enumerate(['Sham','Stim']): #loop through sham and then stim data

		#get the paths to and names of files for all sessions
		paths, filenames = GetFileList(subject,stim_or_sham)
		num_sessions = len(filenames['hdf filenames'])
		

		for session in range(0,num_sessions): #loop through all stim or sham sessions
		
			#get filenames prefixed with the correct path
			hdf_files = [paths['hdf path'] + filename for filename in filenames['hdf filenames'][session]]
# 			csv_files = [[paths['spikes path'] + filename for filename in filegroup] for filegroup in filenames['spikes filenames'][session]]
			syncHDF_files = [paths['syncHDF path'] + filename for filename in filenames['syncHDF filenames'][session]]
			tdt_files = [paths['tdt path'] + filename for filename in filenames['tdt filenames'][session]]
				
			#Find external hard drive, allowing flexibility of whether data is located on drive D: or E:
			if os.path.isdir("D:\\Value Stimulation\\Data\\" + subject):
				saved_timesAlign_path = "D:\\Value Stimulation\\Data\\" + subject + "\\TimesAlign\\"
				saved_behavior_path = "D:\\Value Stimulation\\Data\\" + subject + "\\Behavior\\"
				saved_LFPsnippets_path = "D:\\Value Stimulation\\Data\\" + subject + "\\LFP\\"
			elif os.path.isdir("E:\\Value Stimulation\\Data\\" + subject):
				saved_timesAlign_path = "E:\\Value Stimulation\\Data\\" + subject + "\\TimesAlign\\"
				saved_behavior_path = "E:\\Value Stimulation\\Data\\" + subject + "\\Behavior\\"
				saved_LFPsnippets_path = "E:\\Value Stimulation\\Data\\" + subject + "\\LFP\\"
			else:
				raise FileNotFoundError('External Hard Drive not found.')
	
	
			# Only run this session if setting is for all files or if this session was specified in file_list
			hdf_file_unique_number = (hdf_files[0][-7:-4])
			if (not file_list) or (hdf_file_unique_number in file_list):
				
				#### Behavior
				if funct_to_run == 'Behavior' and subject == 'Luigi':
	
					_ = ProcessBehaviorData_2Targ(hdf_files, num_trials_A,num_trials_B, saveFlag, plotFlag, saved_behavior_path, debugFlag)
					print(f'Session Completed: {session+1}/{num_sessions}, Group: {cond_counter+1}/2')
					
				elif funct_to_run == 'Behavior' and subject == 'Mario':
				
					_ = ProcessBehaviorData_3Targ(hdf_files, num_trials_A,num_trials_B, saveFlag, plotFlag, savefigFlag, saved_behavior_path, debugFlag)
					print(f'Session Completed: {session+1}/{num_sessions}, Group: {cond_counter+1}/2')
				
					
				#### Get Win-stay Lose-shift data
				elif funct_to_run == 'Win-stay Lose-shift':
					
					filestr = GetFileStr(hdf_files[0])
					file_save_name = saved_behavior_path + 'BehaviorVectorsDict' + filestr + '.pkl'
					with open(file_save_name,'rb') as f:
						behavior_dict = pickle.load(f)
						
					win_stay_probs.append(behavior_dict['win_stay_prob'])
					lose_shift_probs.append(behavior_dict['lose_shift_prob'])
					

				#### TimesAlign
				elif funct_to_run == 'TimesAlign':
	
					GetSamplesAlign(subject, hdf_files, syncHDF_files, num_trials_A,num_trials_B, saveFlag, plotFlag, saved_timesAlign_path, debugFlag)
					print(f'Session Completed: {session+1}/{num_sessions}, Group: {cond_counter+1}/2')
					
				
				#### Get LFP snippets	
				elif funct_to_run == 'ProcessLFP': 
					
					num_files = len(hdf_files)
					assert len(hdf_files) == len(tdt_files)
					
					
					filestr = GetFileStr(hdf_files[0])
					if doNeptuneFlag:
						run["current file"] = filestr
						
					if epoch == 'hc':
						epoch_str = 'HoldCenter'
					if epoch == 'rp': 
						epoch_str = 'RewardPeriod'
					if epoch == 'mp':
						epoch_str = 'MvmtPeriod'
					if epoch == 'rt':
						epoch_str = 'RxnTime'
					
					#Get all SampsAlign files corresponding to each hdf file
					for i in range(num_files):		
						
						filestr = GetFileStr(hdf_files[i])
						
						with open(saved_timesAlign_path + epoch_str + '_SampsAlign_' + filestr + '.pkl','rb') as f:
							samps_align = pickle.load(f)
						if i==0:
							samps_align_list = [samps_align]
						else:
							samps_align_list.append(samps_align)
						
					time_begin = time.time()
					
					ProcessLFP(tdt_files,samps_align_list,epoch,t_before,t_after,
					saved_LFPsnippets_path,saveFlag,mode,debugFlag,test,spect_mode,downsampleFlag)
					
					if doNeptuneFlag:
						run["runtime for last file"] = str((time.time() - time_begin)/60) + " mins"
						print(f'runtime for {filestr}: {str((time.time() - time_begin)/60)} mins.')

				else: 
					print('funct_to_run entered incorrectly')
					return
				
				
	#### Plot Win-stay Lose-shift	
	if funct_to_run == 'Win-stay Lose-shift':	
		if plotFlag:
			avg_win_stay_prob = np.mean(win_stay_probs,axis=0)
			avg_lose_shift_prob = np.mean(lose_shift_probs,axis=0)
			
			std_win_stay_prob = np.std(win_stay_probs,axis=0)
			std_lose_shift_prob = np.std(lose_shift_probs,axis=0)
		
			fig,ax=plt.subplots()
			ax.plot(np.full_like(win_stay_probs,1),win_stay_probs,'o',color='green',alpha=0.5)
			ax.errorbar(1.2,avg_win_stay_prob,yerr=std_win_stay_prob,fmt='o',color='green',ecolor='green')	
			ax.plot(np.full_like(lose_shift_probs,2),lose_shift_probs,'o',color='red',alpha=0.5)
			ax.errorbar(2.2,avg_lose_shift_prob,yerr=std_lose_shift_prob,fmt='o',color='red',ecolor='red')	
			ax.set_xticks([1.1,2.1],['win-stay','lose-shift'])	
			ax.set_ylabel('Probability')
			ax.set_ylim([0,1])
			ax.set_xlim([0.6,2.6])
			ax.set_title(subject)
			
		return win_stay_probs, lose_shift_probs
	
	
	if doNeptuneFlag:			
		run["time end"] = time.ctime(time.time())			
		run.stop() #stop neptune logging
	
	return
			
			
#% Functions which process raw data files

def GetFileStr(filename):
	#gets the filename which is sandwiched between the first 4 digits of 
	# the subject's name and the .hdf tag.
	# had to do this way since there's so much variation in filename length
	
	if 'Mario' in filename:
		match = re.search(r'(mari.*?\.hdf)', filename)
		
	elif 'Luigi' in filename:
		match = re.search(r'(luig.*?\.hdf)', filename)
	
	if match:
		return match.group(1)
	else:
		return None




def ProcessBehaviorData_2Targ(hdf_files, num_trials_A,num_trials_B, saveFlag, plotFlag, file_path, debugFlag):
	'''
	For 2-Target Decision-making task done by Luigi (Santacruz et al. 2017)
	
	Method which takes the hdf files for a session and gets information about 
	each trial (such as if it was free/forced, rewarded/unrewarded, etc, as well as
	which target was chosen). It also computes behavioral metrics, such as the 
	subjective value of the targets and reward prediction error (RPE) for each trial. 
	These data are saved into a dictionary which can be used by LFP_CdStim_Analysis.
	
	Value is computed empirically (as opposed to using Q-Learning) 
	
	Parameters
	-------
	hdf_files : list of hdf filenames
		If more than one hdf file for a session, then files should be listed in the order in which they were saved.
	plotFlag : bool.
		if True, will produce a variety of plots displaying the data processed by this function.
	file_path : str. 
		Path to folder in which BehaviorVectorsDict file will be saved.

	Output
	-------
	behavior_dict : dict with the following keys:
		'Choice_L': 1D array. =1 for LV choices, =0 otherwise
		'Choice_H': 1D array. =1 for HV choices, =0 otherwise
		'Side': 1D array. =-1 for left choices, =1 for right choices 
		'Q_low': 1D array. Subjective value of the LV target. Range: [0,1]
		'Q_high': 1D array. Subjective value of the HV target. Range: [0,1]
		'TrialType': 1D array. =-1 for forced choices, =1 for free choices
		'num_trials_A': int. Number of trials in Block A
		'num_trials_B': int. Number of trials in Block B
		'Rewarded': 1D array. =1 if trial was rewarded, =0 if trial was not rewarded
		'PosRPE': 1D array. Subjective RPE for rewarded trials. Range: [0,1]
		'NegRPE': 1D array. Subjective RPE for unrewarded trials. Range: [-1,0]
		'WinStays': 1D array. =1 if the trial was a win-stay choice. =0 otherwise. 
			See WinStayLoseShift function for more details
		'LoseShifts': 1D array. =1 if the trial was a lose-shift choice. =0 otherwise.
			See WinStayLoseShift function for more details

	'''
	
	num_files = len(hdf_files)				
	
	#load behavior over all hdf files at once for behavioral data section
	behavior = BehaviorAnalysis.ChoiceBehavior_TwoTargets_Stimulation(hdf_files,num_trials_A,num_trials_B)
	
	# get choices, rewards, and trial type
	choices,rewards,instructed_or_freechoice = behavior.GetChoicesAndRewards()
	L_choices = np.zeros_like(choices)
	L_choices[choices==1] = 1 #1 if he chose LV, 0 otherwise
	H_choices = np.zeros_like(choices)
	H_choices[choices==2] = 1 #1 if he chose HV, 0 otherwise
	
	win_stays, lose_shifts = WinStayLoseShift(choices,instructed_or_freechoice,rewards)
	
	print('num wins: ' + str(np.nansum(~np.isnan(win_stays))))
	print('num win_stays: ' + str(np.nansum(win_stays)))
	print('num win_shifts: ' + str(np.nansum(~np.isnan(win_stays)) - np.nansum(win_stays)))
	print('num losses: ' + str(np.nansum(~np.isnan(lose_shifts))))
	print('num lose_shifts: ' + str(np.nansum(lose_shifts)))
	print('num lose_stays: ' + str(np.nansum(~np.isnan(lose_shifts)) - np.nansum(lose_shifts)))

	print('num trials: ' + str(len(rewards))	)
	print('num rewards: ' + str(sum(rewards)))
	print('num no rewards: ' + str(len(rewards) - sum(rewards)))

	
	# make time regressor 
	time = np.arange(len(choices))
	
	# calculate value of targets throughout trials
	values, win_sz = behavior.CalcValue_2Targs(choices,rewards,win_sz=10,smooth=True)
	
	# calculate RPE: rpe(t) = r(t) - Q(t). Note: Q just means value. Doesn't have to use q-learning specifically.
	rpe=[]
	for trial,choice in enumerate(choices):
		if choice==1: #lv choice
			rpe.append(rewards[trial] - values[0][trial]) #lv value
		elif choice==2: #hv choice
			rpe.append(rewards[trial] - values[1][trial]) #hv value
	rpe = np.transpose(np.array(rpe))
	
	#split rpe into positive and negative rpe signals
	pos_rpe = copy.deepcopy(rpe) #use deep copy so I can change one without changing the other
	neg_rpe = copy.deepcopy(rpe)
	pos_rpe[pos_rpe<0] = 0 #for positive rpe, set all negative rpes to zero
	neg_rpe[neg_rpe>0] = 0 #for negative rpe, set all positive rpes to zero
	
	# calculate reaction time
	for i in range(num_files):
		Calc_RT = BehaviorAnalysis.CalcRxnTime(hdf_files[i])
		rxn_time_temp, total_vel = Calc_RT.compute_rt_per_trial_FreeChoiceTask()
		
		if i==0:
			rxn_time = rxn_time_temp
		else:
			rxn_time = np.hstack((rxn_time,rxn_time_temp))
		
	# calculate movement time
	mvt_time = behavior.CalcMovtTime()
	
	# get which side (left or right) each choice was
	choice_side = behavior.GetTargetSideSelection()
	
	# get which center holds were stimulated
	stim_holds= (choices==1) * (instructed_or_freechoice==1) #get forced LV holds
	stim_holds[:100] = False #only want blB and blAp holds
	#print(stim_holds)
	for i,el in enumerate(stim_holds):
		if el:
			stim_holds[i]=1
		else:
			stim_holds[i]=0

	
	# Prob(LV) for free choices in blAp
	block_Ap_start = num_trials_A + num_trials_B
	probLV = sum((instructed_or_freechoice[block_Ap_start:]==2)*(choices[block_Ap_start:]==1)) / sum(instructed_or_freechoice[block_Ap_start:]==2)
	probLV = np.round(probLV,4)
	probHV = sum((instructed_or_freechoice[block_Ap_start:]==2)*(choices[block_Ap_start:]==2)) / sum(instructed_or_freechoice[block_Ap_start:]==2)
	probHV = np.round(probHV,4)
	
	
	# Zscore numeric data (except value and rpe), and make sure categorical data is -1/1
	choices = ((choices-1.5)*2).astype(int) # -1=LV choice, 1=HV choice
	mvt_time = zscore(mvt_time)
	rxn_time = zscore(rxn_time)
	choice_side = ((choice_side-0.5)*2).astype(int) #-1 or 1 
	instructed_or_freechoice = ((instructed_or_freechoice-1.5)*2).astype(int) #-1=instructed choice, 1=free choice
	
	#find how many free choices were in blA
	num_trials_A_free = sum(instructed_or_freechoice[:num_trials_A] == 1)
	
						
	# Put data into dict to save out
	behavior_dict = {'Choice_L':L_choices, 'Choice_H':H_choices, 'Side':choice_side, 
					  'Q_low':values[0], 'Q_high':values[1], 'TrialType':instructed_or_freechoice,
					  'num_trials_A':num_trials_A, 'num_trials_B':num_trials_B,
					  'Rewarded':rewards, 'PosRPE':pos_rpe, 'NegRPE':neg_rpe,
					  'WinStays':win_stays,'LoseShifts':lose_shifts}

	if plotFlag:

		fig = plt.figure(constrained_layout=True)
		gs = fig.add_gridspec(2, 1)
		#choices
		ax=fig.add_subplot(gs[0])
		ax.plot(behavior_dict['Choice_L'],'.',color='blue',label='LV',markersize=3)
		sliding_choices = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_L'], num_trials_slide=10)
		ax.plot(sliding_choices,color='blue')
		ax.plot(behavior_dict['Choice_H'],'.',color='green',label='HV',markersize=3)
		sliding_choices = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_H'], num_trials_slide=10)
		ax.plot(sliding_choices,color='green')
		ax.legend()
		ax.set_title('Choices')
		#values
		ax=fig.add_subplot(gs[1])
		ax.plot(behavior_dict['Q_low'],color='#34b1eb',label='LV')
		ax.plot(behavior_dict['Q_high'],color='#4ceb34',label='HV')
		ax.legend()
		ax.set_ylim([0,1])
		ax.set_title('Value')
		title = ''.join(hdf_files)
		title=title[-26:-4]
		#fig.suptitle(title + f'\nProb(LV) in Block Ap: {probLV}, Alpha = {np.round(alpha,2)}, Beta = {np.round(beta,2)}')
		fig.suptitle(title + f'\nProb(a=LV) in Block Ap: {probLV}, Value win sz: {win_sz}')
		fig.tight_layout()

		
		fig = plt.figure(constrained_layout=True)
		gs = fig.add_gridspec(2, 1)
		#side
		ax=fig.add_subplot(gs[0])
		ax.plot(behavior_dict['Side'],'.',color='#ad48db')
		ax.set_title('Side (L or R)')
		#trial type
		ax=fig.add_subplot(gs[1])
		ax.plot(behavior_dict['TrialType'],'.',color='#db48cd')
		ax.set_title('Trial Type')	
		fig.suptitle(title + f'\nProb(a=LV) in Block Ap: {probLV}, Value win sz: {win_sz}')
		fig.tight_layout()
		

		
		# figure to check RPE
		fig,axs = plt.subplots(3,1)
		#Rewards
		axs[0].plot(behavior_dict['Rewarded'],'.',color='orange')
		axs[0].set_title('Rewarded')
		#posrpe
		axs[1].plot(behavior_dict['PosRPE'],'.',color='blue')
		axs[1].set_title('PosRPE')
		#negrpe
		axs[2].plot(behavior_dict['NegRPE'],'.',color='purple')
		axs[2].set_title('NegRPE')
		
		title = ''.join(hdf_files)
		title=title[-26:-4]
		fig.suptitle(title + f'\nProb(a=LV) in Block Ap: {probLV}, Value win sz: {win_sz}')
		fig.tight_layout()
		
		
		#winstay loseshift plot
		wins = np.argwhere(behavior_dict['Rewarded']==1)
		losses = np.argwhere(behavior_dict['Rewarded']==0)
		freechoices = np.argwhere(behavior_dict['TrialType']==1)
		fig,axs = plt.subplots(2,1)
		ax = axs[0] #wins
		ax.plot(behavior_dict['WinStays'],'.g',label='win-stays')
		ax.vlines(wins,0,0.5,label='rewarded trials',color='orange')
		ax.vlines(freechoices,0.5,1,label='freechoice trials',color='k')
		ax.set_ylim([-.1,1.1])
		ax.set_xlim([0,100])
		ax.set_title('Win-stays and Win-shifts')
		ax.legend()
		ax = axs[1] #losses
		ax.plot(behavior_dict['LoseShifts'],'.r',label='lose-shifts')
		ax.vlines(losses,0,0.5,label='unrewarded trials',color='orange')
		ax.vlines(freechoices,0.5,1,label='freechoice trials',color='k')
		ax.set_ylim([-.1,1.1])
		ax.set_xlim([0,100])
		ax.set_title('Lose-stays and Lose-shifts')
		ax.legend()
		fig.tight_layout()
		
		
		## choice behavior and rewards figure for Fig 1
		#smooth choice data 
		window_length=10
		P_LV = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_L'],window_length)
# 		P_MV = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_M'],window_length)
		P_HV = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_H'],window_length)

		fig,ax = plt.subplots()

		ax.plot(P_LV,color='red',label='LV Target')
# 		ax.plot(P_MV,color='orange',label='MV Target')
		ax.plot(P_HV,color='blue',label='HV Target')

		#Get reward data for each choice
		LV_rew = np.nonzero((behavior_dict['Choice_L']==1) & (behavior_dict['Rewarded']==1))
		LV_unrew = np.nonzero((behavior_dict['Choice_L']==1) & (behavior_dict['Rewarded']==0))

# 		MV_rew = np.nonzero((behavior_dict['Choice_M']==1) & (behavior_dict['Rewarded']==1))
# 		MV_unrew = np.nonzero((behavior_dict['Choice_M']==1) & (behavior_dict['Rewarded']==0))

		HV_rew = np.nonzero((behavior_dict['Choice_H']==1) & (behavior_dict['Rewarded']==1))
		HV_unrew = np.nonzero((behavior_dict['Choice_H']==1) & (behavior_dict['Rewarded']==0))

		ymin_hi, ymax_hi = 1.1, 1.2 #for rewarded trials
		ymin_lo, ymax_lo = -0.2, -0.1 #for unrewarded trials

		ax.vlines(LV_rew,ymin_hi,ymax_hi,color='red')
		ax.vlines(LV_unrew,ymin_lo,ymax_lo,color='red')
# 		ax.vlines(MV_rew,ymin_hi,ymax_hi,color='orange')
# 		ax.vlines(MV_unrew,ymin_lo,ymax_lo,color='orange')
		ax.vlines(HV_rew,ymin_hi,ymax_hi,color='blue')
		ax.vlines(HV_unrew,ymin_lo,ymax_lo,color='blue')
		
		tick_marks = [np.mean([ymin_lo,ymax_lo]),0,0.25,0.5,0.75,1,np.mean([ymin_hi,ymax_hi])]
		tick_labels = ['Unrewarded',0,0.25,0.5,0.75,1,'Rewarded']
		ax.set_yticks(tick_marks,tick_labels)
		
		ax.set_ylabel('Choice Probability')
		ax.set_xlabel('Trials')
		
		ax.set_xlim([0,100])

		ax.legend()
		ax.set_title(title)
		fig.tight_layout()		

	#Create title from hdf file name
	filestr = GetFileStr(hdf_files[0])
	
	if saveFlag:
		
		file_save_name = file_path + 'BehaviorVectorsDict' + filestr + '.pkl'
		with open(file_save_name,'wb') as f:
			pickle.dump(behavior_dict,f)
			
		print(file_save_name + ' saved.')

	if debugFlag: 
		print(f"\nbehavior2targ {filestr}:")
		print("unique states:",np.unique(behavior.state))
		print("first 100 states:",behavior.state[:100])
		print("number of 'check_reward' states:",np.sum(behavior.state==b'check_reward'))
		input(f'Current line of code: {GetLineNumber()} \nPress enter to continue')
	
	return #behavior_dict



def ProcessBehaviorData_3Targ(hdf_files, num_trials_A,num_trials_B, saveFlag, plotFlag, savefigFlag, file_path, debugFlag):
	'''
	For 3-Target Decision-making task done by Mario (Santacruz et al. 2017)
	
	Method which takes the hdf files for a session and gets information about 
	each trial (such as if it was free/forced, rewarded/unrewarded, etc, as well as
	which target was chosen). It also computes behavioral metrics, such as the 
	subjective value of the targets and reward prediction error (RPE) for each trial. 
	These data are saved into a dictionary which can be used by LFP_CdStim_Analysis.
	
	Value is computed empirically (as opposed to using Q-Learning) 
	
	Parameters
	-------
	hdf_files : list of hdf filenames
		If more than one hdf file for a session, then files should be listed in the order in which they were saved.
	plotFlag : bool.
		if True, will produce a variety of plots displaying the data processed by this function.
	file_path : str. 
		Path to folder in which BehaviorVectorsDict file will be saved.

	Output
	-------
	behavior_dict : dict with the following keys:
		'Choice_L': 1D array. =1 for LV choices, =0 otherwise
		'Choice_M': 1D array. =1 for MV choices, =0 otherwise
		'Choice_H': 1D array. =1 for HV choices, =0 otherwise
		'Side': 1D array. =-1 for left choices, =1 for right choices 
		'Q_low': 1D array. Subjective value of the LV target. Range: [0,1]
		'Q_med': 1D array. Subjective value of the MV target. Range: [0,1]
		'Q_high': 1D array. Subjective value of the HV target. Range: [0,1]
		'TrialType': 1D array. =-1 for forced choices, =1 for free choices
		'num_trials_A': int. Number of trials in Block A
		'num_trials_B': int. Number of trials in Block B
		'Rewarded': 1D array. =1 if trial was rewarded, =0 if trial was not rewarded
		'PosRPE': 1D array. Subjective RPE for rewarded trials. Range: [0,1]
		'NegRPE': 1D array. Subjective RPE for unrewarded trials. Range: [-1,0]
		'Context': 1D array of strs. Describes which targets were offered during each trial:
			'L-H'= LV and HV targets offered
			'L-M'= LV and MV targets offered
			'M-H'= MV and HV targets offered
			''= forced choices (where only one target is offered)
		'WinStays': 1D array. =1 if the trial was a win-stay choice. =0 otherwise. 
			See WinStayLoseShift function for more details
		'LoseShifts': 1D array. =1 if the trial was a lose-shift choice. =0 otherwise.
			See WinStayLoseShift function for more details

	'''
	
	title = ''.join(hdf_files)
	title=title[-26:-4]
	
	
	num_files = len(hdf_files)				
	
	##load behavior over all hdf files at once for behavioral data section
	behavior = BehaviorAnalysis.ChoiceBehavior_ThreeTargets_Stimulation(hdf_files,num_trials_A,num_trials_B)
	
	## get choices, rewards, and trial type
	targets_on,choices,rewards,instructed_or_freechoice = behavior.GetChoicesAndRewards()	
	
	win_stays, lose_shifts = WinStayLoseShift(choices,instructed_or_freechoice,rewards)
	
	L_choices = [] #1=LV targ was chosen, 0=otherwise
	M_choices = [] #1=MV targ was chosen, 0=otherwise
	H_choices = [] #1=HV targ was chosen, 0=otherwise	
	
	context = [] #which targets were shown
	
	# Get contexts and target specific choices
	for i, choice in enumerate(choices):
		targs_presented = targets_on[i] #targs_presented is array of three boolean values: LHM

		# L-M targets presented
		if (targs_presented[0]==1) and (targs_presented[2]==1):
			context.append('L-M')
			
			if choice==1: #MV targ chosen
				H_choices.append(0)
				M_choices.append(1)
				L_choices.append(0)
				
			else: #LV targ chosen
				H_choices.append(0)
				M_choices.append(0)
				L_choices.append(1)

		# L-H targets presented
		if (targs_presented[0]==1) and (targs_presented[1]==1):
			context.append('L-H')
			
			if choice==2: #HV targ chosen
				H_choices.append(1)
				M_choices.append(0)
				L_choices.append(0)
				
			else: #LV targ chosen
				H_choices.append(0)
				M_choices.append(0)
				L_choices.append(1)

		# M-H targets presented
		if (targs_presented[1]==1) and (targs_presented[2]==1):
			context.append('M-H')
			
			if choice==2: #HV targ chosen
				H_choices.append(1)
				M_choices.append(0)
				L_choices.append(0)
				
			else: #MV targ chosen
				H_choices.append(0)
				M_choices.append(1)
				L_choices.append(0)
		
		## Forced Choices
		# L target presented
		if (targs_presented[0]==1) and (targs_presented[1]==0) and (targs_presented[2]==0):
			context.append('')

			H_choices.append(0)
			M_choices.append(0)
			L_choices.append(1)
		
		# M target presented
		if (targs_presented[0]==0) and (targs_presented[1]==0) and (targs_presented[2]==1):
			context.append('')

			H_choices.append(0)
			M_choices.append(1)
			L_choices.append(0)
			
		# H target presented
		if (targs_presented[0]==0) and (targs_presented[1]==1) and (targs_presented[2]==0):
			context.append('')

			H_choices.append(1)
			M_choices.append(0)
			L_choices.append(0)
	
	
	## calculate value of targets throughout trials
	values, win_sz = behavior.CalcValue_3Targs(choices,rewards,win_sz=10,smooth=True)
	
	## get q-learning parameters and values
	Q_low, Q_med, Q_high, prob_choice_low, prob_choice_med, prob_choice_high, accuracy, log_prob_total, pos_alpha, neg_alpha, beta = \
		BehaviorAnalysis.DistQlearning_3Targs(choices[:num_trials_A], rewards[:num_trials_A], targets_on[:num_trials_A], instructed_or_freechoice[:num_trials_A])

		
	
	## calculate RPE: rpe(t) = r(t) - Q(t). Note: Q just means value. Doesn't have to use q-learning specifically.
	rpe=[]
	for trial,choice in enumerate(choices):
		if choice==0: #lv choice
			rpe.append(rewards[trial] - values[0][trial]) #lv value
		elif choice==1: #mv choice
			rpe.append(rewards[trial] - values[1][trial]) #mv value
		elif choice==2: #hv choice
			rpe.append(rewards[trial] - values[2][trial]) #hv value
	rpe = np.transpose(np.array(rpe))
	
	## split rpe into positive and negative rpe signals
	pos_rpe = copy.deepcopy(rpe) #use deep copy so I can change one without changing the other
	neg_rpe = copy.deepcopy(rpe)
	pos_rpe[pos_rpe<0] = 0 #for positive rpe, set all negative rpes to zero
	neg_rpe[neg_rpe>0] = 0 #for negative rpe, set all positive rpes to zero
	
	## calculate reaction time
	for i in range(num_files):
		Calc_RT = BehaviorAnalysis.CalcRxnTime(hdf_files[i])
		rxn_time_temp, total_vel = Calc_RT.compute_rt_per_trial_FreeChoiceTask()
		
		if i==0:
			rxn_time = rxn_time_temp
		else:
			rxn_time = np.hstack((rxn_time,rxn_time_temp))
		
	## calculate movement time
	mvt_time = behavior.CalcMovtTime()
	
	## get which side (left or right) each choice was
	choice_side = behavior.GetTargetSideSelection()
	
	# Prob(LV) for free choices in blAp
	block_Ap_start = num_trials_A + num_trials_B
	probMV = sum((instructed_or_freechoice[block_Ap_start:]==2)*(choices[block_Ap_start:]==1)) / sum(instructed_or_freechoice[block_Ap_start:]==2)
	probMV = np.round(probMV,4)
	probHV = sum((instructed_or_freechoice[block_Ap_start:]==2)*(choices[block_Ap_start:]==2)) / sum(instructed_or_freechoice[block_Ap_start:]==2)
	probHV = np.round(probHV,4)
	
	
	# Zscore numeric data (except value and rpe), and make sure categorical data is -1/1
	#choices = ((np.array(all_choices)-1.5)*2).astype(int) # -1=non-optimal choice, 1=optimal choice
	#L_choices = ((np.array(L_choices)-.5)*2).astype(int) # -1=did not choose LV, 1=did choose LV
	#M_choices = ((np.array(M_choices)-.5)*2).astype(int) # -1=did not choose MV, 1=did choose MV
	#H_choices = ((np.array(H_choices)-.5)*2).astype(int) # -1=did not choose HV, 1=did choose HV
	mvt_time = zscore(mvt_time)
	rxn_time = zscore(rxn_time)
	instructed_or_freechoice = ((instructed_or_freechoice-1.5)*2).astype(int) #free=1, forced=-1
	#choice_side = ((choice_side-0.5)*2).astype(int) # this is already -1 or 1 
	
	#find how many free choices were in blA
	num_trials_A_free = sum(instructed_or_freechoice[:num_trials_A] == 2)



	#Put data into dict to save out
	behavior_dict = {'Choice_L':np.array(L_choices), 'Choice_M':np.array(M_choices), 'Choice_H':np.array(H_choices), 'Side':choice_side, 
					  'Q_low':values[0], 'Q_med':values[1], 'Q_high':values[2], 'TrialType':instructed_or_freechoice,
					  'num_trials_A':num_trials_A, 'num_trials_B':num_trials_B,
					  'Rewarded':rewards, 'PosRPE':pos_rpe, 'NegRPE':neg_rpe, 'Context':np.array(context),
					  'WinStays':win_stays,'LoseShifts':lose_shifts}


	if plotFlag:

		
				
		##Q-learning modeling vs empirical plots
	
		#HV
		colors = ['#048204','#68ED53'] #empirical, model
		BlA_freechoices = behavior_dict['TrialType'][:num_trials_A]==1
		modeled_value, modeled_choiceprob = Q_high, prob_choice_high[BlA_freechoices]
		emp_value, emp_choiceprob = behavior_dict['Q_high'][:num_trials_A], behavior_dict['Choice_H'][:num_trials_A][BlA_freechoices]
		fig,axs=plt.subplots(1,2)
		ax=axs[0] #free choice prob
		sliding_freechoices = BehaviorAnalysis.trial_sliding_avg(emp_choiceprob, num_trials_slide=10)
		ax.plot(sliding_freechoices,color=colors[0],label='Empirical')
		ax.plot(modeled_choiceprob,color=colors[1],label='Model')
		ax.set_xlabel('Free Choice Trials')
		ax.set_ylim([0,1])
		ax.set_ylabel('Choice Probability')
		ax=axs[1] #value
		ax.plot(emp_value,color=colors[0],label='Empirical')
		ax.plot(modeled_value,color=colors[1],label='Model')
		ax.set_xlabel('All Trials')
		ax.set_ylim([0,1])
		ax.set_ylabel('Value')
		ax.legend()
		fig.suptitle(f"Best fitting parameters:\n +alpha: {round(pos_alpha,3)}, -alpha: {round(neg_alpha,3)}, beta: {round(beta,3)}")
		fig.tight_layout()
		
		#MV
		colors = ['#f7aa02','#f7e06d'] #empirical, model
		modeled_value, modeled_choiceprob = Q_med, prob_choice_med[BlA_freechoices]
		emp_value, emp_choiceprob = behavior_dict['Q_med'][:num_trials_A], behavior_dict['Choice_M'][:num_trials_A][BlA_freechoices]
		fig,axs=plt.subplots(1,2)
		ax=axs[0] #free choice prob
		sliding_freechoices = BehaviorAnalysis.trial_sliding_avg(emp_choiceprob, num_trials_slide=10)
		ax.plot(sliding_freechoices,color=colors[0],label='Empirical')
		ax.plot(modeled_choiceprob,color=colors[1],label='Model')
		ax.set_xlabel('Free Choice Trials')
		ax.set_ylim([0,1])
		ax.set_ylabel('Choice Probability')
		ax=axs[1] #value
		ax.plot(emp_value,color=colors[0],label='Empirical')
		ax.plot(modeled_value,color=colors[1],label='Model')
		ax.set_xlabel('All Trials')
		ax.set_ylim([0,1])
		ax.set_ylabel('Value')
		ax.legend()
		fig.suptitle(f"Best fitting parameters:\n +alpha: {round(pos_alpha,3)}, -alpha: {round(neg_alpha,3)}, beta: {round(beta,3)}")
		fig.tight_layout()
		
		#LV
		colors = ['#0000FF','#26ACEA'] #empirical, model
		modeled_value, modeled_choiceprob = Q_low, prob_choice_low[BlA_freechoices]
		emp_value, emp_choiceprob = behavior_dict['Q_low'][:num_trials_A], behavior_dict['Choice_L'][:num_trials_A][BlA_freechoices]
		fig,axs=plt.subplots(1,2)
		ax=axs[0] #free choice prob
		sliding_freechoices = BehaviorAnalysis.trial_sliding_avg(emp_choiceprob, num_trials_slide=10)
		ax.plot(sliding_freechoices,color=colors[0],label='Empirical')
		ax.plot(modeled_choiceprob,color=colors[1],label='Model')
		ax.set_xlabel('Free Choice Trials')
		ax.set_ylim([0,1])
		ax.set_ylabel('Choice Probability')
		ax=axs[1] #value
		ax.plot(emp_value,color=colors[0],label='Empirical')
		ax.plot(modeled_value,color=colors[1],label='Model')
		ax.set_xlabel('All Trials')
		ax.set_ylim([0,1])
		ax.set_ylabel('Value')
		ax.legend()
		fig.suptitle(f"Best fitting parameters:\n +alpha: {round(pos_alpha,3)}, -alpha: {round(neg_alpha,3)}, beta: {round(beta,3)}")
		fig.tight_layout()
		
		
		
		##choices
		fig = plt.figure(constrained_layout=True)
		gs = fig.add_gridspec(3, 1)
		ax=fig.add_subplot(gs[0])
		ax.plot(behavior_dict['Choice_H'],'.',color='green',label='HV')
		sliding_choices = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_H'], num_trials_slide=10)
		ax.plot(sliding_choices,color='green')
		ax.set_title('HV Choices')
		ax=fig.add_subplot(gs[1])
		ax.plot(behavior_dict['Choice_M'],'.',color='orange',label='MV')
		sliding_choices = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_M'], num_trials_slide=10)
		ax.plot(sliding_choices,color='orange')
		ax.set_title('MV Choices')
		ax=fig.add_subplot(gs[2])
		ax.plot(behavior_dict['Choice_L'],'.',color='blue',label='LV')
		sliding_choices = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_L'], num_trials_slide=10)
		ax.plot(sliding_choices,color='blue')
		ax.set_title('LV Choices')
		#fig.suptitle(title + f'\nProb(LV) in Block Ap: {probLV}, Alpha = {np.round(alpha,2)}, Beta = {np.round(beta,2)}')
		fig.suptitle(title + f'\nProb(a=MV) in Block Ap: {probMV}, Value win sz: {win_sz}')
		fig.tight_layout()

		
		##values
		fig = plt.figure(constrained_layout=True)
		gs = fig.add_gridspec(2, 1)
		ax=fig.add_subplot(gs[0])
		ax.plot(behavior_dict['Q_high'],color='#4ceb34',label='HV')
		ax.plot(behavior_dict['Q_med'],color='orange',label='MV')
		ax.plot(behavior_dict['Q_low'],color='#34b1eb',label='LV')
		ax.legend()
		ax.set_ylim([0,1])
		ax.set_title('Value')
		fig.suptitle(title + f'\nProb(a=MV) in Block Ap: {probMV}, Value win sz: {win_sz}')
		fig.tight_layout()

		
		fig = plt.figure(constrained_layout=True)
		gs = fig.add_gridspec(2, 1)
		#side
		ax=fig.add_subplot(gs[0])
		ax.plot(behavior_dict['Side'],'.',color='#ad48db')
		ax.set_title('Side (L or R)')
		#trial type
		ax=fig.add_subplot(gs[1])
		ax.plot(behavior_dict['TrialType'],'.',color='#db48cd')
		ax.set_title('Trial Type')	
		fig.suptitle(title + f'\nProb(a=MV) in Block Ap: {probMV}, Value win sz: {win_sz}')
		fig.tight_layout()
		
		
		# figure to check RPE
		fig,axs = plt.subplots(3,1)
		#Rewards
		axs[0].plot(behavior_dict['Rewarded'],'.',color='orange')
		axs[0].set_title('Rewarded')
		#posrpe
		axs[1].plot(behavior_dict['PosRPE'],'.',color='blue')
		axs[1].set_title('PosRPE')
		#negrpe
		axs[2].plot(behavior_dict['NegRPE'],'.',color='purple')
		axs[2].set_title('NegRPE')
		
		title = ''.join(hdf_files)
		title=title[-26:-4]
		fig.suptitle(title + f'\nProb(a=MV) in Block Ap: {probMV}, Value win sz: {win_sz}')
		fig.tight_layout()
		
		
		#winstay loseshift plot
		wins = np.argwhere(behavior_dict['Rewarded']==1)
		losses = np.argwhere(behavior_dict['Rewarded']==0)
		freechoices = np.argwhere(behavior_dict['TrialType']==1)
		fig,axs = plt.subplots(2,1)
		ax = axs[0] #wins
		ax.plot(behavior_dict['WinStays'],'.g',label='win-stays')
		ax.vlines(wins,0,0.5,label='rewarded trials',color='orange')
		ax.vlines(freechoices,0.5,1,label='freechoice trials',color='k')
		ax.set_ylim([-.1,1.1])
		ax.set_xlim([0,100])
		ax.set_title('Win-stays and Win-shifts')
		ax.legend()
		ax = axs[1] #losses
		ax.plot(behavior_dict['LoseShifts'],'.r',label='lose-shifts')
		ax.vlines(losses,0,0.5,label='unrewarded trials',color='orange')
		ax.vlines(freechoices,0.5,1,label='freechoice trials',color='k')
		ax.set_ylim([-.1,1.1])
		ax.set_xlim([0,100])
		ax.set_title('Lose-stays and Lose-shifts')
		ax.legend()
		fig.tight_layout()
		
		
		# choice behavior and rewards figure for Fig 1
		#smooth choice data 
		window_length=20
		P_LV = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_L'],window_length)
		P_MV = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_M'],window_length)
		P_HV = BehaviorAnalysis.trial_sliding_avg(behavior_dict['Choice_H'],window_length)

		fig,ax = plt.subplots()
		
		trial_begin=0
		trial_end=150
		
		ax.plot(P_LV,color='red',label='LV')
		ax.plot(P_MV,color='gold',label='MV')
		ax.plot(P_HV,color='blue',label='HV')

		#Get reward data for each choice
		LV_rew = np.nonzero((behavior_dict['Choice_L'][trial_begin:trial_end]==1) & (behavior_dict['Rewarded'][trial_begin:trial_end]==1))
		LV_unrew = np.nonzero((behavior_dict['Choice_L'][trial_begin:trial_end]==1) & (behavior_dict['Rewarded'][trial_begin:trial_end]==0))

		MV_rew = np.nonzero((behavior_dict['Choice_M'][trial_begin:trial_end]==1) & (behavior_dict['Rewarded'][trial_begin:trial_end]==1))
		MV_unrew = np.nonzero((behavior_dict['Choice_M'][trial_begin:trial_end]==1) & (behavior_dict['Rewarded'][trial_begin:trial_end]==0))

		HV_rew = np.nonzero((behavior_dict['Choice_H'][trial_begin:trial_end]==1) & (behavior_dict['Rewarded'][trial_begin:trial_end]==1))
		HV_unrew = np.nonzero((behavior_dict['Choice_H'][trial_begin:trial_end]==1) & (behavior_dict['Rewarded'][trial_begin:trial_end]==0))

		ax2 = ax.twinx()
		ymin_hi, ymax_hi = 1.05, 1.15 #for rewarded trials
		ymin_lo, ymax_lo = -0.15, -0.05 #for unrewarded trials
		buffer=0.05
		

		
		ax2.vlines(LV_rew,ymin_hi,ymax_hi,color='red',clip_on=False)
		ax2.vlines(LV_unrew,ymin_lo,ymax_lo,color='red',clip_on=False)
		ax2.vlines(MV_rew,ymin_hi,ymax_hi,color='gold',clip_on=False)
		ax2.vlines(MV_unrew,ymin_lo,ymax_lo,color='gold',clip_on=False)
		ax2.vlines(HV_rew,ymin_hi,ymax_hi,color='blue',clip_on=False)
		ax2.vlines(HV_unrew,ymin_lo,ymax_lo,color='blue',clip_on=False)
		
		#Format fig
		labelsize=14
		
		
		tick_marks = [0,0.25,0.5,0.75,1]
		tick_labels = [0,0.25,0.5,0.75,1]
		ax.set_yticks(tick_marks,tick_labels)
		ax.yaxis.set_tick_params(labelsize=labelsize)
		ax.set_ylabel('Choice Probability',fontsize=labelsize)
		ax.set_ylim([ymin_lo-buffer,ymax_hi])
		
		tick_marks = [np.mean([ymin_lo,ymax_lo]),np.mean([ymin_hi,ymax_hi])]
		tick_labels = ['No\nReward','Reward']
# 		tick_labels = []
		ax2.set_yticks(tick_marks,tick_labels)
		ax2.yaxis.set_tick_params(labelsize=labelsize)
		ax2.set_ylim([ymin_lo-buffer,ymax_hi])
		
		ax.set_xlabel('Trials',fontsize=labelsize)
		ax.xaxis.set_tick_params(labelsize=labelsize)
		ax.set_xlim([trial_begin,trial_end]) #limit to block A
		ax2.set_xlim([trial_begin,trial_end]) #limit to block A
		
		ax.hlines([0,1],trial_begin,trial_end,'k',linewidth=1)
		ax.vlines([trial_begin,trial_end],0,1,'k',linewidth=1,clip_on=False)
		ax.spines[['top','right','left']].set_visible(False)
		ax2.spines[['top','right','left']].set_visible(False)

		ax.legend(fontsize=labelsize, ncol=3,loc='upper center', bbox_to_anchor=(0.5, 1.2))
# 		ax.set_title(title)
		fig.tight_layout()
		
		if savefigFlag: plt.savefig(r"C:\Users\coleb\Desktop\Santacruz Lab\Value Stimulation\Reward Directionality Paper\Figures\Fig1\BehaviorAndRewards.svg")


	#Create title from hdf file name
	filestr = GetFileStr(hdf_files[0])
	
	if saveFlag:
		
		file_save_name = file_path + 'BehaviorVectorsDict' + filestr + '.pkl'
		with open(file_save_name,'wb') as f:
			pickle.dump(behavior_dict,f)
		print(file_save_name + ' saved.')

	if debugFlag: 
		print(f"\nbehavior3targ {filestr}:")
		print("unique states:",np.unique(behavior.state))
		print("first 100 states:",behavior.state[:100])
		print("number of 'check_reward' states:",np.sum(behavior.state==b'check_reward'))
		input(f'Current line of code: {GetLineNumber()} \nPress enter to continue')
	
	return #behavior_dict


# def WinStayLoseShift_3Targ(choice_char,X_choices,instructed_or_freechoice,rewards,context):
# 	
# 	#initialize vector
# 	win_stays = np.full_like(X_choices,np.nan) #0=win shift, 1=win stay, nan=not applicable
# 	lose_shifts = np.full_like(X_choices,np.nan) #0=lose stay, 1=lose shift, nan=not applicable
# 	
# 	#get all the trials where the target was offered
# 	X_offered = choice_char in context
# 	X_offered_ind = np.argwhere(X_offered)
# 	
# 	for i,ind in enumerate(X_offered_ind): #loop thru all trials where the targ was offered
# 	
# 		if i == 0: #need to compare to prev trials, so cant use first trial
# 			continue
# 		
# 		prev_ind = X_offered_ind[i-1]
# 	
# 		if instructed_or_freechoice[ind] == 1: #if trial is freechoice
# 		
# 			if X_choices[prev_ind] == 1: #if the target was chosen the last time it was offered
# 			
# 				win = bool(rewards[prev_ind] == 1)
# 				lose = bool(rewards[prev_ind] == 0)
# 				stay = bool(X_choices[ind] == 1)
# 				shift = bool(X_choices[ind] == 0)
# 				
# 				if win and stay:
# 					win_stays[ind] = 1
# 				if win and shift:
# 					win_stays[ind] = 0
# 				if lose and stay:
# 					lose_shifts[ind] = 0
# 				if lose and shift:
# 					lose_shifts[ind] = 1
# 				
# 	return win_stays, lose_shifts




def WinStayLoseShift(choices,instructed_or_freechoice,rewards):
	'''
	Win-stay lose-shift behavior is a function of analyzing decision making behavior.
	A win-stay choice is when the previous choice led to a reward, and the current choice is to the same target.
	A lose-shift choice is when the previous choice was not rewarded, and the current choice is to a different target.
	
	This function only takes into account the trial immediately previous to the current trial.
	If the target chosen in the prev trial is not offered during current trial, then 
	the current trial is not classified as a win-stay nor a lose-shift. See 01-08-25 slides for explanation.
	
	
	Parameters
	----------
	Outputs from DecisionMakingBehavior.GetChoicesAndRewards()

	Returns
	-------
	win_stays: 1D array. =1 if the trial was a win-stay choice. =0 otherwise. 
	lose_shifts: 1D array. =1 if the trial was a lose-shift choice. =0 otherwise.

	'''
	#initialize vectors to record the behavior for each trial
	win_stays = np.full_like(choices,np.nan,dtype=float) #0=win shift, 1=win stay, nan=not applicable
	lose_shifts = np.full_like(choices,np.nan,dtype=float) #0=lose stay, 1=lose shift, nan=not applicable
	
	for trial in range(len(choices)):
	
		if trial > 0: #need to compare to prev trials, so cant use first trial
			if instructed_or_freechoice[trial]==2: #use only free choices
			
				win = bool(rewards[trial-1] == 1)
				lose = bool(rewards[trial-1] == 0)
				stay = bool(choices[trial] == choices[trial-1])
				shift = bool(choices[trial] != choices[trial-1])
				
				if win and stay:
					win_stays[trial] = 1
				if win and shift:
					win_stays[trial] = 0
				if lose and stay:
					lose_shifts[trial] = 0
				if lose and shift:
					lose_shifts[trial] = 1
					
	return win_stays, lose_shifts


	
def GetSamplesAlign(subject, hdf_files, syncHDF_files, num_trials_A,num_trials_B, saveFlag, plotFlag, file_path, debugFlag):
	'''
	Gets the array of indices (sample numbers) corresponding to the hold_center, 
	target, and check_reward time points of the given session.
	This facilitates time-aligned analyses.
	
	Parameters
	----------
	hdf_files : list of hdf files for a single session
	syncHDF_files : list of syncHDF_files files which are used to make the alignment between behavior data and spike data
	num_trials_A : int. number of trials in the A block. 100 for a two target task, 150 for a three target task
	num_trials_B : int. number of trials in the B block. 100 for both Mario and Luigi
	saveFlag : bool. True = save results as pickle files in directory specified by file_path. False = do not save results
	plotFlag : bool. True = plot results. False = do not plot anything
	file_path : str. Path where to save out data.
		
	Returns
	-------
	hold_center_TDT_ind : 1D array containing the TDT indices for the center hold onset times of the given session
	reward_period_TDT_ind : 1D array containing the TDT indices for the check_reward times of the given session
	mvmt_period_TDT_ind : 1D array containing the TDT indices for the target prompt times of the given session
	rxn_time_TDT_ind : 1D array containing the TDT indices for the target prompt time plus the reaction time.
		This should correspond more precisely to the start of joystick movement.
	'''
	
	assert len(hdf_files) == len(syncHDF_files), f'Number of hdf files does not match number of syncHDF files! {hdf_files}'
	num_files = len(hdf_files)
	
	if plotFlag:
		fig1,ax1=plt.subplots() #times
		fig2,ax2=plt.subplots() #diff btwn times
		fig3,ax3=plt.subplots() #diff btwn diffs btwn times
		
	
	end_of_prev_file_hdf = 0 
	end_of_prev_file_TDT = 0
	
	fs_hdf = 60 #hdf fs is always 60
	
	for i in range(num_files):
	
		# load behavior data
		if num_files>1: # if more than one file in a session, do each individually since there is a syncHDF for each individual hdf file. filename needs to be in a list for BehaviorAnalysis
			if subject == 'Luigi':
				cb = BehaviorAnalysis.ChoiceBehavior_TwoTargets_Stimulation([hdf_files[i]],num_trials_A,num_trials_B)
			elif subject == 'Mario':
				cb = BehaviorAnalysis.ChoiceBehavior_ThreeTargets_Stimulation([hdf_files[i]],num_trials_A,num_trials_B)
		else:
			if subject == 'Luigi':
				cb = BehaviorAnalysis.ChoiceBehavior_TwoTargets_Stimulation(hdf_files,num_trials_A,num_trials_B)
			elif subject == 'Mario':
				cb = BehaviorAnalysis.ChoiceBehavior_ThreeTargets_Stimulation(hdf_files,num_trials_A,num_trials_B)
		
		# Find times of successful trials
		ind_hold_center = cb.ind_check_reward_states - 4 #times corresponding to hold center onset
		ind_mvmt_period = cb.ind_check_reward_states - 3 #times corresponding to target prompt
		ind_reward_period = cb.ind_check_reward_states #times corresponding to reward period onset
		
		# align spike tdt times with hold center hdf indices using syncHDF files
		hold_center_TDT_ind, DIO_freq = BehaviorAnalysis.get_HDFstate_TDT_LFPsamples(ind_hold_center,cb.state_time,syncHDF_files[i])
		reward_period_TDT_ind, DIO_freq1 = BehaviorAnalysis.get_HDFstate_TDT_LFPsamples(ind_reward_period,cb.state_time,syncHDF_files[i])	
		mvmt_period_TDT_ind, DIO_freq2 = BehaviorAnalysis.get_HDFstate_TDT_LFPsamples(ind_mvmt_period,cb.state_time,syncHDF_files[i])
		
		# make sure tdt sampling rate is same for all
		assert DIO_freq==DIO_freq1==DIO_freq2
		
		# Ensure that the hdf indexes we take actually correspond to the correct states
		assert (np.unique(cb.state[ind_hold_center]) == np.array([b'hold_center', b'hold_center_and_stimulate'])).all() or np.unique(cb.state[ind_hold_center]) == np.array([b'hold_center'])
		assert np.unique(cb.state[ind_mvmt_period]) == np.array([b'target'])
		assert np.unique(cb.state[ind_reward_period]) == np.array([b'check_reward'])

		# Ensure that we have a 1 to 1 correspondence btwn indices we put in and indices we got out.
		assert len(hold_center_TDT_ind) == len(ind_hold_center), f'Repeat hold times! {hdf_files}'	
		assert len(reward_period_TDT_ind) == len(ind_reward_period), f'Repeat hold times! {hdf_files}'
		assert len(mvmt_period_TDT_ind) == len(ind_mvmt_period), f'Repeat hold times! {hdf_files}'
				
		if plotFlag:
			
			tdt_times = reward_period_TDT_ind / DIO_freq / 60  + end_of_prev_file_TDT #convert from samples to seocnds to minutes
			hdf_times = (cb.state_time[ind_reward_period]) / fs_hdf / 60 + end_of_prev_file_hdf #convert from samples to seocnds to minutes
			end_of_prev_file_hdf = hdf_times[-1]
			end_of_prev_file_TDT = tdt_times[-1]
			
			ax=ax1
			ax.set_title('TDT and HDF clock alignment')
			ax.set_xlabel('Time of Reward Period on TDT clock (min)')
			ax.set_ylabel('HDF clock time (min)')
			ax.plot(tdt_times,hdf_times,'o',alpha=0.5)
			ax.plot([0,np.max(tdt_times)],[0,np.max(tdt_times)],'k--') #unity line

			ax=ax2
			ax.set_title('TDT and HDF clock difference')
			ax.set_xlabel('Time of Reward Period on TDT clock (min)')
			ax.set_ylabel('Diff btwn TDT and HDF clocks (sec)')
			ax.plot(tdt_times,(tdt_times-hdf_times)*60,'o')
			
			ax=ax3
			ax.set_title('rate of change of TDT and HDF clock difference')
			ax.set_xlabel('Time of Reward Period on TDT clock (min)')
			ax.set_ylabel('Diff btwn TDT and HDF clock differences (hdf samples)')
			ax.plot(tdt_times[:-1],np.diff(tdt_times-hdf_times)*60*fs_hdf,'.')
			ax.hlines(1,tdt_times[0],tdt_times[-2],linestyle='--',color='tab:orange',label='time of one hdf sample')
			ax.legend()
			
			fig1.suptitle(GetFileStr(hdf_files[0]))# + '\nAllTrials')
			fig1.tight_layout()
			fig2.suptitle(GetFileStr(hdf_files[0]))# + '\nAllTrials')
			fig2.tight_layout()
			fig3.suptitle(GetFileStr(hdf_files[0]))# + '\nAllTrials')
			fig3.tight_layout()
		
		## Make a SampsAlign for when mvmt actually begins 
		
		# Get rxn times for all trials
		Rxn = BehaviorAnalysis.CalcRxnTime(hdf_files[i])
		rxn_times,_ = Rxn.compute_rt_per_trial_FreeChoiceTask()
		assert len(rxn_times) == len(mvmt_period_TDT_ind)
		#print(rxn_times)
		
		# convert rxn time to samples
		rxn_times_TDT = rxn_times * DIO_freq
		
		#add rxn times to the mvmt period alignment samples
		rxn_time_TDT_ind = np.add(mvmt_period_TDT_ind, rxn_times_TDT)
		
		if plotFlag:
			fig,ax = plt.subplots()
			ax.plot(mvmt_period_TDT_ind/DIO_freq/60,rxn_times,'o')
			ax.set_xlabel('Time of Target prompt (min)')
			ax.set_ylabel('Rxn Time (s)')
				
			fig,ax = plt.subplots()
			ax.hist(rxn_times,bins=15)
			ax.set_xlabel('Rxn time (s)')
			ax.set_ylabel('count')
		
				
		#Create title from hdf file name
		filestr = GetFileStr(hdf_files[i])
		
		if saveFlag:
			
			file_save_name = file_path + 'HoldCenter_SampsAlign_' + filestr + '.pkl'
			with open(file_save_name,'wb') as f:
				pickle.dump(hold_center_TDT_ind,f)
			print(file_save_name + ' saved.')
			
			file_save_name = file_path + 'RewardPeriod_SampsAlign_' + filestr + '.pkl'
			with open(file_save_name,'wb') as f:
				pickle.dump(reward_period_TDT_ind,f)
			print(file_save_name + ' saved.')	
			
			file_save_name = file_path + 'MvmtPeriod_SampsAlign_' + filestr + '.pkl'
			with open(file_save_name,'wb') as f:
				pickle.dump(mvmt_period_TDT_ind,f)
			print(file_save_name + ' saved.')
			
			file_save_name = file_path + 'RxnTime_SampsAlign_' + filestr + '.pkl'
			with open(file_save_name,'wb') as f:
				pickle.dump(rxn_time_TDT_ind,f)
			print(file_save_name + ' saved.')
			
		if debugFlag: 
			print(f'\n\nGetSamplesAlign, {filestr}, ({i+1}/{num_files}):')
			print(f'\nhc: {type(hold_center_TDT_ind)} {np.shape(hold_center_TDT_ind)}')#\n{hold_center_TDT_ind}')
			print(f'\nmp: {type(mvmt_period_TDT_ind)} {np.shape(mvmt_period_TDT_ind)}')#\n{mvmt_period_TDT_ind}')
			print(f'\nrp: {type(reward_period_TDT_ind)} {np.shape(reward_period_TDT_ind)}')#\n{reward_period_TDT_ind}')
			input(f'Current line of code: {GetLineNumber()} \nPress enter to continue')

	return 


# def ComputePSD(snippet,fs):
# 	'''
# 	Compute the power spectral density (PSD, V**2/Hz) vs freq curve for a given snippet of LFP signal.
# 	
# 	Uses neurodsp package (Voytek lab) to compute PSD.
# 	
# 	
# 	Input
# 	-------
# 	snippet: 1D array. LFP activity over which to compute PSD.
# 	fs: float. Sampling Frequency of LFP recording.
# 	
# 	Output
# 	-------
# 	f: 1D array. Array of frequencies corresponding to computed PSD values
# 	psd: 1D array. Array of PSD values for each frequency given in f.
# 	
# 	'''
# 	f = neurodsp.utils.data.create_freqs(4.,200.,2.) #4-200Hz w 2Hz resolution
# 	n_cycles = f/4
# 	f, psd = neurodsp.spectral.compute_spectrum(snippet, fs, method='wavelet',freqs=f,n_cycles=n_cycles)
# 	f, psd = neurodsp.spectral.trim_spectrum(f,psd,[4.,200.])
# 	
# 	return f,psd


# def ComputeSpectrogram_CWT(snippet,fs,t_before,t_after):
# 	'''
# 	Compute the power spectral density (PSD, V**2/Hz) over frequency and time (aka spectrogram)
# 	for a given time-aligned snippet of LFP signal.
# 	
# 	Uses neurodsp package (Voytek lab) to compute spectrogram.
# 	
# 	Input
# 	-------
# 	snippet: 1D array. LFP activity over which to compute PSD.
# 	fs: float. Sampling Frequency of LFP recording.
# 	t_before: float. How far snippet extends before time-alignment point. Used for time-alignment of the spectrogram.
# 	t_after: float. How far snippet extends after time-alignment point. Used for time-alignment of the spectrogram.
# 	
# 	Output
# 	-------
# 	f: 1D array. Array of frequencies corresponding to computed PSD values
# 	t: 1D array. Array of times corresponding to computed PSD values
# 	Sxx: 2D array. Array of PSD values for each frequency and time point given by f and t.
# 	
# 	'''
# 	
# 	f = neurodsp.utils.data.create_freqs(1.,100.,1.) #1-100Hz w 1Hz resolution
# 	t = neurodsp.utils.data.create_times(t_before+t_after,fs,start_val=-t_before)
# 	n_cycles = f/4
# 	
# 	#if there's a rounding error making the snippet and time vector off by 1 sample
# 	if (len(t) != len(snippet)) and (abs(len(t)-len(snippet)) == 1): 
# 		t=np.resize(t,np.shape(snippet)) #make to be the same size
# 		
# 	Sxx = neurodsp.timefrequency.compute_wavelet_transform(snippet,fs,f,n_cycles=n_cycles)
# # 	f,t,Sxx = neurodsp.spectral.trim_spectrogram(f,t,Sxx,f_range=None,t_range=[-0.2,1.]) #trim spectrogram to only be over hold
# # 	#downsample along time dim because 500Hz of temporal res isn't needed and will make file size humongous
# 	t = t[::5] #downsample by factor of 5, giving resulting temporal res of 10ms or 100hz, which should be sufficient for analysis.
# 	Sxx = Sxx[:,::5]

# # 	Sxx = np.array(Sxx, dtype='complex64') #change precision to reduce memory demand
# 	Sxx = np.array(abs(Sxx)) #take abs to get rid of imag component to reduce memory demand
# 	
# 	assert np.shape(Sxx) == (len(f),len(t))
# 	
# # 	print(Sxx.shape)
# # 	xxx
# 	
# 	return f,t,Sxx #Sxx = spectrogram for snippet
    

# def ComputeSpectrogram_NoDownsampling(snippet,fs,t_before,t_after):
# 	
# 	f = neurodsp.utils.data.create_freqs(0.,200.,2.) #4-60Hz w 2Hz resolution
# 	t = neurodsp.utils.data.create_times(t_before+t_after,fs,start_val=-t_before)
# 	n_cycles = f/4
# 	
# 	if not len(t) == len(snippet): #if there's a rounding error making the snippet and time vector off by 1 sample
# 		assert abs(len(t)-len(snippet)) == 1
# 		t=np.resize(t,np.shape(snippet)) #make to be the same size
# 		
# 	Sxx = neurodsp.timefrequency.compute_wavelet_transform(snippet,fs,f,n_cycles=n_cycles)
# # 	f,t,Sxx = neurodsp.spectral.trim_spectrogram(f,t,Sxx,f_range=None,t_range=[-0.2,1.]) #trim spectrogram to only be over hold

# # 	Sxx = np.array(Sxx, dtype='complex64') #change precision to reduce memory demand
# 	Sxx = np.array(abs(Sxx)) #take abs to get rid of imag component to reduce memory demand
# 	
# 	assert np.shape(Sxx) == (len(f),len(t))
# 	
# # 	print(Sxx.shape)
# # 	xxx
# 	
# 	return f,t,Sxx #Sxx = spectrogram for snippet

    
def ComputeSpectrogram_SFFT(snippet,fs,t_before,t_after,downsampleFlag):
	'''
	Compute the power spectral density (PSD, V**2/Hz) over frequency and time (aka spectrogram)
	for a given time-aligned snippet of LFP signal.
	
	Uses scipy.signal.spectrogram which uses short-time fast Fourier transform (sfft) to compute spectrogram.
	
	Input
	-------
	snippet: 1D array. LFP activity over which to compute PSD.
	fs: float. Sampling Frequency of LFP recording.
	t_before: float. How far snippet extends before time-alignment point. Used for time-alignment of the spectrogram.
	t_after: float. How far snippet extends after time-alignment point. Used for time-alignment of the spectrogram.
	downsampleFlag: bool. Has LFP signal been downsampled from its original fs? Used to fine tune fft settings.
	
	Output
	-------
	f: 1D array. Array of frequencies corresponding to computed PSD values
	t: 1D array. Array of times corresponding to computed PSD values
	Sxx: 2D array. Array of PSD values for each frequency and time point given by f and t.
	
	'''	
	if downsampleFlag:
		nperseg = 64
	else:
		nperseg = 256
		
	f, t, Sxx = spect(snippet, fs, nperseg = nperseg, noverlap = nperseg*7//8, nfft=nperseg*8)
	t = t-t_before
	
	f_crop = f<100
	f = f[f_crop]
	Sxx = Sxx[f_crop,:]
	
# 	print(t)
# 	print(f)
# 	print(Sxx.shape)
# 	print('dt: ' + str(np.mean(np.diff(t))))
# 	print('df: ' + str(np.mean(np.diff(f))))
# 	xxx  
# 	
	return f,t,Sxx #Sxx = spectrogram for snippet


def GetTestSig(test,fs,t_before,t_after):
	'''
	test : str. 'chirp', 'pulses', or 'sines'
		'chirp' is a signal that increases frequency linearly over time. Used to test processing fidelity over all freqs
		'pulses' has square wave pulses of decreasing durations. Used to test temporal resolution
		'sines' compounds sine waves of various frequencies. Used to test spectral resolution
	fs : float. Desired sampling frequency of the signal
	'''
	t = np.arange(0,t_before+t_after,1/fs)
	
	if test == 'chirp':
		#Test using a chirp
		c = 80 #chirp rate
		sig = np.sin(2*np.pi*(c/2*t**2))
			
	if test == 'pulses':
		#Test temporal res using a series of square wave pulses
		l = len(t)
		sq_freq = 10 #10 cycles/second = 10Hz
		duty = np.zeros_like(t)
		duty[l//10:3*l//20] = 0.5 #pulse width of 0.05sec
		duty[3*l//10:7*l//20] = 0.1 #pulse width of 0.01sec
		duty[5*l//10:11*l//20] = 0.05 #pulse width of 0.005sec
		duty[7*l//10:15*l//20] = 0.02 #pulse width of 0.002sec 
		duty[9*l//10:19*l//20] = 0.01 #pulse width of 0.001sec
		sig = signal.square(2*np.pi*sq_freq*t,duty)
			
	if test == 'sines':
		freqs = [8,20,25,28,30]
		start_inds = np.array([1,3,5,7,9]) * len(t) // 10
		sig = np.ones_like(t)
		for freq,start_ind in zip(freqs,start_inds):
			step = np.full_like(t,-1)
			step[start_ind:] = 1
			sig = sig + np.sin(2*np.pi*t*freq) * np.heaviside(step,np.ones_like(t))

	return sig


def ProcessLFP(tdt_files,samps_align_list,epoch,t_before,t_after,file_path,
					 saveFlag,mode,debugFlag,test=None,spect_mode='wavelet',downsampleFlag=True):
	'''
	Processes LFP data and saves out time-aligned LFP recordings, spectrograms, and PSD curves.
	
	Takes raw TDT files, takes time-aligned snippets of LFP recordings (as defined by samps_align arrays),
	applies line noise filtering, downsamples to 1000Hz (if desired), and computes spectrograms
	and PSD curves for each trial and each channel.
	
	Input
	-------
	tdt_files : list of strs. 
		List of TDT filenames for the session (in the order which they were recorded).
	samps_align_list : list of 1D arrays. 
		List of arrays containing the TDT indices corresponding to chosen time-alignment point onset.
		Length of this list must match length of tdt_files list.
	epoch : str.
	t_before : float. 
		Amount of time (in seconds) before each time-alignment sample to include.
	t_after : float. 
		Amount of time (in seconds) after each time-alignment sample to include.
	file_path : str. 
		Path to folder in which data file will be saved.
	saveFlag : bool.
		Choose whether to save out data or not.
	mode : str or None.
		Choose which data to output.
		'snips'= processed LFP recording snippets (voltage traces) for each trial and each channel
		'spects'= spectrograms (power vs freq vs time) for each trial and each channel
		'psds'= PSD (power vs freq) for each trial and each channel
		None= all of the above
	debugFlag : bool.
		Choose to print out data as script progresses in order to aid with debugging.
	test : str. 'chirp', 'pulses', 'sines', or False
		Used to bypass using actual data to run tests on signal processing pipeline by passing through predefined test signals.
		'chirp' is a signal that increases frequency linearly over time. Used to test processing fidelity over all freqs
		'pulses' has square wave pulses of decreasing durations. Used to test temporal resolution
		'sines' compounds sine waves of various frequencies. Used to test spectral resolution
		See scratch_chirp.py for more details. 
	spect_mode : 'wavelet' or 'sfft'
		Choose which function will be used to calculate spectrograms.
	downsampleFlag : bool.
		Choose whether raw LFP signal will be downsampled from fs=3051Hz to fs=1000Hz.
		This is to facilitate faster processing times.

	Output
	-------
	LFP_snippets : dict with the following keys:
		'name': str describing this processed data (epoch, session, datatype)
		'fs': sampling frequency of processed data
		'chs': list of all channels present in data
		'LFP': 3D array. Processed time-aligned LFP recordings for each ch and each trial.
		't': 1D array. Array of times corresponding to values in LFP
		'shape':tuple describing what the shape and the axes of 'LFP' correspond to
	Spectrograms : dict with the following keys:
		'name': str describing the data (epoch, session, datatype)
		'fs': sampling frequency of processed data used to compute Sxx
		'chs': list of all channels present in data
		'Sxx': 4D array. Time-aligned spectrograms for each ch and each trial. 
		'f': 1D array. Array of frequencies corresponding to values in Sxx
		't': 1D array. Array of times corresponding to values in Sxx
		'shape': tuple describing what the shape and the axes of 'Sxx' correspond to
	PSD : dict with the following keys:
		'name': str describing the data (epoch, session, datatype)
		'fs': sampling frequency of processed data used to compute PSD
		'chs': list of all channels present in data
		'PSD': 3D array. PSD vs f curve for each ch and each trial. 
		'f': 1D array. Array of frequencies corresponding to PSD values
		'shape': tuple describing what the shape and the axes of 'PSD' correspond to

	'''
		
	
	if test:
		test_name = 'TEST_'
		print('TEST: ' + test)
	else:
		test_name = ''
		
	print('downsample: ' + str(downsampleFlag))
	print('spectrogram mode: ' + spect_mode)
	print('epoch: ' + epoch)
	
	if mode:
		print('data mode: ' + mode)
		if mode == 'snips':
			doSnips = True
			doSpects = False
			doPSDs = False
		if mode == 'spects':
			doSnips = False
			doSpects = True
			doPSDs = False
		if mode == 'psds':
			doSnips = False
			doSpects = False
			doPSDs = True
	else: #if no mode selected, do all
		doSnips = True
		doSpects = True
		doPSDs = True
		print('data mode: all')
		
	num_files = len(tdt_files) #number of files within the current session

	# ensure there is a samps_align file for each tdt file
	assert len(samps_align_list) == num_files
	
	
	for i in range(num_files):
		
		# create unique str to identify this file
		filestr = tdt_files[i][-21:-8]
		print('\n' + filestr)
		
		# get samps_align for current file
		samps_align=np.rint(samps_align_list[i]).astype('int64') 
		
		
		if not test:
			# read in tdt file
			LFP1 = tdt.read_block(tdt_files[i], store='LFP1')
			LFP2 = tdt.read_block(tdt_files[i], store='LFP2')
		
			hasLFP2 = bool(LFP2.streams) #to see if there is an LFP2 stream
			fs_orig = LFP1.streams.LFP1.fs #orig fs (before any potential downsampling)
			if hasLFP2:
				assert fs_orig == LFP2.streams.LFP2.fs
				
			# get all channels for file
			if hasLFP2:			
				all_chs = LFP1.streams.LFP1.channel + list(np.array(LFP2.streams.LFP2.channel)+len(LFP1.streams.LFP1.channel))
			else:
				all_chs = LFP1.streams.LFP1.channel
				
			num_chs_LFP1 = len(LFP1.streams.LFP1.channel)	
			
		if test:
			all_chs = [0,1,2]
			fs_orig = 3051.7578125
			
		num_chs = len(all_chs)
		
			
		#Set up to downsample
		if downsampleFlag:
			q = int(np.floor(fs_orig/500)) #downsampling factor to get down to 500Hz
			fs_new = fs_orig/q #new fs after downsampling
			fs_out = fs_new
		if not downsampleFlag:
			fs_out = fs_orig
		

		if debugFlag: 
			print('tdt:LFP1 shape\n',np.shape(LFP1.streams.LFP1.data))
			print('hasLFP2:' + str(hasLFP2))
			if hasLFP2:	print('tdt:LFP2 shape\n',np.shape(LFP2.streams.LFP2.data))
			input(f'Current line of code: {GetLineNumber()} \nPress enter to continue')
		
		
		#set up to notch filter out 60,120,180 Hz line noise
		Q=30 #quality factor
		b60,a60 = iirnotch(60.,Q,fs_out) #this is done after downsampling would be done
		b120,a120 = iirnotch(120.,Q,fs_out)
		b180,a180 = iirnotch(180.,Q,fs_out)
		
		#convert t_before and t_after to sample units
		samps_before = int(np.rint(t_before*fs_orig)) #these use fs_orig since they interact with original signal (before downsampling)
		samps_after = int(np.rint(t_after*fs_orig))

		
		num_trials = len(samps_align)
		if downsampleFlag:
			num_samps_per_trial = int(np.rint((samps_before + samps_after)/q)) #divide by downsampling factor
		else:
			num_samps_per_trial = samps_before + samps_after
		
		#Initialize variables to store data
		if doSnips:
			snips_data = np.full((num_chs,num_trials,num_samps_per_trial),np.nan)
		spects = []
		psds = []
	
		#Loop thru channels
		for ch in range(num_chs):
			spects.append([]) #using lists instead of predefined arrays allows for flexibility in Sxx and psd calculation parameters
			psds.append([])
			chtime_start = time.time()
			
			#Loop thru trials
			for trial,samp_align in enumerate(samps_align):
			
				#Get window of samples for current trial
				samp_begin = samp_align - samps_before
				samp_end = samp_align + samps_after
				
				#Get time-aligned snippet from TDT stream
				if not test:
					if ch < num_chs_LFP1: #LFP1 stream
						sig = LFP1.streams.LFP1.data[ch,samp_begin:samp_end]
					else: #LFP2 stream
						sig = LFP2.streams.LFP2.data[ch-num_chs_LFP1,samp_begin:samp_end]
				
				#Get test signal
				if test:
					sig = GetTestSig(test,fs_orig,t_before,t_after)				
								
				#downsample
				if downsampleFlag:
					sig = signal.decimate(sig,q) #downsample down to 500Hz. This function applies anti-aliasing filters.
			
				# Filter line noise	
				sig = filtfilt(b60,a60,sig) #filter out 60hz noise
				sig = filtfilt(b60,a60,sig) #filter out 60hz noise, again
				sig = filtfilt(b120,a120,sig) #filter out 120hz noise
				sig = filtfilt(b180,a180,sig) #filter out 180hz noise
				
				if doSnips:
					#store snippet
					snips_data[ch,trial,:] = sig
			
				if doSpects:
					# Get spectrogram
# 					if spect_mode == 'wavelet':
# 						f,t,Sxx = ComputeSpectrogram_CWT(sig,fs_out,t_before,t_after)
					if spect_mode == 'sfft':
						f,t,Sxx = ComputeSpectrogram_SFFT(sig,fs_out,t_before,t_after,downsampleFlag)
					
					spects[ch].append(Sxx) 
			
# 				if doPSDs:
# 					# Get PSD
# 					f_psd,psd = ComputePSD(sig,fs_out) 
# 					psds[ch].append(psd)

			
			chtime = np.rint(time.time() - chtime_start)
			num_chs_todo = num_chs-ch-1
			time_left = np.round(chtime*num_chs_todo/60,2) #mins
			print(f'Channel {ch+1}/{num_chs} done. Approx time remaining: {time_left} mins.',end='\n')
# 				pbar.update(1)
		
		#put data from this file into an array to be saved out
		if i==0:
			if doSnips:
				snips_data_out = snips_data
			if spects:
				spects_out = np.array(spects)
			if doPSDs:
				psds_out = np.array(psds)
		
		#if more than one file, combine data along trials axis	
		else: 
			if doSnips:
				snips_data_out = np.append(snips_data_out,snips_data,axis=1) 
			if spects:
				spects_out = np.append(spects_out,np.array(spects),axis=1) 
			if doPSDs:
				psds_out = np.append(psds_out,np.array(psds),axis=1) 
			
	print('\n')
	
	#Organize data into dictionaries
	if doSnips:
		LFP_snippets = {'name':'LFP_'+ epoch + '_' + filestr, 'fs':fs_out, 
					 'chs':all_chs,'LFP':snips_data_out, 't':np.linspace(-t_before,t_after,len(sig)),
					 'shape':('num_chs','num_trials','num_samps_per_trial')}
	
	if doSpects:
		Spectrograms = {'name':'LFP_'+ epoch + '_Spectrograms' + filestr, 'fs':fs_out, 
					 'chs':all_chs,'Sxx':spects_out, 
					 'f':f, 't':t, 'shape':('num_chs','num_trials','len(f)','len(t)')}

	if doPSDs:
		PSD = {'name':'LFP_'+ epoch + '_' + filestr, 'fs':fs_out, 
					 'chs':all_chs,'PSD':psds_out, 
					 'f':f_psd, 'shape':('num_chs','num_trials','len(f)')}
	
	#Save out dictionaries
	if saveFlag:
		
		if doSnips:
			file_save_name = file_path + test_name + 'LFP_snippets_'+ epoch + '_' + filestr + '.pkl'
			with open(file_save_name,'wb') as f:
				pickle.dump(LFP_snippets,f)
			print('\n' + file_save_name + ' saved.')
			del LFP_snippets
			del snips_data_out
			del snips_data
	
		if doSpects:
			file_save_name = file_path + test_name + 'Spectrograms_'+ epoch + '_' + filestr + '.pkl'
			with open(file_save_name,'wb') as f:
				pickle.dump(Spectrograms,f)
			print(file_save_name + ' saved.')
			del Spectrograms
			del spects_out
			del spects
	
		if doPSDs:
			file_save_name = file_path + test_name + 'PSDs_'+ epoch + '_' + filestr + '.pkl'
			with open(file_save_name,'wb') as f:
				pickle.dump(PSD,f)
			print('\n' + file_save_name + ' saved.')
			del PSD
			del psds_out
			del psds


			
	return #LFP_snippets, Spectrograms, psds_out


