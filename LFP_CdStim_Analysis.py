# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:23:00 2024

@author: coleb

Analyze processed LFP data from the caudate stimulation experiment (Santacruz et al. 2017) 

Classes and methods which analyze time-aligned LFP snippets and spectrograms using
decision-making behavioral metrics.
"""

# import pandas as pd


	   
	   
#%% Import libs
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
# import matplotlib.colors as mpl_colors
import numpy as np
# from scipy import signal
# from scipy.stats import spearmanr
# from scipy.stats import mannwhitneyu
# from scipy.stats import ttest_ind
from scipy.stats import tukey_hsd
from scipy.stats import zscore
from scipy.stats import f_oneway
import statsmodels.api as sm
import statsmodels.formula.api as smf
# import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "svg"
import pandas as pd
# from matplotlib.gridspec import GridSpec
# from matplotlib_venn import venn2,venn3
from LFP_CdStim_Processing import GetFileStr
from SortedFiles import GetFileList
from GoodChans import GetGoodChans
import time
from pandas.core.common import flatten
from scipy.stats import ttest_ind
from mne_connectivity import spectral_connectivity_epochs
# import neptune
import inspect


# import scot

#%% General functions

def get_line_number(): #to aid w/ debugging
    return inspect.currentframe().f_back.f_lineno

def GetIndexes(behavior,subject,block,choice,context,trial_type,rewarded,stayshift,epoch,debug):
	'''
	- behavior: dict of vectors describing subject behavior and task variables/outcomes
			This data is saved out by LFP_CdStim_Processing.
	- subject: 'Mario' or 'Luigi'
	- block: 'BlA', 'BlA1', 'BlA2', 'BlB', 'BlAp', or 'AllBl'
			Choose which block of trials to include.
			BlA=Block A, BlA1=1st half of Block A, BlA2=2nd half of Block A,
			BlB=Block B, BlAp=Block A prime (aka Block A')
			See Santacruz et al. 2017 for description of blocks
	- choice: 'a=any', 'a=LV', 'a=MV', or 'a=HV'
			Choose to include trials in which subject chose a certain target.
			a=any means to include all choice types, 
			a=LV includes only trials in which the low-value target was chosen
			a=MV includes only trials in which the medium-value target was chosen
			a=HV includes only trials in which the high-value target was chosen
	- context: 'L-M', 'L-H', 'M-H', or 'All Contexts'
			Choose to include trials in which certain targets were presented (for Mario 3-Target Task)
			L-M = Low and Medium-value targs presented
			L-H = Low and High-value targs presented
			M-H = Medium and High-value targs presented
			All Contexts = all target context types		  
	- trial_type: 'Free Choices', 'Forced Choices', or 'Free and Forced Choices'
			Choose which trial types to include. 
			Free choices are when two options are presented. 
			Forced choices are when only one option is presented.
	- rewarded: 'Rewarded', 'Unrewarded', or 'Rewarded and Unrewarded'
			Choose to include trials which were rewarded and/or unrewarded.
	- stayshift: 'Win-stay', 'Win-shift', 'Lose-shift', 'Lose-stay', or 'stayshift not analyzed'
			Choose to include trials which fall into one of four win/lose stay/shift categories
	- epoch: 'hc', 'rp', or 'mp'
			Which part of a trial is being focused on. 
			hc=hold_center, rp=reward period, mp=movement period
	- debug: bool
			Choose to print out data as script progresses in order to aid with debugging.
	'''
	
	if debug:
		print("behavior vector")
		print(behavior.keys())
		for key in behavior.keys(): 
			if 'num_trials' not in key: print(behavior[key].shape, np.unique(behavior[key]))
		input(f'Current line of code: {get_line_number()} \nPress enter to continue')
	
	if subject == 'Luigi':
		num_trials_A = num_trials_B = 100
	elif subject == 'Mario':
		num_trials_A = 150
		num_trials_B = 100
		
	# get appropriate indexes for specified choices
	if choice=='a=any':
		if subject == 'Luigi':
			choice_ind = (behavior['Choice_L'] == 1) | (behavior['Choice_H'] == 1)
		elif subject == 'Mario':
			choice_ind = (behavior['Choice_L'] == 1) | (behavior['Choice_H'] == 1) | (behavior['Choice_M'] == 1)
	elif choice=='a=LV':
		choice_ind = (behavior['Choice_L'] == 1) 
	elif choice=='a=MV':
		choice_ind = (behavior['Choice_M'] == 1) 
	elif choice=='a=HV':
		choice_ind = (behavior['Choice_H'] == 1)
		
	#get indexes for specified trial type
	if trial_type == 'Free Choices':
		trial_type_ind = (behavior['TrialType'] == 1)
	if trial_type == 'Forced Choices':
		trial_type_ind = (behavior['TrialType'] == -1)
	if trial_type == 'Free and Forced Choices':
		trial_type_ind = np.full_like(behavior['TrialType'],True,dtype=bool)
	
	#get indexes for specified contexts
	if context == 'All Contexts':
		context_ind = np.full_like(behavior['TrialType'],True,dtype=bool)
	else:
		context_ind = behavior['Context'] == context
		
	#get indexes for rewarded/unrewarded trials
	if rewarded == 'Rewarded':
		reward_ind = (behavior['Rewarded'] == 1)
	elif rewarded == 'Unrewarded':
		reward_ind = (behavior['Rewarded'] == 0)
	elif rewarded == 'Rewarded and Unrewarded': 
		reward_ind = np.full_like(behavior['Rewarded'],True,dtype=bool)
	
	#get indexes for win-stay/lose-shift trials
	if stayshift == 'Win-stay':
		stayshift_ind = (behavior['WinStays'] == 1)
	elif stayshift == 'Win-shift':
		stayshift_ind = (behavior['WinStays'] == 0)
	elif stayshift == 'Lose-shift':
		stayshift_ind = (behavior['LoseShifts'] == 1)
	elif stayshift == 'Lose-stay':
		stayshift_ind = (behavior['LoseShifts'] == 0)
	elif stayshift == 'stayshift not analyzed':
		stayshift_ind = np.full_like(behavior['Rewarded'],True,dtype=bool)
		
	if debug:
		print("GetIndexes:\nlen(choice_ind),len(trial_type_ind),len(context_ind),len(reward_ind)")	 
		print(len(choice_ind),len(trial_type_ind),len(context_ind),len(reward_ind))
		input(f'Current line of code: {get_line_number()} \nPress enter to continue')
		
	ind = choice_ind & trial_type_ind & context_ind & reward_ind & stayshift_ind
	ind = np.nonzero(ind)[0] #convert vector of bools to actual indexes
	
	if stayshift and epoch=='rp':
		ind = ind - 1 #to look at the rewards that precede the winstay/loseshift decision, shift indices back one
	
	#get indexes for specified block
	if block == 'BlA':
		ind = ind[ind < num_trials_A]
	elif block == 'BlA1':
		ind = ind[ind < num_trials_A//2]
	elif block == 'BlA2':
		ind = ind[np.array(ind>num_trials_A//2) & np.array(ind<num_trials_A)]
	elif block == 'BlB':
		ind = ind[np.array(ind>num_trials_A) & np.array(ind<num_trials_A+num_trials_B)]
	elif block == 'BlAp':
		ind = ind[ind > num_trials_A+num_trials_B]

	return ind


def Plot_WinStayLoseShift(win_stay_probs_L, lose_shift_probs_L, win_stay_probs_M, lose_shift_probs_M):
	
	win_stay_probs_list = [win_stay_probs_L,win_stay_probs_M]
	lose_shift_probs_list = [lose_shift_probs_L,lose_shift_probs_M]
	names = ['Monkey L', 'Monkey M']
	x_list = [1,2]
	dx = 0.2
	hdx=dx/2	

	fig,ax=plt.subplots()
	
	for i,(x,win_stay_probs,lose_shift_probs) in enumerate(zip(x_list,win_stay_probs_list,lose_shift_probs_list)):

		avg_win_stay_prob = np.mean(win_stay_probs,axis=0)
		avg_lose_shift_prob = np.mean(lose_shift_probs,axis=0)
		
		std_win_stay_prob = np.std(win_stay_probs,axis=0)
		std_lose_shift_prob = np.std(lose_shift_probs,axis=0)
		
		ax.errorbar(x-hdx,avg_win_stay_prob,yerr=std_win_stay_prob,fmt='.',color='k')	
		ax.bar(x-hdx,avg_win_stay_prob,color='gray',hatch='',label='Win-stay',width=dx)
		
		ax.errorbar(x+hdx,avg_lose_shift_prob,yerr=std_lose_shift_prob,fmt='.',color='k')	
		ax.bar(x+hdx,avg_lose_shift_prob,color='gray',hatch='//',label='Lose-shift',width=dx)

		if i==0: ax.legend(fontsize=14)
		
# 	ax.set_title(subject)
	labelsize=14
	ax.set_xticks(x_list,names)	
	ax.set_ylabel('Probability',fontsize=labelsize)
	ax.set_ylim([0,1])
	ax.set_xlim([0.6,2.4])
	ax.spines[['right', 'top']].set_visible(False)
	ax.yaxis.set_tick_params(labelsize=labelsize)
	ax.xaxis.set_tick_params(labelsize=labelsize)


def PlotFreeChoiceProbs(free_choice_probs_L,free_choice_probs_M):

	probs_list = [free_choice_probs_L,free_choice_probs_M]
	names = ['Monkey L', 'Monkey M']
	x_list = [1,1.6]
	dx = 0.15
	hdx=dx/2	

	fig,ax=plt.subplots()
	
	for i,(x,probs) in enumerate(zip(x_list,probs_list)):

		
		avg_prob_l = np.nanmean(probs[:,0],axis=0)
		std_prob_l = np.nanstd(probs[:,0],axis=0) / np.sqrt(np.size(probs[:,0],axis=0))
		
		avg_prob_h = np.nanmean(probs[:,1],axis=0)
		std_prob_h = np.nanstd(probs[:,1],axis=0) / np.sqrt(np.size(probs[:,1],axis=0))
		
		if i==1: #for Mario only
			avg_prob_m = np.nanmean(probs[:,2],axis=0)
			std_prob_m = np.nanstd(probs[:,2],axis=0) / np.sqrt(np.size(probs[:,2],axis=0))
		
		
		
		
		if i==0:
			ax.errorbar(x-hdx,avg_prob_l,yerr=std_prob_l,fmt='.',color='k')	
			ax.bar(x-hdx,avg_prob_l,color='red',hatch='',width=dx)
			
			ax.errorbar(x+hdx,avg_prob_h,yerr=std_prob_h,fmt='.',color='k')	
			ax.bar(x+hdx,avg_prob_h,color='blue',hatch='',width=dx)
			
		else:
			ax.errorbar(x-2*hdx,avg_prob_l,yerr=std_prob_l,fmt='.',color='k')	
			ax.bar(x-2*hdx,avg_prob_l,color='red',hatch='',label='LV',width=dx)
			
			ax.errorbar(x,avg_prob_m,yerr=std_prob_m,fmt='.',color='k')	
			ax.bar(x,avg_prob_m,color='gold',hatch='',label='MV',width=dx)
			
			ax.errorbar(x+2*hdx,avg_prob_h,yerr=std_prob_h,fmt='.',color='k')	
			ax.bar(x+2*hdx,avg_prob_h,color='blue',hatch='',label='HV',width=dx)

		if i==1: ax.legend(fontsize=14,ncols=3,loc='upper center', bbox_to_anchor=(0.5, 1.2))
		
# 	ax.set_title(subject)
	labelsize=14
	ax.set_xticks(x_list,names)	
	ax.set_ylabel('Free Choice Probability',fontsize=labelsize)
	ax.set_ylim([0,1])
	ax.set_xlim([0.7,2.6])
	ax.spines[['right', 'top']].set_visible(False)
	ax.yaxis.set_tick_params(labelsize=labelsize)
	ax.xaxis.set_tick_params(labelsize=labelsize)
	
	plt.savefig(r"C:\Users\coleb\Desktop\Santacruz Lab\Value Stimulation\Reward Directionality Paper\Figures\Fig1\SessAvgBehavior.svg")
	
	print('Luigi:')
	print(f_oneway(free_choice_probs_L[:,0],free_choice_probs_L[:,1]))
	print(tukey_hsd(free_choice_probs_L[:,0],free_choice_probs_L[:,1]))
	print('Mario:')
	print(f_oneway(free_choice_probs_M[:,0],free_choice_probs_M[:,1],free_choice_probs_M[:,2]))
	print(tukey_hsd(free_choice_probs_M[:,0],free_choice_probs_M[:,1],free_choice_probs_M[:,2]))

#%% Spectral Power
class SpectralPowerAnalysis_TwoAreas():
	'''
	Class which loads and plots power spectral data saved out by LFP_CdStim_Processing.
	This class averages together data from the trials and sessions which meet specified conditions,
	allowing comparison across groups, conditions, blocks, etc.
	This class can plot spectrograms and frequency band power which is obtained from the spectrogram.
	
	Parameters
	----------
		- subject: 'Mario' or 'Luigi'
		- areas: list of 2 strings. E.g. ['Cd','ACC'] or ['M1','PMd']
				Choose which brain areas data to include
		- freq_band_names: list of strings. E.g. ['beta', 'theta', 'gamma']
				Names of the frequency bands which will be analyzed
		- freq_bands: list of tuples. E.g. [(3.5,8.5), (11.5,38.5), (60.5,100.5)]
				(lower limit, upper limit) for frequency ranges for each band named in freq_band_names
				len(freq_band_names) must = len(freq_bands)
		- stim_or_sham: 'All Sessions', 'Sham', or 'Stim'
				Choose which sessions to include.
		- epoch: 'hc', 'rp', or 'mp'
				Choose which part of a trial will be loaded. 
				hc=hold_center, rp=reward period, mp=movement period
		- block: 'BlA', 'BlA1', 'BlA2', 'BlB', 'BlAp', or 'AllBl'
				Choose which block of trials to include.
				BlA=Block A, BlA1=1st half of Block A, BlA2=2nd half of Block A,
				BlB=Block B, BlAp=Block A prime (aka Block A')
				See Santacruz et al. 2017 for description of blocks
		- choice: 'a=any', 'a=LV', 'a=MV', or 'a=HV'
				Choose to include trials in which subject chose a certain target.
				a=any means to include all choice types, 
				a=LV includes only trials in which the low-value target was chosen
				a=MV includes only trials in which the medium-value target was chosen
				a=HV includes only trials in which the high-value target was chosen
		- context: 'L-M', 'L-H', 'M-H', or 'All Contexts'
				Choose to include trials in which certain targets were presented (for Mario 3-Target Task)
				L-M = Low and Medium-value targs presented
				L-H = Low and High-value targs presented
				M-H = Medium and High-value targs presented
				All Contexts = all target context types		  
		- trial_type: 'Free Choices', 'Forced Choices', or 'Free and Forced Choices'
				Choose which trial types to include. 
				Free choices are when two options are presented. 
				Forced choices are when only one option is presented.
		- rewarded: 'Rewarded', 'Unrewarded', or 'Rewarded and Unrewarded Trials'
				Choose to include trials which were rewarded and/or unrewarded.
		- stayshift: 'Win-stay', 'Win-shift', 'Lose-shift', 'Lose-stay', or 'stayshift not analyzed'
				Choose to include trials which fall into one of four win/lose stay/shift categories
		- psd_or_zscr: 'psd' or 'zscr'
				Choose to normalize spectrograms using zscore normalization
				or whether to leave in native units of power spectral density (V**2/Hz)
		- doNeptune: bool
				Choose whether to track code run progress on online Neptune GUI (app.neptune.ai).
				Usually only doNeptune=True for long (>1hr) runs which I want to keep an eye on remotely.
		- debug: bool
				Choose to print out data as script progresses in order to aid with debugging.
		'''
	
	def __init__(self,subject,areas,freq_band_names,freq_bands,stim_or_sham,
			  epoch,block,choice,context,trial_type,rewarded,stayshift,psd_or_zscr, 
			  doNeptune,debug):
		
		self.subject = subject
		
		self.areas = areas
		self.linestyles = ['solid','dashed'] #distinct linestyle for each area
		
		self.freq_band_names = freq_band_names
		self.freq_bands = freq_bands
		self.colors = ['red','blue','gold','violet'] #colors for up to 4 frequency bands
		self.cmap = 'plasma' #colormap for spectrograms
		
		self.stim_or_sham = stim_or_sham
		self.epoch = epoch
		self.block = block
		self.choice = choice
		self.context = context
		self.trial_type = trial_type
		self.rewarded = rewarded
		self.stayshift = stayshift
		self.plot_subtitle = f'{epoch}, {stim_or_sham}, {block}, {choice}, {context}, {trial_type}, {rewarded}, {stayshift}'
		
		self.psd_or_zscr = psd_or_zscr
		
		if self.psd_or_zscr == 'zscr':
			self.pwr_label = 'z-score'
		if self.psd_or_zscr == 'psd':
			self.pwr_label = 'Power (V^2/Hz)'
		if self.epoch == 'hc':
			self.tlabel = 'Time since beginning of center hold (s)'
		if self.epoch == 'rp':
			self.tlabel = 'Time since end of target hold (s)'
		if self.epoch == 'mp':
			self.tlabel = 'Time since end of center hold (s)'
		self.flabel = 'Frequency (Hz)'
		
		self.doNeptune = doNeptune
		self.debug = debug
		
		#Find external hard drive, allowing flexibility of whether data is located on drive D: or E:
		if os.path.isdir("E:\\Value Stimulation\\Data\\" + self.subject + "\\LFP\\"):
			self.LFPfolderpath = "E:\\Value Stimulation\\Data\\" + self.subject + "\\LFP\\"
		elif os.path.isdir("D:\\Value Stimulation\\Data\\" + self.subject + "\\LFP\\"):
			self.LFPfolderpath = "D:\\Value Stimulation\\Data\\" + self.subject + "\\LFP\\"
		else:
			raise FileNotFoundError('External Hard Drive not found.')
		
# 		#PSD
# 		print('Loading PSDs for each session:')
# 		self.data_list = self.GetSessionsData('PSDs')
# 		self.psd, self.f_psd = self.GetPSD_AcrossSessions()
		
		self.file_list = self.GetSessionsFilenames('Spectrograms')
		
		self.behavior_list = self.GetSessionsData('Behavior')
		



	#%%Loading data methods 
	
	def GetSessionsFilenames(self,datatype):
		'''
		Get the filenames for the specified data for each session.
		List of sessions defined in SortedFiles.py and by self.stim_or_sham
		
		Parameters
		----------
		datatype : 'Behavior', 'PSDs', 'LFP_snippets', or 'Spectrograms'
			Choose which data type to load.
	
		Returns
		-------
		data_list : list. List of filenames for the specified data for each session
	
		'''
		
		
		if self.stim_or_sham == 'All Sessions':
			conds = ['Sham','Stim']
		else:
			conds = [self.stim_or_sham]
	
			
		file_list = [] #each element is the desired file name for a session
		for cond in conds: #loop through sham and/or stim groups
			#get the paths to and names of files for all sessions
			paths, filenames = GetFileList(self.subject,cond)
			num_sessions = len(filenames['hdf filenames'])
			
		
			for session in range(0,num_sessions): #loop thru sessions
			
				if datatype != 'Behavior':
					#file id made from each tdt filename
					tdt_files = [paths['tdt path'] + filename for filename in filenames['tdt filenames'][session]]
					file_id = tdt_files[0][-21:-8]
		
					#for current session
					file_load_name = f'{self.LFPfolderpath}{datatype}_{self.epoch}_{file_id}.pkl' 
	# 				file_load_name = f'{self.LFPfolderpath}{datatype}_{file_id}.pkl' 
	
				else: #for behavior files
					#file id made from each hdf filename using GetFileStr
					hdf_files = [paths['hdf path'] + filename for filename in filenames['hdf filenames'][session]]
					filestr = GetFileStr(hdf_files[0])
					file_path = self.LFPfolderpath.replace("LFP","Behavior") #go to behavior folder
					file_load_name = f'{file_path}BehaviorVectorsDict{filestr}.pkl'
					
				file_list.append(file_load_name)
				
		return file_list


	def GetSessionsData(self,datatype):
		'''
		Load the specified data for each session.
		List of sessions defined in SortedFiles.py and by self.stim_or_sham
		
		Parameters
		----------
		datatype : 'Behavior', 'PSDs', 'LFP_snippets', or 'Spectrograms'
			Choose which data type to load.
	
		Returns
		-------
		data_list : list in which each element is the data for one session	
		'''

		data_list = [] #each element is the loaded data for a session
		
		file_list = self.GetSessionsFilenames(datatype)
		
		for file in file_list:
			
			with open(file,'rb') as f:
				data = pickle.load(f)
			data_list.append(data)
			
		print(datatype + ' loaded.')

		return data_list
	
		
	def GetFreeChoiceProbs(self,num_trials,overwriteFlag):
		'''
		this method calculates the free choice probability of choosing each target over the last num_trials of Block A.
		
		Parameters
		-------------
		- num_trials: int. 
			the probability will be calculated over the last num_trials of Block A
		- overwriteFlag: bool. 
			If true, will not load previously saved results. Will get results anew and save them, overwriting previously saved results.
			If false, will load previously save results (if available).
			
		Output
		-------------
		- free_choice_probs: k x n array, where k=num_sessions and n=num_targs
				Each element of the array is the free choice probability of target n being
				chosen during the last num_trials of Block A of session k
		'''
		
		file_load_name = self.LFPfolderpath + 'FreeChoiceProbs_' + self.subject + '_' + self.plot_subtitle + '.pkl'
		print('-'*10 + '\nLoading Free Choice Probabilities')
		
		#if the saved data already exists, then just load it instead of re-running it (unless user wants to overwrite)
		if not overwriteFlag and os.path.isfile(file_load_name):
			with open(file_load_name,'rb') as f_:
				free_choice_probs = pickle.load(f_)
				print('\n' + file_load_name + ' loaded.\n')

		else: #if the data doesn't exist, then run anew from data		
	
			num_sessions = len(self.file_list)
			
			
			for k,file_name in enumerate(self.file_list): #loop thru sessions
			
				with open(file_name,'rb') as f_:
					session_data = pickle.load(f_)
				print(f'{file_name} loaded. ({k+1}/{num_sessions})')
				
				
				
				if k==0:
					if self.subject == 'Luigi':
						num_targs = 2
						num_trials_A = num_trials_B = 100
					elif self.subject == 'Mario':
						num_targs = 3
						num_trials_A = 150
						num_trials_B = 100
						
					free_choice_probs = np.full((num_sessions,num_targs),np.nan) # to store probs for each target for each session
				
				HV_freechoices = GetIndexes(self.behavior_list[k],self.subject,
									 'BlA','a=HV','All Contexts','Free Choices',
									 'Rewarded and Unrewarded','stayshift not analyzed',
									 self.epoch,self.debug)
				num_HV_freechoices = len(HV_freechoices[HV_freechoices > (num_trials_A-num_trials)])
				LV_freechoices = GetIndexes(self.behavior_list[k],self.subject,
									 'BlA','a=LV','All Contexts','Free Choices',
									 'Rewarded and Unrewarded','stayshift not analyzed',
									 self.epoch,self.debug)
				num_LV_freechoices = len(LV_freechoices[LV_freechoices > (num_trials_A-num_trials)])
				
				#choice contexts
				if self.subject == 'Mario':
					
					MV_freechoices = GetIndexes(self.behavior_list[k],self.subject,
										 'BlA','a=MV','All Contexts','Free Choices',
										 'Rewarded and Unrewarded','stayshift not analyzed',
										 self.epoch,self.debug)
					num_MV_freechoices = len(MV_freechoices[MV_freechoices > (num_trials_A-num_trials)])
				
					LH = GetIndexes(self.behavior_list[k],self.subject,'BlA','a=any','L-H',
						  'Free Choices','Rewarded and Unrewarded',
						  'stayshift not analyzed',self.epoch,self.debug)
					num_LH = len(LH[LH > (num_trials_A-num_trials)])
					MH = GetIndexes(self.behavior_list[k],self.subject,'BlA','a=any','M-H',
						  'Free Choices','Rewarded and Unrewarded',
						  'stayshift not analyzed',self.epoch,self.debug)
					num_MH = len(MH[MH > (num_trials_A-num_trials)])
					LM = GetIndexes(self.behavior_list[k],self.subject,'BlA','a=any','L-M',
						  'Free Choices','Rewarded and Unrewarded',
						  'stayshift not analyzed',self.epoch,self.debug)
					num_LM = len(LM[LM > (num_trials_A-num_trials)])
					
					#choice probability = num target choices / num of times target offered
					free_choice_probs[k,1] = num_HV_freechoices / (num_LH + num_MH) 
					free_choice_probs[k,2] = num_MV_freechoices / (num_LM + num_MH)
					free_choice_probs[k,0] = num_LV_freechoices / (num_LH + num_LM)
					
				elif self.subject == 'Luigi':
					freechoices = GetIndexes(self.behavior_list[k],self.subject,
										 'BlA','a=any','All Contexts','Free Choices',
										 'Rewarded and Unrewarded','stayshift not analyzed',
										 self.epoch,self.debug)
					num_freechoices = len(freechoices[freechoices > (num_trials_A-num_trials)])
					
					free_choice_probs[k,1] = num_HV_freechoices / (num_freechoices) #choice probability = num target choices / num of times target offered
					free_choice_probs[k,0] = num_LV_freechoices / (num_freechoices)
				
				
				del session_data	
			
			#save out data so we don't have to rerun everytime
			with open(file_load_name,'wb') as f_:
				pickle.dump(free_choice_probs,f_)
			print('\n' + file_load_name + ' saved.\n')
	
		self.free_choice_probs = free_choice_probs
		 
		return free_choice_probs

	

	#%% Processing data methods
	
	def GetSpect_SessionsAvgd(self,overwriteFlag):
		'''
		Get the session averaged spectrogram for each area
		
		Load spectrograms for each session, average across trials that meet conditions
		specified by object initialization, and then average these results across channels and sessions.
		
		- overwriteFlag: bool. 
			If true, will not load previously saved results. Will get results anew and save them, overwriting previously saved results.
			If false, will load previously save results (if available).
		
		Does not return anything, but results in new object attributes:
		- t: 1D array of time points of spectrograms
		- f: 1D array of frequency bins of spectrogram
		- spects_eachsess: 4D array of trial-averaged spectrograms for each session.
			shape = num_areas x num_sessions x len(f) x len(t)
		- spects_sessavg: 3D array containing session-averaged spectrograms
			shape = num_areas x len(f) x len(t)

		'''

		file_load_name = self.LFPfolderpath + 'Spects_' + self.subject + '_' + self.plot_subtitle + '.pkl'
		print('-'*10 + '\nLoading Spectrograms')
		
		#if the saved data already exists, then just load it instead of re-running it (unless user wants to overwrite)
		if (not overwriteFlag) and (os.path.isfile(file_load_name)):
			
			with open(file_load_name,'rb') as f_:
				(f,t,spects_sessavg,spect_eachsess) = pickle.load(f_)
				print('\n' + file_load_name + ' loaded.\n')

		else: #if the data doesn't exist, then average processed spectrograms over desired trials and good chs for each session
	
			num_sessions = len(self.file_list)
			
			
			for k,file_name in enumerate(self.file_list): #loop thru sessions
			
				try:
					
					with open(file_name,'rb') as f_:
						session_data = pickle.load(f_)
					print(f'{file_name} loaded. ({k+1}/{num_sessions})')
					
					if k==0:
						f = session_data['f']
						t = session_data['t']
						spect_eachsess = np.full((2,num_sessions,len(f),len(t)),np.nan) #2 is for number of areas
						spects_sessavg = np.full((2,len(f),len(t)),np.nan) #2 is for number of areas
				
					#get indexes for desired trials for this session
					inds = GetIndexes(self.behavior_list[k],self.subject,self.block,
							  self.choice,self.context,self.trial_type,self.rewarded,
							  self.stayshift,self.epoch,self.debug)
					
					for j,area in enumerate(self.areas):
						
						#print(session_data['name'])
						chs,num_good_chs,ch_locs = GetGoodChans(session_data['name'],area) #get good channels
						spect_area = []
						
						for i in range(num_good_chs): #loop thru good chs
						
							goodch = chs[i]
							
							if self.psd_or_zscr == 'psd':
								spect_ch = np.nanmean(abs(np.array(session_data['Sxx'][goodch])[inds]), axis=0) #avg across trials
							if self.psd_or_zscr == 'zscr':
								spect_ch = zscore(np.nanmean(abs(np.array(session_data['Sxx'][goodch])[inds]),axis=0),axis=1) #avg across trials, then zscore each freq bin
							
							assert np.size(spect_ch,axis=0) == np.size(session_data['f']) #to make sure i avgd over the right axis
							assert np.size(spect_ch,axis=1) == np.size(session_data['t'])
							assert (f == session_data['f']).all() #make sure all sessions have the same f and t vectors
							assert (t == session_data['t']).all() #make sure all sessions have the same f and t vectors
							
							spect_area.append(spect_ch)
		
						spect_eachsess[j,k,:,:] = np.nanmean(spect_area,axis=0) #avg across channels
					
					del session_data	
					
				except:
					print('SKIPPED')
					continue
				
			spects_sessavg = np.nanmean(spect_eachsess,axis=1) #avg across sessions
			
			#save out data so we don't have to rerun everytime
			with open(file_load_name,'wb') as f_:
				pickle.dump((f,t,spects_sessavg,spect_eachsess),f_)
			print('\n' + file_load_name + ' saved.\n')
			
		self.f = f
		self.t = t
		self.spects_sessavg, self.spects_eachsess = spects_sessavg, spect_eachsess
		 
		return 



	def GetSpect_SessionsAvgd_SpecificCh(self,ch):
		'''
		Get the session averaged spectrogram for a single channel.
		
		Load spectrograms for only one specified channel for each session, average across trials that meet conditions
		specified by object initialization, and then average these results across sessions.
		
		Does not return anything, but results in new object attributes:
		- t: 1D array of time points of spectrograms
		- f: 1D array of frequency bins of spectrogram
		- spects_eachsess: 3D array of trial-averaged spectrograms for each session.
			shape = num_sessions (that have ch as a goodch) x len(f) x len(t)
		- spects_sessavg: 2D array of session-averaged spectrogram
			shape = len(f) x len(t)

		'''
		
# 		file_load_name = self.LFPfolderpath + 'Spects_' + self.subject + '_' + self.plot_subtitle + '.pkl'
		print('-'*10 + '\nLoading Spectrograms')

#	     		#if the saved data already exists, then just load it instead of re-running it (unless user wants to overwrite)
# 		if not overwriteFlag and os.path.isfile(file_load_name):
# 			with open(file_load_name,'rb') as f_:
# 				(f,t,spects_sessavg_zscr,spect_eachsess_zscr,spects_sessavg,spect_eachsess) = pickle.load(f_)
# 				print('\n' + file_load_name + ' loaded.\n')

# 		else: #if the data doesn't exist, then get coherograms from lfp snippets		
# 	
		num_sessions = len(self.file_list)
		
		
		for k,file_name in enumerate(self.file_list): #loop thru sessions
		
			with open(file_name,'rb') as f_:
				session_data = pickle.load(f_)
			print(f'{file_name} loaded. ({k+1}/{num_sessions})')
			
			if k==0:
				f = session_data['f']
				t = session_data['t']
				spect_eachsess = []
				spects_sessavg = np.full((len(f),len(t)),np.nan)
		
			#get indexes for desired trials for this session
			ind = GetIndexes(self.behavior_list[k],self.subject,self.block,
						 self.choice,self.context,self.trial_type,self.rewarded,
						 self.stayshift,self.epoch,self.debug)
			#print(f'num trials: {len(ind)}')
			
			
			#print(session_data['name'])
			good_chs = np.array([])
			for area in self.areas:
				good_chs_area,_,_ = GetGoodChans(session_data['name'],area) #get good channels for each area
				good_chs = np.append(good_chs,good_chs_area)
				
			
			if isinstance(ch,np.int32): #if only 1 channel, don't need to avg over chs
				if ch not in good_chs: #ensure ch is a good ch for this session
					print(f'---\nch not a good ch for {file_name}\n---')
				else:
					
					if self.psd_or_zscr == 'psd':
						spect_ch = np.nanmean(abs(np.array(session_data['Sxx'][ch])[ind]), axis=0) #avg across trials
					if self.psd_or_zscr == 'zscr':
						spect_ch = np.nanmean(zscore(abs(np.array(session_data['Sxx'][ch])[ind]),axis=2), axis=0) #zscore each freq bin, then avg across trials
					
					spect_eachsess.append(spect_ch)
				
			elif len(ch)>1: #if more than 1 channel, then need to avg over chs
				sess_chs = np.array(list(set(good_chs) & set(ch))) # intersection of good chs and chs I want to look at
				if len(sess_chs)>0:
					
					if self.psd_or_zscr == 'psd':
						spect_ch = np.nanmean(abs(np.array(session_data['Sxx'])[:,ind,:,:]), axis=1) #avg across trials
					if self.psd_or_zscr == 'zscr':
						spect_ch_zscr = np.nanmean(zscore(abs(np.array(session_data['Sxx'][sess_chs])[:,ind,:,:]),axis=3), axis=1) #zscore each freq bin, then avg across trials

					spect_eachsess.append(np.nanmean(spect_ch,axis=0)) #avg across channels
				else:
					print(f'---\nchs not good chs for {file_name}\n---')
				
				
			del session_data	#to free up memory

		
		spect_eachsess = np.array(spect_eachsess)
# 		print(f'spect_eachsess {spect_eachsess.shape}')
		
		spects_sessavg = np.nanmean(spect_eachsess,axis=0) #avg across sessions
# 		print(f'spects_sessavg {spects_sessavg.shape}')
		
# 		#save out data so we don't have to rerun everytime
# 		with open(file_load_name,'wb') as f_:
# 			pickle.dump((f,t,spects_sessavg_zscr,spect_eachsess_zscr,spects_sessavg,spect_eachsess),f_)
# 		print('\n' + file_load_name + ' saved.\n')
			
		self.f = f
		self.t = t
		self.spects_sessavg, self.spects_eachsess = spects_sessavg, spect_eachsess
		
		self.plot_subtitle = self.plot_subtitle + f'\nCh {ch}'
		 
		return 				



	def GetSpect_RepSess(self,LFP_path,file_name,behavior_file,overwriteFlag):
		'''
		Get the trial averaged spectrogram for a single representative session.
		
		Load spectrograms for only one specified session, average across trials that meet conditions
		specified by object initialization, and then average these results across channels.
		
		Parameters
		-----------
		- LFP_path: str. 
			Path to folder where pkl file of spectrograms for chosen session resides.
			E.g. 'D:\\Value Stimulation\\Data\\Mario\\LFP\\'
		- file_name: str. 
			Pkl file of spectrograms for chosen session.
			E.g. 'Spectrograms_rp_Mario20161220.pkl'
		- behavior_file: str.
			Full path to behavior vectors pkl file for chosen session.
			E.g. 'D:\\Value Stimulation\\Data\\Mario\\Behavior\\BehaviorVectorsDictmari20161220_05_te2795.hdf.pkl'		
		- overwriteFlag: bool. 
			If true, will not load previously saved results. Will get results anew and save them, overwriting previously saved results.
			If false, will load previously save results (if available).
		
		Does not return anything, but results in new object attributes:
		- t: 1D array of time points of spectrogram
		- f: 1D array of frequency bins of spectrogram
		- spects_repsess: 3D array of trial-averaged spectrograms for the representative session.
			shape = num_areas x len(f) x len(t)

		'''
		
		file_load_name = f'{self.LFPfolderpath}Spects_{self.subject}_{self.plot_subtitle}_{file_name}.pkl'
		print('-'*10 + '\nLoading Spectrograms')
		
		#if the saved data already exists, then just load it instead of re-running it (unless user wants to overwrite)
		if not overwriteFlag and os.path.isfile(file_load_name):
			with open(file_load_name,'rb') as f_:
				(f,t,spects_repsess) = pickle.load(f_)
				print('\n' + file_load_name + ' loaded.\n')

		else: #if the data doesn't exist, then load and avg spects over chs and trials		
			
			
			with open(LFP_path + file_name,'rb') as f_:
				session_data = pickle.load(f_)
			print(f'{file_name} loaded.')
			
			with open(behavior_file,'rb') as f_:
				behavior = pickle.load(f_)
			print(f'{file_name} loaded.')
			
			f = session_data['f']
			t = session_data['t']
			spects_repsess = np.full((2,len(f),len(t)),np.nan) #2 is for number of areas
		
			#get indexes for desired trials for this session
			inds = GetIndexes(behavior,self.subject,self.block,
						 self.choice,self.context,self.trial_type,self.rewarded,
						 self.stayshift,self.epoch,self.debug)
			#print(f'num trials: {len(ind)}')
			
			for j,area in enumerate(self.areas):
				
				#print(session_data['name'])
				chs,num_good_chs,ch_locs = GetGoodChans(session_data['name'],area) #get good channels
				if 'TEST' in file_name:
					chs,num_good_chs = [0,1,2], 3
					
				spect_area = []
				
				for i in range(num_good_chs): #loop thru good chs
				
					goodch = chs[i]
					
					if self.psd_or_zscr == 'psd':
						spect_ch = np.nanmean(abs(np.array(session_data['Sxx'][goodch])[inds]), axis=0) #avg across trials
					if self.psd_or_zscr == 'zscr':
						spect_ch = zscore(np.nanmean(abs(np.array(session_data['Sxx'][goodch])[inds]),axis=0),axis=1) #avg across trials, then zscore each freq bin
											
					assert np.size(spect_ch,axis=0) == np.size(session_data['f']) #to make sure i avgd over the right axis
					assert np.size(spect_ch,axis=1) == np.size(session_data['t'])
					assert (f == session_data['f']).all() #make sure all sessions have the same f and t vectors
					assert (t == session_data['t']).all() #make sure all sessions have the same f and t vectors

					spect_area.append(spect_ch)

				spects_repsess[j,:,:] = np.nanmean(spect_area,axis=0) #avg across channels
				
			del session_data	
					
			if 'TEST' not in file_name:
				#save out data so we don't have to rerun everytime
				with open(file_load_name,'wb') as f_:
					pickle.dump((f,t,spects_repsess),f_)
				print('\n' + file_load_name + ' saved.\n')
			
		self.f = f
		self.t = t
		self.spects_repsess = spects_repsess
		self.plot_subtitle = self.plot_subtitle + '\n' + file_name
		 
		return 
	
	
	#%% Frequency band methods
	
	
	def GetBandPower(self,f1,f2):
		'''
		Takes Sxx (spectrogram) data and averages it over specified frequency band
		
		sig is the integral of Sxx from f1 to f2
		pwr is the integral of sig over time

		Parameters
		----------
		f1 : float or int
			lower bound of frequency bin.
		f2 : float or int
			upper bound of frequency bin..

		Returns
		-------
		sigs : 1D array
			time course data of average frequency band power.
		pwrs : float
			time averaged frequency band power

		'''
		
		spects = self.spects_eachsess
		#self.spect_eachsess is num_areas x num_sessions x len(f) x len(t)
		
		bound1 = self.f > f1
		bound2 = self.f < f2
		freq_band = bound1 & bound2

		sigs = np.nanmean(spects[:,:,freq_band,:],axis=2) #avg over freq band
		pwrs = np.nanmean(sigs,axis=2) #avg over time
		
		return sigs, pwrs
	
	
	def GetBandPower_RepSess(self,f1,f2):
		'''
		For representative session data
		Takes Sxx (spectrogram) data and averages it over specified frequency band
		
		sig is the integral of Sxx from f1 to f2
		pwr is the integral of sig over time

		Parameters
		----------
		f1 : float or int
			lower bound of frequency bin.
		f2 : float or int
			upper bound of frequency bin..

		Returns
		-------
		sigs : 1D array
			time course data of average frequency band power.
		pwrs : float
			time averaged frequency band power

		'''
		spects = self.spects_repsess
		#self.spect_repsess is num_areas x num_sessions x len(f) x len(t)
		
		bound1 = self.f > f1
		bound2 = self.f < f2
		freq_band = bound1 & bound2

		sigs = np.nanmean(spects[:,freq_band,:],axis=1) #avg over freq band
		pwrs = np.nanmean(sigs,axis=1) #avg over time
		
		return sigs, pwrs
	
		
	
	#%% Plotting methods

	def PlotSpects(self,t1,t2,f1,f2,v1,v2, doylabelFlag,doColorbarFlag,alldecoratorsFlag):
		'''
		Plots the session averaged spectrograms for both areas

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		f1 : float
			Set lower limit of freq axis for plot.
		f2 : float
			Set upper limit of freq axis for plot.
		v1 : float
			Set lower limit of power (colorbar axis) for plot.
		v2 : float
			Set upper limit of power (colorbar axis) for plot.
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		doColorbarFlag : bool
			Set whether plot will have colorbar.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.

		'''
		

			
		spects = self.spects_sessavg	
			
		for i,area in enumerate(self.areas): #loop thru areas
		
			fig,ax = plt.subplots()
# 			ax.set_aspect(0.01745)
			c = ax.pcolormesh(self.t, self.f, spects[i,:,:], vmin=v1,vmax=v2, cmap=self.cmap)
			
			ax.set_ylim([f1,f2])
			ax.set_xlim([t1,t2])
			ax.vlines(0,f1,f2,color='k',linewidth=3,linestyle='--') #to indicate t=0
			
			for j in range(len(self.freq_band_names)):
				ax.hlines(self.freq_bands[j],t1,t2,self.colors[j],linewidth=2) #to show freq bands
			
			if alldecoratorsFlag:
				ax.set_xlabel(self.tlabel)
				ax.set_ylabel(self.flabel)
				cbar = fig.colorbar(c, ax=ax)
				cbar.set_label(self.pwr_label,rotation = 270)
				cbar.ax.get_yaxis().labelpad = 15
				ax.set_title(area)
				fig.suptitle(self.plot_subtitle,fontsize=7)
			else:
				ax.set_xticklabels([])
				ax.set_title(area,fontsize=16)
				if not doylabelFlag:
					ax.set_yticklabels([])
				else:
					ax.yaxis.set_tick_params(labelsize=16)
					ax.set_ylabel(self.flabel,fontsize=16)
				if doColorbarFlag:
					cbar = fig.colorbar(c, ax=ax)
					cbar.set_label(self.pwr_label,rotation = 270,fontsize=16)
					cbar.ax.get_yaxis().labelpad = 15
					cbar.ax.yaxis.set_tick_params(labelsize=16)

# 			ax.set_aspect(1.6)
			fig.tight_layout() 




	def PlotSpects_EachSess(self,t1,t2,f1,f2,v1,v2, doylabelFlag,doColorbarFlag,alldecoratorsFlag):
		'''
		Plots the session averaged spectrograms for both areas

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		f1 : float
			Set lower limit of freq axis for plot.
		f2 : float
			Set upper limit of freq axis for plot.
		v1 : float
			Set lower limit of power (colorbar axis) for plot.
		v2 : float
			Set upper limit of power (colorbar axis) for plot.
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		doColorbarFlag : bool
			Set whether plot will have colorbar.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.

		'''
		
		spects = self.spects_eachsess
		
		num_spects = self.spects_eachsess.shape[1]
		
		for sess in range(num_spects):
			
			for i,area in enumerate(self.areas): #loop thru areas
			
				fig,ax = plt.subplots()
	# 			ax.set_aspect(0.01745)
				c = ax.pcolormesh(self.t, self.f, spects[i,sess,:,:], vmin=v1,vmax=v2, cmap=self.cmap)
				
				ax.set_ylim([f1,f2])
				ax.set_xlim([t1,t2])
				ax.vlines(0,f1,f2,color='k',linewidth=3,linestyle='--') #to indicate t=0
				
				for j in range(len(self.freq_band_names)):
					ax.hlines(self.freq_bands[j],t1,t2,self.colors[j],linewidth=2) #to show freq bands
				
				if alldecoratorsFlag:
					ax.set_xlabel(self.tlabel)
					ax.set_ylabel(self.flabel)
					cbar = fig.colorbar(c, ax=ax)
					cbar.set_label(self.pwr_label,rotation = 270)
					cbar.ax.get_yaxis().labelpad = 15
					ax.set_title(area)
					fig.suptitle(self.plot_subtitle + '\n' + self.file_list[sess],fontsize=7)
				else:
					ax.set_xticklabels([])
					ax.set_title(area,fontsize=16)
					if not doylabelFlag:
						ax.set_yticklabels([])
					else:
						ax.yaxis.set_tick_params(labelsize=16)
						ax.set_ylabel(self.flabel,fontsize=16)
					if doColorbarFlag:
						cbar = fig.colorbar(c, ax=ax)
						cbar.set_label(self.pwr_label,rotation = 270,fontsize=16)
						cbar.ax.get_yaxis().labelpad = 15
						cbar.ax.yaxis.set_tick_params(labelsize=16)
	
	# 			ax.set_aspect(1.6)
				fig.tight_layout() 



	def PlotSpect(self,t1,t2,f1,f2,v1,v2, doylabelFlag,doColorbarFlag,alldecoratorsFlag):
		'''
		For plotting the single channel session averaged spectrogram

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		f1 : float
			Set lower limit of freq axis for plot.
		f2 : float
			Set upper limit of freq axis for plot.
		v1 : float
			Set lower limit of power (colorbar axis) for plot.
		v2 : float
			Set upper limit of power (colorbar axis) for plot.
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		doColorbarFlag : bool
			Set whether plot will have colorbar.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.

		'''
			
		spects = self.spects_sessavg

		
		fig,ax = plt.subplots()
# 			ax.set_aspect(0.01745)
		c = ax.pcolormesh(self.t, self.f, spects, vmin=v1,vmax=v2, cmap=self.cmap)
		
		ax.set_ylim([f1,f2])
		ax.set_xlim([t1,t2])
		ax.vlines(0,f1,f2,color='k',linewidth=3,linestyle='--')
		
		for j in range(len(self.freq_band_names)):
			ax.hlines(self.freq_bands[j],t1,t2,self.colors[j],linewidth=2) #to show freq bands
		
		if alldecoratorsFlag:
			ax.set_xlabel(self.tlabel)
			ax.set_ylabel(self.flabel)
			cbar = fig.colorbar(c, ax=ax)
			cbar.set_label(self.pwr_label,rotation = 270)
			cbar.ax.get_yaxis().labelpad = 15
			fig.suptitle(self.plot_subtitle,fontsize=7)
		else:
			ax.set_xticklabels([])
			if not doylabelFlag:
				ax.set_yticklabels([])
			else:
				ax.yaxis.set_tick_params(labelsize=16)
				ax.set_ylabel(self.flabel,fontsize=16)
			if doColorbarFlag:
				cbar = fig.colorbar(c, ax=ax)
				cbar.set_label(self.pwr_label,rotation = 270,fontsize=16)
				cbar.ax.get_yaxis().labelpad = 15
				cbar.ax.yaxis.set_tick_params(labelsize=16)

# 			ax.set_aspect(1.6)
		fig.tight_layout() 


	def PlotSpects_Inset(self,t1,t2,f1,f2,v1,v2, doylabelFlag,doColorbarFlag,alldecoratorsFlag,savefigFlag):
		'''
		Plots the session averaged spectrograms for both areas
		
		Plots the 2nd specified area inset on the spectrogram of the 1st specified area.
		Good for when the spectrograms for both areas are similar.

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		f1 : float
			Set lower limit of freq axis for plot.
		f2 : float
			Set upper limit of freq axis for plot.
		v1 : float
			Set lower limit of power (colorbar axis) for plot.
		v2 : float
			Set upper limit of power (colorbar axis) for plot.
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		doColorbarFlag : bool
			Set whether plot will have colorbar.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.
		savefigFlag : bool
			Choose whether to save the figure to the Reward Directionality Paper folder

		'''			
			
		spects = self.spects_sessavg
		
		
		fig, ax1 = plt.subplots()
		
		left, bottom, width, height = [0.48, 0.55, 0.25, 0.25]
		ax2 = fig.add_axes([left, bottom, width, height])
		
		axs = [ax1,ax2] #ax1 is main figure (Cd), ax2 is inset (ACC)
		
		for i,(area,ax) in enumerate(zip(self.areas,axs)): #loop thru areas
		
			
			ax.set_aspect(0.015)
			c = ax.pcolormesh(self.t, self.f, spects[i,:,:], vmin=v1,vmax=v2, cmap=self.cmap, rasterized = True)
			
			ax.set_ylim([f1,f2])
			#to ensure no white space on right edge
			if t2 > self.t[-1]:
				ax.set_xlim(t1,self.t[-1])
			else:
				ax.set_xlim(t1,t2)
			
# 			for j in range(len(self.freq_band_names)):
# 				ax.hlines(self.freq_bands[j],t1,t2,self.colors[j],linewidth=2) #to show freq bands
		
		ax1.vlines(0,f1,f2,color='k',linewidth=2,linestyle='--')
		ax2.vlines(0,f1,f2,color='k',linewidth=1,linestyle='--')
		
		if alldecoratorsFlag:
			ax1.set_xlabel(self.tlabel)
			ax1.set_ylabel(self.flabel)
			cbar = fig.colorbar(c, ax=ax1)
			cbar.set_label(self.pwr_label,rotation = 270)
			cbar.ax.get_yaxis().labelpad = 12
			ax1.set_title(self.areas[0])
			fig.suptitle(self.plot_subtitle,fontsize=7)
		else:
			ax1.set_xticklabels([])
			ax2.set_xticklabels([])
			ax2.set_yticklabels([])
			ax1.set_title(self.plot_subtitle,fontsize=22)
			
			yticks = ax1.get_yticks()
			xticks = ax1.get_xticks()
			ax2.set_xticks(xticks[:-1])
			ax2.set_yticks(yticks[1:]) #get rid of [1:] after we make spects start from 0
# 			ax.set_title(area,fontsize=16)
			if not doylabelFlag:
				ax1.set_yticklabels([])
			else:
				ax1.yaxis.set_tick_params(labelsize=16)
				ax1.set_ylabel(self.flabel,fontsize=16)
			if doColorbarFlag:
				cbar = fig.colorbar(c, ax=ax1,fraction=0.0335, pad=0.04)
				cbar.set_label(self.pwr_label,rotation = 270,fontsize=16)
				cbar.ax.get_yaxis().labelpad = 15
				cbar.ax.yaxis.set_tick_params(labelsize=16)

		#ax.hlines([4,8,12,30],self.t_spect[0],self.t_spect[-1],color='k',linewidth=1) #freq bin lines
		
		
# 			fig.tight_layout() 		
		
		if savefigFlag: plt.savefig(fr"C:\Users\coleb\Desktop\Santacruz Lab\Value Stimulation\Reward Directionality Paper\Figures\Fig2\{self.subject}_{self.plot_subtitle}_spects.svg")
		
	def PlotSpects_RepSess(self,t1,t2,f1,f2,v1,v2, doylabelFlag,doColorbarFlag,alldecoratorsFlag):
		'''
		Plots the spectrograms of a representative session
		
		Plots the 2nd specified area inset on the spectrogram of the 1st specified area.
		Good for when the spectrograms for both areas are similar.

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		f1 : float
			Set lower limit of freq axis for plot.
		f2 : float
			Set upper limit of freq axis for plot.
		v1 : float
			Set lower limit of power (colorbar axis) for plot.
		v2 : float
			Set upper limit of power (colorbar axis) for plot.
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		doColorbarFlag : bool
			Set whether plot will have colorbar.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.

		'''	
			
		spects = self.spects_repsess

		for i,area in enumerate(self.areas): #loop thru areas
		
			fig,ax = plt.subplots()
			c = ax.pcolormesh(self.t, self.f, spects[i,:,:], vmin=v1,vmax=v2, cmap=self.cmap)
			
			ax.set_ylim([f1,f2])
			ax.set_xlim([t1,t2])
			ax.vlines(0,f1,f2,color='k',linewidth=3,linestyle='--')
			
			if alldecoratorsFlag:
				ax.set_xlabel(self.tlabel)
				ax.set_ylabel(self.flabel)
				cbar = fig.colorbar(c, ax=ax)
				cbar.set_label(self.pwr_label,rotation = 270)
				cbar.ax.get_yaxis().labelpad = 15
				ax.set_title(area)
				fig.suptitle(self.plot_subtitle,fontsize=7)
			else:
				ax.set_xticklabels([])
				ax.set_title(area,fontsize=16)
				if not doylabelFlag:
					ax.set_yticklabels([])
				else:
					ax.yaxis.set_tick_params(labelsize=16)
					ax.set_ylabel('Frequency (Hz)',fontsize=16)
				if doColorbarFlag:
					cbar = fig.colorbar(c, ax=ax)
					cbar.set_label(self.pwr_label,rotation = 270,fontsize=16)
					cbar.ax.get_yaxis().labelpad = 15
					cbar.ax.yaxis.set_tick_params(labelsize=16)

# 			ax.set_aspect(1.6)
			fig.tight_layout() 


	def PlotBandPower_RepSess(self):
		'''
		For representative session data
		Creates line plots of band power vs time
		Takes Sxx (spectrogram) data and averages it over specified frequency band

		'''
			
		for i,area in enumerate(self.areas):
			
			fig,ax=plt.subplots()
			
			for band_name,freq_band,color in zip(self.freq_band_names,self.freq_bands,self.colors):
				
				sigs,_ = self.GetBandPower_RepSess(freq_band[0],freq_band[1])
				ax.plot(self.t,sigs[i,:],color=color,label=band_name)
		
			ax.set_ylabel(self.pwr_label)
			ax.set_xlabel(self.tlabel)
			ax.set_title(area)
			ax.legend()
			
		fig.suptitle(self.plot_subtitle,fontsize=7)
		fig.tight_layout()	
		
		
		
		
	def PlotBandPowersTogether(self,t1,t2,y1,y2,doLegendFlag,doylabelFlag,alldecoratorsFlag,savefigFlag):
		'''
		Creates line plots of band power vs time
		Takes Sxx (spectrogram) data and averages it over specified frequency band

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		y1 : float
			Set lower limit of power axis for plot.
		y2 : float
			Set upper limit of power axis for plot.
		doLegendFlag : bool
			Set whether legend will be on or not
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.		
		savefigFlag : bool
			Choose whether to save the figure to the Reward Directionality Paper folder
		'''
			
		fig,ax = plt.subplots()
# 		ax.set_aspect(0.6)
		for (f1,f2),band_name,color in zip(self.freq_bands,self.freq_band_names,self.colors):
			 
			for i,(area,linestyle) in enumerate(zip(self.areas,self.linestyles)):
				
				sigs,_ = self.GetBandPower(f1,f2)
				avg=np.nanmean(sigs[i,:,:],axis=0) #avg across sessions
				sem=np.nanstd(sigs[i,:,:],axis=0) / np.sqrt(np.size(sigs[i,:,:],axis=0)) #sem across sessions
				ax.plot(self.t,avg,linestyle=linestyle,color=color,label=f'{area} {band_name}')
				ax.fill_between(self.t, avg+sem,avg-sem, color=color,alpha=0.2)
				
		if doLegendFlag: ax.legend(fontsize=16)
		
		ax.set_xlabel(self.tlabel,fontsize=16)
		ax.xaxis.set_tick_params(labelsize=16)
		
		if alldecoratorsFlag:
			ax.set_title(f'{self.areas[0]} vs {self.areas[1]} for all freq bands')
			ax.set_ylabel(self.pwr_label)
			fig.suptitle(self.plot_subtitle,fontsize=7)
			ax.legend()
			ax.set_xlabel(self.tlabel,fontsize=10)
			ax.xaxis.set_tick_params(labelsize=10)
		else:
			if not doylabelFlag:
				ax.set_yticklabels([])
			else:
				ax.yaxis.set_tick_params(labelsize=16)
				ax.set_ylabel(self.pwr_label,fontsize=16)
		
		ax.vlines(0,y1,y2,color='k',linewidth=2,linestyle='--')
		#ax.set_xticks(np.arange(0,1,0.1))
		ax.set_xlim([t1,t2])
		ax.set_ylim([y1,y2])
	
# 		ax.set_aspect(0.5)
# 		fig.tight_layout()	
		
		if savefigFlag: plt.savefig(fr"C:\Users\coleb\Desktop\Santacruz Lab\Value Stimulation\Reward Directionality Paper\Figures\Fig2\{self.subject}_{self.plot_subtitle}_bandpower.svg")
		
		
	
# 	def TemporalAnalysis(self,f1,f2,zscore_or_psd):
# 	
# 		#get power traces
# 		if stdzd:
# 			psths = self.GetUnitData(session,ch,sc,block,'stdzd_hc_psth')
# 		else:
# 			psths = self.GetUnitData(session,ch,sc,block,'hc_psth')
# 		num_timebins = psths.shape[1]		
# 		
# 		assert behavior_regressor != 'const', "Cannot do temporal analysis for a constant"
# 		
# 		#get entire behavior regressor matrix
# 		behavior_labels = self.GetSessionData(session,block,'hc_reg_labels')
# 		#print(behavior_labels)
# 		
# 		for i in range(len(behavior_labels)):
# 			if i==0:
# 				behavior_regressor_matrix = self.GetSessionData(session,block,behavior_labels[i])
# 			else:
# 				behavior_regressor_matrix = np.vstack((behavior_regressor_matrix,self.GetSessionData(session,block,behavior_labels[i])))
# 		behavior_regressor_matrix = np.transpose(behavior_regressor_matrix)
# 		
# # 		behavior_regressor_matrix = np.transpose(np.vstack((self.GetSessionData(session,block,'Choice_L'),
# # 													  self.GetSessionData(session,block,'Side'),
# # 													  self.GetSessionData(session,block,'Q_low'),
# # 													  self.GetSessionData(session,block,'Q_high'),
# # 													  self.GetSessionData(session,block,'time'))))

# 		num_holds = behavior_regressor_matrix.shape[0]
# 		
# 		#find specified behavior_regressor
# 		for idx,behav in enumerate(behavior_labels):
# 			if behav == behavior_regressor:
# 				behavior = behavior_regressor_matrix[:,idx]
# 				behav_idx=idx
# 				
# 		#preallocate for beta coeffs and alpha significance levels
# 		beta_coeffs = np.zeros(num_timebins)
# 		alpha_sig = np.zeros(num_timebins)
# 		
# 		#regress each psth time bin against specified behavior
# 		for k in range(num_timebins):
# 			timebin_data = psths[:,k] #get timebin k for all holds
# 			
# 			#Perform regression	using all behavior regressors (not just specified one)
# 			model = sm.OLS(timebin_data, sm.add_constant(behavior_regressor_matrix,has_constant='add'), missing='raise',hasconst=True)
# 			res = model.fit()
# 			
# 			#get beta coeff and alpha significance level of specified behavior
# 			beta_coeffs[k] = res.params[behav_idx]
# 			alpha_sig[k] = res.pvalues[behav_idx]	
# 		
# 		
# 		#Plot 
# 		fig = plt.figure()

# 		gs = GridSpec(3, 1, figure=fig)
# 		ax1 = fig.add_subplot(gs[0:2])
# 		ax2 = fig.add_subplot(gs[2])
# 		
# 		#Top plot: FR (psths) vs time for each hold, sorted by specified behavior
# 		ax=ax1

# 		#Bin holds along a continuous spectrum 
# 		if 'Q' in behavior_regressor: 
# 			bin_width=0.05
# 		elif 't' in behavior_regressor:
# 			bin_width=20
# 			behavior = np.exp(behavior)
# 		else:
# 			bin_width = 0.5
# 		bin_centers=np.arange(np.min(behavior), np.max(behavior)+bin_width, bin_width)
# 		colors = plt.cm.jet(np.linspace(0,1,len(bin_centers)))
# 		
# 		for i,bin_center in enumerate(bin_centers):
# 			
# 			#create bins
# 			low_edge = bin_center - bin_width
# 			high_edge = bin_center + bin_width
# 			
# 			holds_in_bin=[]
# 			for hold in range(num_holds):
# 				
# 				#bin holds according to value of each hold
# 				if (behavior[hold] >= low_edge) and (behavior[hold] < high_edge):
# 					holds_in_bin.append(psths[hold])
# 			
# 			if len(holds_in_bin)>0:
# 				avg = np.mean(np.array(holds_in_bin),axis=0)
# 				sem = np.std(np.array(holds_in_bin),axis=0) / np.sqrt(len(holds_in_bin))
# 				x=np.arange(0,len(avg))
# 				
# 				#plot each bin with a separate color
# 				ax.plot(avg, color=colors[i])
# 				ax.fill_between(x, avg+sem,avg-sem, color=colors[i],alpha=0.1)
# 		
# # 		axcb = fig.colorbar(ax)
# # 		axcb.set_label('Value')
# 		ax.set_ylabel('FR (Hz)')
# 		norm=mpl_colors.Normalize(vmin=np.min(bin_centers),vmax=np.max(bin_centers))
# 		fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet), ax=ax, label=behavior_regressor, location='top')
# 		
# 		
# 		#Bottom plot: beta coeff vs time, colored by significance level
# 		ax=ax2
# 		
# 		for k in range(num_timebins):
# 			
# 			if alpha_sig[k]<=0.05:
# 				color='red' #significant
# 			elif alpha_sig[k]<0.1:
# 				color='blue' #trending
# 			else:
# 				color='grey' #nonsignificant
# 				
# 			ax.plot(k,beta_coeffs[k],'o',color=color)
# 		
# 		ax.hlines(0,0,num_timebins-1,'k')
# 		ax.set_ylabel('Beta Coeff')
# 		ax.set_xlabel('Time relative to hold prompt (s)')
# 		
# 		ax2.set_xticks([0,9,19],['-1','0','1'])
# 		ax1.set_xticks([0,9,19],['','',''])
# 		#fig.suptitle(f'Temporal Analysis of a single unit\n{session}, Ch:{ch}, sc:{sc}')
# 		fig.suptitle(f'{session}, Ch:{ch}, sc:{sc}')
# 		fig.tight_layout()
		
		
#%% Connectivity

class ConnectivityAnalysis_TwoAreas():
	'''
	Class which computes and plots spectral connectivity between specified brain areas 
	using LFP snippets data saved out by LFP_CdStim_Processing.
	Connectivity is computed for every possible channel combination across trials.
	Data for each channel combo are then averaged to create a grand average connectivity 
	between the two brain areas.
	This class can plot time-frequency representations of connectivity (like a coherogram),
	and frequency band connectivity.
	
	Parameters
	----------
		- conn_method: str. E.g. 'coh', 'wpli', 'cacoh'
			Connectivity measure to compute.
			See https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html
			for full list of options
		- subject: 'Mario' or 'Luigi'
		- areas: list of 2 strings. E.g. ['Cd','ACC'] or ['M1','PMd']
				Choose which brain areas to compute connectivity between
		- freq_band_names: list of strings. E.g. ['beta', 'theta', 'gamma']
				Names of the frequency bands which will be analyzed
		- freq_bands: list of tuples. E.g. [(3.5,8.5), (11.5,38.5), (60.5,100.5)]
				(lower limit, upper limit) for frequency ranges for each band named in freq_band_names
				len(freq_band_names) must = len(freq_bands)
		- stim_or_sham: 'All Sessions', 'Sham', or 'Stim'
				Choose which sessions to include.
		- epoch: 'hc', 'rp', or 'mp'
				Choose which part of a trial will be loaded. 
				hc=hold_center, rp=reward period, mp=movement period
		- block: 'BlA', 'BlA1', 'BlA2', 'BlB', 'BlAp', or 'AllBl'
				Choose which block of trials to include.
				BlA=Block A, BlA1=1st half of Block A, BlA2=2nd half of Block A,
				BlB=Block B, BlAp=Block A prime (aka Block A')
				See Santacruz et al. 2017 for description of blocks
		- choice: 'a=any', 'a=LV', 'a=MV', or 'a=HV'
				Choose to include trials in which subject chose a certain target.
				a=any means to include all choice types, 
				a=LV includes only trials in which the low-value target was chosen
				a=MV includes only trials in which the medium-value target was chosen
				a=HV includes only trials in which the high-value target was chosen
		- context: 'L-M', 'L-H', 'M-H', or 'All Contexts'
				Choose to include trials in which certain targets were presented (for Mario 3-Target Task)
				L-M = Low and Medium-value targs presented
				L-H = Low and High-value targs presented
				M-H = Medium and High-value targs presented
				All Contexts = all target context types		  
		- trial_type: 'Free Choices', 'Forced Choices', or 'Free and Forced Choices'
				Choose which trial types to include. 
				Free choices are when two options are presented. 
				Forced choices are when only one option is presented.
		- rewarded: 'Rewarded', 'Unrewarded', or 'Rewarded and Unrewarded Trials'
				Choose to include trials which were rewarded and/or unrewarded.
		- stayshift: 'Win-stay', 'Win-shift', 'Lose-shift', 'Lose-stay', or 'stayshift not analyzed'
				Choose to include trials which fall into one of four win/lose stay/shift categories
		- doNeptune: bool
				Choose whether to track code run progress on online Neptune GUI (app.neptune.ai).
				Usually only doNeptune=True for long (>1hr) runs which I want to keep an eye on remotely.
		'''
	def __init__(self,subject,areas,freq_band_names,freq_bands,stim_or_sham,
			  epoch,block,choice,context,trial_type,rewarded,stayshift, 
			  doNeptune,debug):
		
		self.subject = subject
		
		self.areas = areas
		self.directions = [f'{self.areas[0]}→{self.areas[1]}',f'{self.areas[1]}→{self.areas[0]}']
		self.linestyles = ['solid','dashed'] #distinct linestyle for each direction
		
		self.freq_band_names = freq_band_names
		self.freq_bands = freq_bands
		self.colors = ['red','blue','gold','violet'] #colors for up to 4 frequency bands
		self.cmap = 'viridis' #colormap for coherograms
		self.gccmap = 'cividis' #colormap for GC time-frequency plots
		
		self.stim_or_sham = stim_or_sham
		self.epoch = epoch
		self.block = block
		self.choice = choice
		self.context = context
		self.trial_type = trial_type
		self.rewarded = rewarded
		self.stayshift = stayshift
		self.plot_subtitle = f'{epoch}, {stim_or_sham}, {block}, {choice}, {context}, {trial_type}, {rewarded}, {stayshift}'
		
		if self.epoch == 'hc':
			self.tlabel = 'Time since beginning of center hold (s)'
		if self.epoch == 'rp':
			self.tlabel = 'Time since end of target hold (s)'
		if self.epoch == 'mp':
			self.tlabel = 'Time since end of center hold (s)'
		self.flabel = 'Frequency (Hz)'
		self.gclabel = 'Granger Causality'
		
		
		self.doNeptune = doNeptune
		self.debug = debug
		
		#Find external hard drive, allowing flexibility of whether data is located on drive D: or E:
		if os.path.isdir("E:\\Value Stimulation\\Data\\" + self.subject + "\\LFP\\"):
			self.LFPfolderpath = "E:\\Value Stimulation\\Data\\" + self.subject + "\\LFP\\"
		elif os.path.isdir("D:\\Value Stimulation\\Data\\" + self.subject + "\\LFP\\"):
			self.LFPfolderpath = "D:\\Value Stimulation\\Data\\" + self.subject + "\\LFP\\"
		else:
			raise FileNotFoundError('External Hard Drive not found.')
		
		self.file_list = self.GetSessionsFilenames('LFP_snippets')	
		
		self.behavior_list = self.GetSessionsData('Behavior')



	#%%Loading data methods 
	
	def GetSessionsFilenames(self,datatype):
		'''
		Get the filenames for the specified data for each session.
		List of sessions defined in SortedFiles.py and by self.stim_or_sham
		
		Parameters
		----------
		datatype : 'Behavior', 'PSDs', 'LFP_snippets', or 'Spectrograms'
			Choose which data type to load.
	
		Returns
		-------
		data_list : list. List of filenames for the specified data for each session
	
		'''
		
		
		if self.stim_or_sham == 'All Sessions':
			conds = ['Sham','Stim']
		else:
			conds = [self.stim_or_sham]
	
			
		file_list = [] #each element is the desired file name for a session
		for cond in conds: #loop through sham and/or stim groups
			#get the paths to and names of files for all sessions
			paths, filenames = GetFileList(self.subject,cond)
			num_sessions = len(filenames['hdf filenames'])
			
		
			for session in range(0,num_sessions): #loop thru sessions
			
				if datatype != 'Behavior':
					#file id made from each tdt filename
					tdt_files = [paths['tdt path'] + filename for filename in filenames['tdt filenames'][session]]
					file_id = tdt_files[0][-21:-8]
		
					#for current session
					file_load_name = f'{self.LFPfolderpath}{datatype}_{self.epoch}_{file_id}.pkl' 
	# 				file_load_name = f'{self.LFPfolderpath}{datatype}_{file_id}.pkl' 
	
				else: #for behavior files
					#file id made from each hdf filename using GetFileStr
					hdf_files = [paths['hdf path'] + filename for filename in filenames['hdf filenames'][session]]
					filestr = GetFileStr(hdf_files[0])
					file_path = self.LFPfolderpath.replace("LFP","Behavior") #go to behavior folder
					file_load_name = f'{file_path}BehaviorVectorsDict{filestr}.pkl'
					
				file_list.append(file_load_name)
				
		return file_list


	def GetSessionsData(self,datatype):
		'''
		Load the specified data for each session.
		List of sessions defined in SortedFiles.py and by self.stim_or_sham
		
		Parameters
		----------
		datatype : 'Behavior', 'PSDs', 'LFP_snippets', or 'Spectrograms'
			Choose which data type to load.
	
		Returns
		-------
		data_list : list in which each element is the data for one session	
		'''

		data_list = [] #each element is the loaded data for a session
		
		file_list = self.GetSessionsFilenames(datatype)
		
		for file in file_list:
			
			with open(file,'rb') as f:
				data = pickle.load(f)
			
			data_list.append(data)
				
		print(datatype + ' loaded.')
			
		return data_list

	
	#%% Processing data methods	
	
									 
	def GetConn_SessionsAvgd(self,conn_method,overwriteFlag):
		'''
		Get the session averaged time-freq representation of connectivity for each area
		
		Load time-aligned LFP snippets for each session and compute connectivity 
		between brain areas according to the method specified by conn_method.
		This connectivity metric is computed for each channel combination before
		the grand average is computed. I.e. connectivity is computed for num_chs_area1 x num_chs_area2
		combinations before being averaged to yield a single grand average of connectivity
		btwn the two brain areas. These grand averages are then averaged across sessions.

		Parameters
		----------
		- conn_method : str. E.g. 'coh', 'wpli', 'cacoh'
			Connectivity measure to compute.
			See https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html
			for full list of options
		- overwriteFlag: bool. 
			If true, will not load previously saved results. Will get results anew and save them, overwriting previously saved results.
			If false, will load previously save results (if available).		


		Does not return anything, but results in new object attributes:
		- t: 1D array of time points of time-freq representations of connectivity
		- f: 1D array of frequency bins of time-freq representations of connectivity
		- conn_eachsess: 3D array of trial-averaged time-freq representations of connectivity for each session.
			shape = num_sessions x len(f) x len(t)
		- conn_sessavg: 2D array of session-averaged time-freq representations of connectivity
			shape = len(f) x len(t)

		'''
		self.connlabel = f'Connectivity ({conn_method})'
		
		
		file_load_name = self.LFPfolderpath + conn_method + '_' + self.subject + '_' + self.plot_subtitle + '.pkl'
		print('-'*10 + '\nGetting Coherograms')
		
		#if the saved data already exists, then just load it instead of re-running it (unless user wants to overwrite)
		if not overwriteFlag and os.path.isfile(file_load_name):
			
			with open(file_load_name,'rb') as f_:
				(f,t,conn_eachsess,conn_sessavg) = pickle.load(f_)
				print('\n' + file_load_name + ' loaded.\n')

		else: #if the data doesn't exist, then get coherograms from lfp snippets
			num_sessions = len(self.file_list)
			conn_eachsess = []
			for k,file_name in enumerate(self.file_list): #loop thru sessions
			
				print('-'*10)
				with open(file_name,'rb') as f_:
					session_data = pickle.load(f_)
				print(file_name + ' loaded.')
	
				#get indexes for desired trials for this session
				ind = GetIndexes(self.behavior_list[k],self.subject,self.block,
							 self.choice,self.context,self.trial_type,self.rewarded,
							 self.stayshift,self.epoch,self.debug)
	
				#get connection combinations
				area1_chs,num_area1_chs,_ = GetGoodChans(session_data['name'],self.areas[0]) #get good channels
				area2_chs,num_area2_chs,_ = GetGoodChans(session_data['name'],self.areas[1])
				num_connections = num_area1_chs * num_area2_chs
				
				if num_connections > 0:
					sources = []
					targets = []
	
					for area1_ch in area1_chs:
						for area2_ch in area2_chs:
							
							sources.append(area1_ch)
							targets.append(area2_ch)
# 							print(area1_ch,'&',area2_ch)
					
					fmin=4
					fmax=60
					cwt_freqs = np.arange(fmin,fmax,1.) 
					
					start_time = time.time()
					connectivity = spectral_connectivity_epochs(
						data = np.transpose(session_data['LFP'][:,ind,:],axes=(1,0,2)), #switch axes of data in order to fit the way MNE wants it.
						# names = session_data['chs']
						method = conn_method,
						indices = (np.array(sources),np.array(targets)),
						sfreq = session_data['fs'],
						mode = 'cwt_morlet',
						fmin = fmin, 
						fmax = fmax,
						cwt_freqs = cwt_freqs,
						cwt_n_cycles = cwt_freqs / 4,
						verbose = False,
						)
					
					t = np.array(connectivity.times) + np.min(session_data['t']) # the +min(t) is to align new t vector with original t vector
					f = np.array(connectivity.freqs)
					conn_eachsess.append(np.nanmean(connectivity.get_data(),axis=0)) #avg across all ch combos
					time_taken = np.round((time.time() - start_time)/60,2)
					print(f'{session_data["name"]} connectivity computed in {time_taken} mins. ({k+1}/{num_sessions})')
					
			conn_sessavg = np.nanmean(conn_eachsess,axis=0) #avg across sessions	
			
			#save out data so we don't have to rerun everytime
			with open(file_load_name,'wb') as f_:
				pickle.dump((f,t,conn_eachsess,conn_sessavg),f_)
			print('\n' + file_load_name + ' saved.\n')
		
		self.f = f
		self.t = t
		self.conn_eachsess, self.conn_sessavg = np.array(conn_eachsess), conn_sessavg 
			
		return
	
	
	
	
	def GetGC_SessionsAvgd(self,overwriteFlag):
		'''
		Get the session averaged time-freq representation of Granger Causality 
		between the two areas in both directions
		
		Same as GetConn_SessionsAvgd, but accounts for bidrectional nature of 
		Granger Causality (GC) result.
		WARNING: Doing all sessions in one go can take forever to run if channel count is high (>20)
		
		Load time-aligned LFP snippets for each session and compute GC 
		between brain areas. GC is computed for each channel combination before
		the grand average is computed. I.e. GC is computed for num_chs_area1 x num_chs_area2
		combinations before being averaged to yield a single grand average of GC
		btwn the two brain areas. These grand averages are then averaged across sessions.

		Parameters
		----------
		- overwriteFlag: bool. 
			If true, will not load previously saved results. Will get results anew and save them, overwriting previously saved results.
			If false, will load previously save results (if available).


		Does not return anything, but results in new object attributes:
		- t: 1D array of time points of time-freq representations of connectivity
		- f: 1D array of frequency bins of time-freq representations of connectivity
		- gc_eachsess: 4D array of trial-averaged time-freq representations of GC for each session.
			shape = num_sessions x num_directions (area1->area2 and area2->area1) x len(f) x len(t)
		- gc_sessavg: 3D array of session-averaged time-freq representations of connectivity
			shape = num_directions (area1->area2 and area2->area1) x len(f) x len(t)
		
		'''
		print('-'*10 + '\nLoading Granger Causality')
		
		file_load_name = self.LFPfolderpath + 'GrangerCausality_' + self.subject + '_' + self.plot_subtitle + '.pkl'
				
		#if the saved data already exists, then just load it instead of re-running it (unless user wants to overwrite)
		if not overwriteFlag and os.path.isfile(file_load_name):
			
			with open(file_load_name,'rb') as f_:
				(f,t,gc_eachsess,gc_sessavg) = pickle.load(f_)
				print('\n' + file_load_name + ' loaded.\n')

		else: #if the data doesn't exist, then get coherograms from lfp snippets
		

			if self.doNeptune:
				run = neptune.init_run(
			    project="ValueStimulation/LFP-dataproccessing",
			    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNzU1MmJkZC1lYzA3LTRhNWEtOGRlMC05N2MyOGViYTlkMGIifQ==",
			)
				
				run["time begin"] = time.ctime(time.time())
				
				
			num_sessions = len(self.file_list)
			gc_eachsess = []
			for k,file_name in enumerate(self.file_list): #loop thru sessions
			
				if self.doNeptune:
					run["current file"] = file_name
					
					
				print('-'*10)
				with open(file_name,'rb') as f_:
					session_data = pickle.load(f_)
				print(file_name + ' loaded.')
	
				ind = GetIndexes(self.behavior_list[k],self.subject,self.block,
						  self.choice,self.context,self.trial_type,self.rewarded,
						  self.stayshift,self.epoch,self.debug)
				data = np.transpose(session_data['LFP'][:,ind,:],axes=(1,0,2)) #switch axes of data in order to fit the way MNE wants it.
		
				# get combinations of Cd and ACC channels
				area1_chs,num_area1_chs,ch_locs = GetGoodChans(session_data['name'],self.areas[0]) #get good channels
				area2_chs,num_area2_chs,ch_locs = GetGoodChans(session_data['name'],self.areas[1])
		
				num_connections = num_area1_chs * num_area2_chs
				
				if num_connections > 0:
			
					cwt_freqs = np.arange(4.,62.,2.) #4-60Hz with 2hz step
					gc_n_lags = 40 #this controls spectral resolution-comp intensity tradeoff. 40 is default
# 					block_size = 1 #num connections to do at the same time. higher numbers are faster but require more memory. default is 1000
# 					full_rank = np.linalg.matrix_rank(session_data['LFP'][1])
# 					rank_size = 1
			
			
					# get coherograms for each channel combo
					start_time = time.time()
					conn_counter = 0
					
					
					for area1_ch in area1_chs:
						for area2_ch in area2_chs:
							
							start = time.time()
							
							#Cd->ACC
							sources = [[area1_ch]]
							targets = [[area2_ch]]
			
							indices = (np.array(sources,dtype='object'),np.array(targets,dtype='object'))		
								
							connectivity = spectral_connectivity_epochs(
								data = data,
								# names = session_data['chs']
								method = 'gc',
								indices = indices,
								sfreq = session_data['fs'],
								mode = 'cwt_morlet',
								fmin = 4., 
								fmax = 60.,
								cwt_freqs = cwt_freqs,
								cwt_n_cycles = cwt_freqs / 4,
								gc_n_lags = gc_n_lags,
								verbose = False,
								)
							
							
							if conn_counter == 0:
								connectivity_data = np.full((2,num_connections,len(connectivity.freqs),len(connectivity.times)),np.nan) #the 2 is for Cd->ACC and ACC->Cd, the 99 is len(cwt_freqs), and the 1017 is temporal dim
							
							connectivity_data[0,conn_counter,:,:] = connectivity.get_data()[0,:,:]
			
			
							#ACC->Cd
							sources = [[area2_ch]]
							targets = [[area1_ch]]
			
							indices = (np.array(sources,dtype='object'),np.array(targets,dtype='object'))		
								
							connectivity = spectral_connectivity_epochs(
								data = data,
								# names = session_data['chs']
								method = 'gc',
								indices = indices,
								sfreq = session_data['fs'],
								mode = 'cwt_morlet',
								fmin = 4., 
								fmax = 60.,
								cwt_freqs = cwt_freqs,
								cwt_n_cycles = cwt_freqs / 4,
								gc_n_lags = gc_n_lags,
								verbose = False,
								)
							
							connectivity_data[1,conn_counter,:,:] = connectivity.get_data()[0,:,:]
			
							
							runtime = np.round((time.time()-start)/60,4)
							time_left = np.round(runtime*(num_connections-conn_counter-1),2)
							print(f'{conn_counter+1}/{num_connections} done. Runtime: {runtime} mins. Time left: {time_left} mins.')
							conn_counter+=1
							
							
					t = np.array(connectivity.times)  + np.min(session_data['t']) # the +min(t) is to align new t vector with original t vector
					f = np.array(connectivity.freqs)
					gc_eachsess.append(np.nanmean(connectivity_data,axis=1)) #avg across all ch combos
					time_taken = np.round((time.time() - start_time)/60,2)
					if self.doNeptune:
						run["runtime for last file"] = str(time_taken) + " mins"
					print(f'{session_data["name"]} granger causality computed in {time_taken} mins. ({k+1}/{num_sessions})')
					
					del session_data
					del connectivity_data
					del connectivity
				
			gc_sessavg = np.nanmean(gc_eachsess,axis=0) #avg across sessions	
			
			#save out data so we don't have to rerun everytime
			with open(file_load_name,'wb') as f_:
				pickle.dump((f,t,gc_eachsess,gc_sessavg),f_)
			print('\n' + file_load_name + ' saved.\n')
		
			
		if self.doNeptune:			
			run["time end"] = time.ctime(time.time())			
			run.stop() #stop neptune logging
		
		self.f = f
		self.t = t
		self.gc_eachsess, self.gc_sessavg = np.array(gc_eachsess), gc_sessavg
		
		return


	def GetGC_RepSession(self,LFP_path,file_name,behavior_file,overwriteFlag):
		'''
		Get the time-freq representation of Granger Causality (GC) of a representative session
		between the two areas in both directions.
		
		This method provides a way to compute GC for one session at a time.
		
		Load time-aligned LFP snippets for a representative session and compute GC 
		between brain areas. GC is computed for each channel combination before
		the grand average is computed. I.e. GC is computed for num_chs_area1 x num_chs_area2
		combinations before being averaged to yield a single grand average of GC.
		
		Parameters
		-----------
		- LFP_path: str. 
			Path to folder where pkl file of time-aligned LFP snippets for chosen session resides.
			E.g. 'D:\\Value Stimulation\\Data\\Mario\\LFP\\'
		- file_name: str. 
			Pkl file of LFP snippets for chosen session.
			E.g. 'LFP_snippets_rp_Mario20161220.pkl'
		- behavior_file: str.
			Full path to behavior vectors pkl file for chosen session.
			E.g. 'D:\\Value Stimulation\\Data\\Mario\\Behavior\\BehaviorVectorsDictmari20161220_05_te2795.hdf.pkl'
		- overwriteFlag: bool. 
			If true, will not load previously saved results. Will get results anew and save them, overwriting previously saved results.
			If false, will load previously save results (if available).
		
		Does not return anything, but results in new object attributes:
		- t: 1D array of time points of time-freq representations of connectivity
		- f: 1D array of frequency bins of time-freq representations of connectivity
		- gc_dir1_rep: 2D array of trial-averaged time-freq representations of GC for rep session.
			direction = area1 --> area2
			shape = len(f) x len(t)
		- gc_dir2_rep: 2D array of trial-averaged time-freq representations of GC for rep session.
			direction = area2 --> area1
			shape = len(f) x len(t)

		'''
		print('-'*10 + '\nLoading Granger Causality')
		
		file_load_name = f'{self.LFPfolderpath}GrangerCausality_{self.subject}_{self.plot_subtitle}_{file_name}.pkl'
				
		#if the saved data already exists, then just load it instead of re-running it (unless user wants to overwrite)
		if not overwriteFlag and os.path.isfile(file_load_name):
			with open(file_load_name,'rb') as f_:
				(f,t,gc_dir1_rep,gc_dir2_rep) = pickle.load(f_)
				print('\n' + file_load_name + ' loaded.\n')

		else: #if the data doesn't exist, then get coherograms from lfp snippets
		
		
		
			if self.doNeptune:
				run = neptune.init_run(
			    project="ValueStimulation/LFP-dataproccessing",
			    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNzU1MmJkZC1lYzA3LTRhNWEtOGRlMC05N2MyOGViYTlkMGIifQ==",
			)
				
				run["time begin"] = time.ctime(time.time())
				
				
			num_sessions = len(self.file_list)
			gc_eachsess = []
			if self.doNeptune:
				run["current file"] = file_name
			
			assert f"_{self.epoch}_" in file_name, 'Chosen epoch does not match rep sess file!'	
			
			print('-'*10)
			with open(LFP_path + file_name + '.pkl','rb') as f_:
				session_data = pickle.load(f_)
			print(file_name + ' loaded.')
			
			with open(behavior_file,'rb') as f_:
				behavior = pickle.load(f_)
			print(f'{file_name} loaded.')

			ind = GetIndexes(behavior,self.subject,self.block,
						 self.choice,self.context,self.trial_type,self.rewarded,
						 self.stayshift,self.epoch,self.debug)
			data = np.transpose(session_data['LFP'][:,ind,:],axes=(1,0,2)) #switch axes of data in order to fit the way MNE wants it.
	
			# get combinations of Cd and ACC channels
			area1_chs,num_area1_chs,ch_locs = GetGoodChans(session_data['name'],self.areas[0]) #get good channels
			area2_chs,num_area2_chs,ch_locs = GetGoodChans(session_data['name'],self.areas[1])
	
			num_connections = num_area1_chs * num_area2_chs
			
			print(num_area1_chs,num_area2_chs)
			
			if num_connections > 0:
		
				cwt_freqs = np.arange(4.,62.,2.) #4-200Hz with 2hz step
				gc_n_lags = 40 #this controls spectral resolution-comp intensity tradeoff. 40 is default
				block_size = 1 #num connections to do at the same time. higher numbers are faster but require more memory. default is 1000
				full_rank = np.linalg.matrix_rank(session_data['LFP'][1])
				rank_size = 1
		
		
				# get coherograms for each channel combo
				start_time = time.time()
				conn_counter = 0
				
				
				for i,area1_ch in enumerate(area1_chs):
					for j,area2_ch in enumerate(area2_chs):
						
# 						if i>33 and i<36:
							
						start = time.time()
						
						#Cd->ACC
						sources = [[area1_ch]]
						targets = [[area2_ch]]
		
						indices = (np.array(sources,dtype='object'),np.array(targets,dtype='object'))		
							
						connectivity = spectral_connectivity_epochs(
							data = data,
							# names = session_data['chs']
							method = 'gc',
							indices = indices,
							sfreq = session_data['fs'],
							mode = 'cwt_morlet',
							fmin = 4., 
							fmax = 60.,
							cwt_freqs = cwt_freqs,
							cwt_n_cycles = cwt_freqs / 4,
							gc_n_lags = gc_n_lags,
							verbose = False,
							)
						
						
						if conn_counter == 0:
							connectivity_data = np.full((2,num_connections,len(connectivity.freqs),len(connectivity.times)),np.nan) #the 2 is for Cd->ACC and ACC->Cd, the 99 is len(cwt_freqs), and the 1017 is temporal dim
						
						connectivity_data[0,conn_counter,:,:] = connectivity.get_data()[0,:,:]
		
		
						#ACC->Cd
						sources = [[area2_ch]]
						targets = [[area1_ch]]
		
						indices = (np.array(sources,dtype='object'),np.array(targets,dtype='object'))		
							
						connectivity = spectral_connectivity_epochs(
							data = data,
							# names = session_data['chs']
							method = 'gc',
							indices = indices,
							sfreq = session_data['fs'],
							mode = 'cwt_morlet',
							fmin = 4., 
							fmax = 60.,
							cwt_freqs = cwt_freqs,
							cwt_n_cycles = cwt_freqs / 4,
							gc_n_lags = gc_n_lags,
							verbose = False,
							)
						
						connectivity_data[1,conn_counter,:,:] = connectivity.get_data()[0,:,:]
		
						
						runtime = np.round((time.time()-start)/60,4)
						time_left = np.round(runtime*(num_connections-conn_counter-1),2)
# 						print(f'{conn_counter+1}/{num_connections} done. Runtime: {runtime} mins. Time left: {time_left} mins.')
						conn_counter+=1
						print(f'{conn_counter}/{num_connections} connections completed. {time_left} mins left.')
						
						
				t = np.array(connectivity.times) + np.min(session_data['t']) # the +min(t) is to align new t vector with original t vector
				f = np.array(connectivity.freqs)
				gc_dir1_rep = np.nanmean(connectivity_data[0,:,:,:],axis=0) #avg across all connections
				gc_dir2_rep = np.nanmean(connectivity_data[1,:,:,:],axis=0) 
				
				time_taken = np.round((time.time() - start_time)/60,2)
				if self.doNeptune:
					run["runtime for last file"] = str(time_taken) + " mins"
				print(f'{session_data["name"]} granger causality computed in {time_taken} mins.')
				
				del session_data
				del connectivity_data
				del connectivity
			
			
			#save out data so we don't have to rerun everytime
			with open(file_load_name,'wb') as f_:
				pickle.dump((f,t,gc_dir1_rep,gc_dir2_rep),f_)
			print('\n' + file_load_name + ' saved.\n')
		
			
		if self.doNeptune:			
			run["time end"] = time.ctime(time.time())			
			run.stop() #stop neptune logging
		
		self.f = f
		self.t = t
		self.gc_dir1_rep = gc_dir1_rep
		self.gc_dir2_rep = gc_dir2_rep
		
		return
	
	#%% Frequency band methods
	
	def GetBandGC_RepSess(self,f1,f2):
		'''
		For representative session data
		Takes time-freq GC data and averages it over specified frequency band
		
		sig is the integral of time-freq representation of GC from f1 to f2

		Parameters
		----------
		f1 : float or int
			lower bound of frequency bin.
		f2 : float or int
			upper bound of frequency bin..

		Returns
		-------
		sigs : list of 1D arrays
			each element is the time course data of average frequency band power.
			sigs[0] is GC in direction area1 --> area2
			sigs[1] is GC in direction area2 --> area1
			
		'''
		
		bound1 = self.f > f1
		bound2 = self.f < f2
		freq_band = bound1 & bound2

		sig_dir1 = np.nanmean(self.gc_dir1_rep[freq_band,:],axis=0) #avg over freq band
		sig_dir2 = np.nanmean(self.gc_dir2_rep[freq_band,:],axis=0) #avg over freq band
		
		return [sig_dir1, sig_dir2]
	

	def GetBandGC(self,f1,f2):
		'''
		Takes time-freq GC data and averages it over specified frequency band
		
		sig is the integral of time-freq representation of GC from f1 to f2

		Parameters
		----------
		f1 : float or int
			lower bound of frequency bin.
		f2 : float or int
			upper bound of frequency bin..

		Returns
		-------
		sigs : list of 1D arrays
			each element is the time course data of average frequency band power.
			sigs[0] is GC in direction area1 --> area2
			sigs[1] is GC in direction area2 --> area1
		'''
		
		bound1 = self.f > f1
		bound2 = self.f < f2
		freq_band = bound1 & bound2
		
		sig_dir1 = np.nanmean(self.gc_eachsess[:,0,freq_band,:],axis=1) #avg over freq band
		sig_dir2 = np.nanmean(self.gc_eachsess[:,1,freq_band,:],axis=1) #avg over freq band
		
		return [sig_dir1, sig_dir2]
	
	
	def GetBandConnectivity(self,f1,f2):
		'''
		Takes time-freq connectivity data and averages it over specified frequency band
		
		sig is the integral of time-freq representation of connectivity from f1 to f2

		Parameters
		----------
		f1 : float or int
			lower bound of frequency bin.
		f2 : float or int
			upper bound of frequency bin..

		Returns
		-------
		sigs : list of 1D arrays
			each element is the time course data of average frequency band power.
			sigs[0] is connectivity in direction area1 --> area2
			sigs[1] is connectivity in direction area2 --> area1
		'''
		conns = self.conn_eachsess
		
		#self.conn_eachsess is num_areas x num_sessions x len(f) x len(t)

		bound1 = self.f > f1
		bound2 = self.f < f2
		freq_band = bound1 & bound2

		sigs = np.nanmean(conns[:,freq_band,:],axis=1) #avg over freq band
		pwrs = np.nanmean(sigs,axis=1) #avg over time
		
		return sigs, pwrs
	
	#%% Plotting methods
		
	def PlotBandConnectivityTogether(self,t1,t2,y1,y2,doLegendFlag,doylabelFlag,alldecoratorsFlag):
		'''
		Creates line plots of frequency band connectivity vs time.
		Takes time-freq representation of connectivity (coherogram) data and 
		averages it over specified frequency bands.

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		y1 : float
			Set lower limit of connectivity axis for plot.
		y2 : float
			Set upper limit of connectivity axis for plot.
		doLegendFlag : bool
			Set whether legend will be on or not
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.		
		savefigFlag : bool
			Choose whether to save the figure to the Reward Directionality Paper folder
		'''
		fig,ax = plt.subplots()
# 		ax.set_aspect(0.5)
		for (f1,f2),band_name,color in zip(self.freq_bands,self.freq_band_names,self.colors):
				
			sigs,_ = self.GetBandConnectivity(f1,f2)
			avg=np.nanmean(sigs,axis=0) #avg across sessions
			sem=np.nanstd(sigs,axis=0) / np.sqrt(np.size(sigs,axis=0)) #sem across sessions
			ax.plot(self.t,avg,color=color,label=f'{band_name}')
			ax.fill_between(self.t, avg+sem,avg-sem, color=color,alpha=0.2)
				
		if doLegendFlag: ax.legend(fontsize=16)
				
		if alldecoratorsFlag:
			ax.set_title(f'{self.areas[0]}-{self.areas[1]} Connectivity for all freq bands')
			ax.set_ylabel(self.connlabel)
			ax.set_xlabel(self.tlabel)
			fig.suptitle(self.plot_subtitle,fontsize=7)
		else:
			if not doylabelFlag:
				ax.set_yticklabels([])
			else:
				ax.yaxis.set_tick_params(labelsize=16)
				ax.set_ylabel(self.flabel,fontsize=16)
# 		ax.set_xticks([])
		ax.set_xlim([t1,t2])
		ax.set_ylim([y1,y2])
		ax.vlines(0,y1,y2,color='k',linewidth=2,linestyle='--')
		ax.xaxis.set_tick_params(labelsize=16)
		ax.set_xlabel(self.tlabel,fontsize=16)
		
		
		fig.tight_layout()


	def PlotBandConnectivityTogether_EachSession(self,t1,t2,y1,y2,doLegendFlag):
		'''
		Same as PlotBandConnectivityTogether, but also plots the connectivity for each
		session in faint color.
		
		Creates line plots of frequency band connectivity vs time
		Takes connectivity (coherogram) data and averages it over specified frequency band

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		y1 : float
			Set lower limit of connectivity axis for plot.
		y2 : float
			Set upper limit of connectivity axis for plot.
		doLegendFlag : bool
			Set whether legend will be on or not
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.		
		savefigFlag : bool
			Choose whether to save the figure to the Reward Directionality Paper folder
		'''
		
		fig,ax = plt.subplots()
		for (f1,f2),band_name,color in zip(self.freq_bands,self.freq_band_names,self.colors):
				
			sigs,_ = self.GetBandConnectivity(f1,f2)
			avg=np.nanmean(sigs,axis=0) #avg across sessions
			ax.plot(self.t,avg,color=color,label=f'{band_name}')
			for i in range(len(sigs)):
				ax.plot(self.t,sigs[i,:],color=color,alpha=0.2)
				
		if doLegendFlag: ax.legend()
		ax.set_title(f'{self.areas[0]}-{self.areas[1]} Connectivity for all freq bands')
		ax.set_ylabel(self.connlabel)
		ax.set_xlabel(self.tlabel)
		#ax.set_xticks(np.arange(0,1,0.1))
		ax.set_xlim([t1,t2])
		ax.set_ylim([y1,y2])
		
		fig.suptitle(self.plot_subtitle,fontsize=7)
		fig.tight_layout()
		

	def PlotCoherogram(self,t1,t2,f1,f2,v1,v2, doylabelFlag,doColorbarFlag,alldecoratorsFlag):
		'''
		Plots the session averaged time-frequency represenation of connectivity between brain areas (like a coherogram) 

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		f1 : float
			Set lower limit of freq axis for plot.
		f2 : float
			Set upper limit of freq axis for plot.
		v1 : float
			Set lower limit of connectivity (colorbar axis) for plot.
		v2 : float
			Set upper limit of connectivity (colorbar axis) for plot.
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		doColorbarFlag : bool
			Set whether plot will have colorbar.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.

		'''
		
		fig,ax = plt.subplots()
# 		ax.set_aspect(0.01745)
		c = ax.pcolormesh(self.t, self.f, self.conn_sessavg, vmin=v1,vmax=v2,cmap=self.cmap)
		ax.set_ylim([f1,f2])
		ax.set_xlim([t1,t2])
		ax.vlines(0,f1,f2,color='k',linewidth=3,linestyle='--') #to inicate t=0
		for j in range(len(self.freq_band_names)):
			ax.hlines(self.freq_bands[j],t1,t2,self.colors[j],linewidth=2) #to show freq bands
		if alldecoratorsFlag:
			ax.set_xlabel(self.tlabel)
			ax.set_ylabel(self.flabel)
			cbar = fig.colorbar(c, ax=ax)
			cbar.set_label(self.connlabel,rotation = 270)
			cbar.ax.get_yaxis().labelpad = 15
			fig.suptitle(self.plot_subtitle,fontsize=7)
		else:
			ax.set_xticklabels([])
			if not doylabelFlag:
				ax.set_yticklabels([])
			else:
				ax.yaxis.set_tick_params(labelsize=16)
				ax.set_ylabel(self.flabel,fontsize=16)
			if doColorbarFlag:
				cbar = fig.colorbar(c, ax=ax)
				cbar.set_label(self.connlabel,rotation = 270,fontsize=16)
				cbar.ax.get_yaxis().labelpad = 15
				cbar.ax.yaxis.set_tick_params(labelsize=16)
		
		
		fig.tight_layout() 

		
	def PlotGCRepSess(self,t1,t2,f1,f2,v1,v2,file_name):
		'''
		Plots the time-frequency represenation of Granger Causality 
		between brain areas (like a coherogram) for one representation session.

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		f1 : float
			Set lower limit of freq axis for plot.
		f2 : float
			Set upper limit of freq axis for plot.
		v1 : float
			Set lower limit of GC (colorbar axis) for plot.
		v2 : float
			Set upper limit of GC (colorbar axis) for plot.
		file_name: str. 
			Pkl file of spectrograms for chosen session.
			E.g. 'LFP_snippets_rp_Mario20161220'
		'''
		
		fig,axs = plt.subplots(1,2)

		#Cd to ACC
		ax=axs[0]
		c = ax.pcolormesh(self.t, self.f, self.gc_dir1_rep,vmin=v1,vmax=v2,cmap=self.gccmap)
		ax.set_xlabel(self.tlabel)
		ax.set_ylim([f1,f2])
		ax.set_xlim([t1,t2])
		ax.set_title(self.directions[0])
# 		cbar = fig.colorbar(c, ax=ax)
		ax.hlines([4,8,12,30],self.t[0],self.t[-1],color='k',linewidth=1) #freq bin lines
		ax.set_ylabel(self.flabel)
# 		cbar.set_label(self.gclabel,rotation = 270)
# 		cbar.ax.get_yaxis().labelpad = 15
		
		#ACC to Cd
		ax=axs[1]
		c = ax.pcolormesh(self.t, self.f, self.gc_dir2_rep,vmin=v1,vmax=v2,cmap=self.gccmap)
		ax.set_xlabel(self.tlabel)
		ax.set_ylim([f1,f2])
		ax.set_xlim([t1,t2])
		ax.set_title(self.directions[1])
		cbar = fig.colorbar(c, ax=ax)
		ax.hlines([4,8,12,30],self.t[0],self.t[-1],color='k',linewidth=1) #freq bin lines
# 		ax.set_ylabel(self.flabel)
		cbar.set_label(self.gclabel,rotation = 270)
		cbar.ax.get_yaxis().labelpad = 15
		
		fig.suptitle(f'GC: Representative session:{file_name}\n' + self.plot_subtitle,fontsize=7)
		fig.tight_layout() 
		
	def PlotGCRepSess2(self,t1,t2,f1,f2,v1,v2, doylabelFlag,doColorbarFlag,alldecoratorsFlag):
		'''
		Plots the time-frequency represenation of Granger Causality 
		between brain areas (like a coherogram) for one representation session.

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		f1 : float
			Set lower limit of freq axis for plot.
		f2 : float
			Set upper limit of freq axis for plot.
		v1 : float
			Set lower limit of connectivity (colorbar axis) for plot.
		v2 : float
			Set upper limit of connectivity (colorbar axis) for plot.
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		doColorbarFlag : bool
			Set whether plot will have colorbar.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.
		'''
		
		gcs = [self.gc_dir1_rep,self.gc_dir2_rep]
		for i,direction in enumerate(self.directions): #loop thru areas
		
			fig,ax = plt.subplots()
			ax.set_aspect(0.01745)
			c = ax.pcolormesh(self.t, self.f, gcs[i], vmin=v1,vmax=v2, cmap=self.gccmap)
			
			ax.set_ylim([f1,f2])
			ax.set_xlim([t1,t2])
			ax.vlines(0,f1,f2,color='k',linewidth=3,linestyle='--')
			
			if alldecoratorsFlag:
				ax.set_xlabel(self.tlabel)
				ax.set_ylabel(self.flabel)
				cbar = fig.colorbar(c, ax=ax)
				cbar.set_label(self.gclabel,rotation = 270)
				cbar.ax.get_yaxis().labelpad = 15
				ax.set_title(direction)
				fig.suptitle(self.plot_subtitle,fontsize=7)
			else:
				ax.set_xticklabels([])
				ax.set_title(direction,fontsize=16)
				if not doylabelFlag:
					ax.set_yticklabels([])
				else:
					ax.yaxis.set_tick_params(labelsize=16)
					ax.set_ylabel(self.flabel,fontsize=16)
				if doColorbarFlag:
					cbar = fig.colorbar(c, ax=ax)
					cbar.set_label(self.gclabel,rotation = 270,fontsize=16)
					cbar.ax.get_yaxis().labelpad = 15
					cbar.ax.yaxis.set_tick_params(labelsize=16)

			#ax.hlines([4,8,12,30],self.t_spect[0],self.t_spect[-1],color='k',linewidth=1) #freq bin lines
# 			ax.set_aspect(1.6)
			fig.tight_layout() 


	def PlotGC(self,t1,t2,f1,f2,v1,v2, doylabelFlag,doColorbarFlag,alldecoratorsFlag):
		'''
		Plots the session averaged time-frequency represenation of Granger Causality 
		between brain areas (like a coherogram).

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		f1 : float
			Set lower limit of freq axis for plot.
		f2 : float
			Set upper limit of freq axis for plot.
		v1 : float
			Set lower limit of connectivity (colorbar axis) for plot.
		v2 : float
			Set upper limit of connectivity (colorbar axis) for plot.
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		doColorbarFlag : bool
			Set whether plot will have colorbar.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.
		'''
		gcs = [self.gc_sessavg[0],self.gc_sessavg[1]]
		for i,direction in enumerate(self.directions): #loop thru directionalities
		
			fig,ax = plt.subplots()
			c = ax.pcolormesh(self.t, self.f, gcs[i], vmin=v1,vmax=v2, cmap=self.gccmap)
			
			ax.set_ylim([f1,f2])
			ax.set_xlim([t1,t2])
			ax.vlines(0,f1,f2,color='k',linewidth=3,linestyle='--')
			
			for j in range(len(self.freq_band_names)):
				ax.hlines(self.freq_bands[j],t1,t2,self.colors[j],linewidth=2) #to show freq bands
			
			if alldecoratorsFlag:
				ax.set_xlabel(self.tlabel)
				ax.set_ylabel(self.flabel)
				cbar = fig.colorbar(c, ax=ax)
				cbar.set_label(self.gclabel,rotation = 270)
				cbar.ax.get_yaxis().labelpad = 15
				ax.set_title(direction)
				fig.suptitle(self.plot_subtitle,fontsize=7)
			else:
				ax.set_xticklabels([])
				ax.set_title(direction,fontsize=16)
				if not doylabelFlag:
					ax.set_yticklabels([])
				else:
					ax.yaxis.set_tick_params(labelsize=16)
					ax.set_ylabel(self.flabel,fontsize=16)
				if doColorbarFlag:
					cbar = fig.colorbar(c, ax=ax)
					cbar.set_label(self.gclabel,rotation = 270,fontsize=16)
					cbar.ax.get_yaxis().labelpad = 15
					cbar.ax.yaxis.set_tick_params(labelsize=16)

			#ax.hlines([4,8,12,30],self.t_spect[0],self.t_spect[-1],color='k',linewidth=1) #freq bin lines
# 			ax.set_aspect(1.6)
			fig.tight_layout() 


	def PlotBandGCRepSessTogether2(self,t1,t2,y1,y2,doLegendFlag,doylabelFlag,alldecoratorsFlag):
		'''
		Creates line plots of frequency band GC vs time for a representative session.
		Takes time-freq representation of GC data and 
		averages it over specified frequency bands.

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		y1 : float
			Set lower limit of connectivity axis for plot.
		y2 : float
			Set upper limit of connectivity axis for plot.
		doLegendFlag : bool
			Set whether legend will be on or not
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.		
		savefigFlag : bool
			Choose whether to save the figure to the Reward Directionality Paper folder
		'''
		
		fig,ax = plt.subplots()
# 		ax.set_aspect(0.5)
		for (f1,f2),band_name,color in zip(self.freq_bands,self.freq_band_names,self.colors):
			 
			for i,(direction,linestyle) in enumerate(zip(self.directions,self.linestyles)):
				
				sigs = self.GetBandGC_RepSess(f1,f2)
# 				avg=np.nanmean(sigs[i],axis=0) #avg across sessions
# 				sem=np.nanstd(sigs[i],axis=0) / np.sqrt(np.size(sigs[i,:,:],axis=0)) #sem across sessions
				ax.plot(self.t,sigs[i],linestyle=linestyle,color=color,label=f'{direction} {band_name}')
# 				ax.fill_between(self.t, avg+sem,avg-sem, color=color,alpha=0.2)
				
		if doLegendFlag: ax.legend(fontsize=16)
		
		ax.set_xlabel(self.tlabel,fontsize=16)
		ax.xaxis.set_tick_params(labelsize=16)
		
		if alldecoratorsFlag:
			ax.set_title(f'{self.directions[0]} vs {self.directions[1]} GC')
			ax.set_ylabel(self.gclabel)
			fig.suptitle(self.plot_subtitle,fontsize=7)
			ax.legend()
			ax.set_xlabel(self.tlabel,fontsize=10)
			ax.xaxis.set_tick_params(labelsize=10)
		else:
			if not doylabelFlag:
				ax.set_yticklabels([])
			else:
				ax.yaxis.set_tick_params(labelsize=16)
				ax.set_ylabel(self.gclabel,fontsize=16)
		
		ax.vlines(0,y1,y2,color='k',linewidth=2,linestyle='--')
		#ax.set_xticks(np.arange(0,1,0.1))
		ax.set_xlim([t1,t2])
		ax.set_ylim([y1,y2])
	
# 		ax.set_aspect(1.6)
		fig.tight_layout()



	def PlotBandGCTogether(self,t1,t2,y1,y2,doLegendFlag,doylabelFlag,alldecoratorsFlag):
		'''
		Creates session averaged line plots of frequency band GC vs time.
		Takes time-freq representation of GC data and 
		averages it over specified frequency bands.

		Parameters
		----------
		t1 : float
			Set lower limit of time axis for plot.
		t2 : float
			Set upper limit of time axis for plot.
		y1 : float
			Set lower limit of connectivity axis for plot.
		y2 : float
			Set upper limit of connectivity axis for plot.
		doLegendFlag : bool
			Set whether legend will be on or not
		doylabelFlag : bool
			Set whether y-axis will have label and tick numbering.
		alldecoratorsFlag : bool
			Set whether all applicable decorators will be turned on.
			Best for stand alone plots.		
		savefigFlag : bool
			Choose whether to save the figure to the Reward Directionality Paper folder
		'''
		
		fig,ax = plt.subplots()
# 		ax.set_aspect(0.5)
		for (f1,f2),band_name,color in zip(self.freq_bands,self.band_names,self.colors):
			 
			for i,(direction,linestyle) in enumerate(zip(self.directions,self.linestyles)):
				
				sigs = self.GetBandGC(f1,f2)
				avg=np.nanmean(sigs[i],axis=0) #avg across sessions
				sem=np.nanstd(sigs[i],axis=0) / np.sqrt(np.size(sigs[i],axis=0)) #sem across sessions
				ax.plot(self.t,avg,linestyle=linestyle,color=color,label=f'{direction} {band_name}')
				ax.fill_between(self.t, avg+sem,avg-sem, color=color,alpha=0.2)
				
		if doLegendFlag: ax.legend(fontsize=16)
		
		ax.set_xlabel(self.tlabel,fontsize=16)
		ax.xaxis.set_tick_params(labelsize=16)
		
		if alldecoratorsFlag:
			ax.set_title(f'{self.directions[0]} vs {self.directions[1]} GC')
			ax.set_ylabel(self.gclabel)
			fig.suptitle(self.plot_subtitle,fontsize=7)
			ax.legend()
			ax.set_xlabel(self.tlabel,fontsize=10)
			ax.xaxis.set_tick_params(labelsize=10)
		else:
			if not doylabelFlag:
				ax.set_yticklabels([])
			else:
				ax.yaxis.set_tick_params(labelsize=16)
				ax.set_ylabel(self.gclabel,fontsize=16)
		
		ax.vlines(0,y1,y2,color='k',linewidth=2,linestyle='--')
		#ax.set_xticks(np.arange(0,1,0.1))
		ax.set_xlim([t1,t2])
		ax.set_ylim([y1,y2])
	
# 		ax.set_aspect(1.6)
		fig.tight_layout()	
		