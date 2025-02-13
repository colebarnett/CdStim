# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:11:53 2024

@author: coleb
"""
import pickle
import numpy as np



subject = 'Luigi'
epoch = 'rp'
stim_or_sham = 'All Sessions'
block = 'BlA'
choice = 'a=any'
context = ''
trial_type = '' 
rewarded = 'Rewarded'


plot_subtitle = f'{epoch}, {stim_or_sham}, {block}, {choice}, {context}, {trial_type}, {rewarded}'
file_path='E:\\Value Stimulation\\Data\\Luigi\\LFP\\'
file_list=[ "LFP_snippets_rp_igi20170915-2",
			"LFP_snippets_rp_Luigi20170831",
			"LFP_snippets_rp_Luigi20170902",
			"LFP_snippets_rp_Luigi20170907",
			"LFP_snippets_rp_Luigi20170927",
			"LFP_snippets_rp_Luigi20171001",
			"LFP_snippets_rp_Luigi20171003",
			"LFP_snippets_rp_Luigi20171005",
			"LFP_snippets_rp_Luigi20171015",
			"LFP_snippets_rp_Luigi20171019",
			"LFP_snippets_rp_Luigi20171028"]


gc_eachsess = []
for file_name in file_list:
	
	repsess_file_load_name = f'{file_path}GrangerCausality_{subject}_{plot_subtitle}_{file_name}.pkl'
	
	try: 
		with open(repsess_file_load_name,'rb') as f_:
			(f,t,gc_Cd2ACC_rep,gc_ACC2Cd_rep) = pickle.load(f_)
			print('\n' + repsess_file_load_name + ' loaded.\n')
		
		gc_eachsess.append(np.array([gc_Cd2ACC_rep,gc_ACC2Cd_rep]))
		

	except FileNotFoundError:
		print(f'{repsess_file_load_name} not found!')
		

gc_eachsess = np.array(gc_eachsess)
gc_sessavg = np.nanmean(gc_eachsess,axis=0) #avg over sessions

print(np.shape(gc_eachsess))
print(np.shape(gc_sessavg))
		
#save out data 
file_load_name = f'{file_path}GrangerCausality_{subject}_{plot_subtitle}.pkl'
with open(file_load_name,'wb') as f_:
	pickle.dump((f,t,gc_eachsess,gc_sessavg),f_)
print('\n' + file_load_name + ' saved.\n')