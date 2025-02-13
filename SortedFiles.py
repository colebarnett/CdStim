# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:19:37 2023

@author: coleb
"""


def GetFileList(subject, stim_or_sham):
	'''
	
	Gets the filenames and paths for the hdf, offline sorted spikes, and syncHDF files
	
	Parameters
	----------
	subject : 'Mario' or 'Luigi'
		Determines which set of files to retrieve
	stim_or_sham : int
		Choose whether to get stim or sham files. 0=sham, 1=stim.

	Returns
	-------
	paths : dictionary containing 3 strings.
		Gives the paths needed to get to the hdf, offline sorted spikes, syncHDF files.	
	filenames : dictionary containing 3 lists of filenames
		Each item in a list corresponds to the file(s) for a session.

	'''
	
	#%% Luigi
	if subject == 'Luigi':
		
		# Specify Data Files
		
		#### hdf files
		
		# Paths to Where the hdf files are:
		path_hdf = "E:\\Value Stimulation\\Data\\Luigi\\hdf\\"

		filenames_sham_hdf = [
		                        ['luig20170822_07_te133.hdf'],
		                        #['luig20170824_02_te139.hdf'], something weird happening with this one. samps_align goes past number of lfp samples. need to check samps align pipeline
		                        ['luig20170831_11_te183.hdf'],
		                        ['luig20170902_08_te197.hdf'],
		                        ['luig20170907_04_te212.hdf'],
								# ['luig20170915_13_te262.hdf'],    #for neural analysis only, exclude for behavior analysis
		                        # ['luig20170924_04_te291.hdf'],  #no learning, exclude
		                        #['luig20170929_11_te376.hdf', skip two part sessions for now for lfp prelim
		                        # 'luig20170929_13_te378.hdf'],
		                        ['luig20171003_08_te399.hdf'],
		                        ['luig20171015_03_te447.hdf'],
		                        ['luig20171019_07_te467.hdf'],
		                        # ['luig20171024_10_te483.hdf'],    #data not saved completely
		                        ['luig20171028_08_te500.hdf']
		                        ]
		
		filenames_stim_hdf = [
								# ['luig20170909_08_te226.hdf'],  #no learning, exclude
								['luig20170915_15_te264.hdf'],
		                        ['luig20170927_07_te361.hdf'],
		                        ['luig20171001_06_te385.hdf'],
		                        ['luig20171005_02_te408.hdf'],
		                        #['luig20171017_04_te454.hdf', skip two part sessions for now for lfp prelim
		                        # 'luig20171017_05_te455.hdf'],
		                        # ['luig20171026_03_te488.hdf',   #for behavior analysis only, exclude for neural analysis
		                        # 'luig20171026_04_te489.hdf'],
		                        # ['luig20171031_04_te509.hdf'],    #data not saved completely 
								# ['luig20171108_03_te543.hdf']     #for neural analysis only, exclude for behavior analysis
		                        ]
		
		####offline sorted spike files
		
		# Paths to Where the offline sorted spike files are:
		#path_spikes = "D:\\Value Stimulation\\Data\\Luigi\\offlineSortedSpikes_original\\" #doesn't have good units marked
		#path_spikes = "D:\\Value Stimulation\\Data\\Luigi\\offlineSortedSpikes_truncated\\" #has good units marked, but some files are truncated
		path_spikes = "D:\\Value Stimulation\\Data\\Luigi\\offlineSortedSpikes\\" #has good units marked and are not truncated

		filenames_sham_spikes = [
		                        [['Luigi20170822_Block-2_eNe1_Offline.csv']],
		                        [['Luigi20170824_Block-1_eNe1_Offline.csv']],
		                        [['Luigi20170831_Block-2_eNe1_Offline.csv'],['Luigi20170831_Block-2_eNe2_Offline.csv']],
		                        [['Luigi20170902_Block-1_eNe1_Offline.csv']],
		                        [['Luigi20170907_Block-1_eNe1_Offline.csv'],['Luigi20170907_Block-1_eNe2_Offline.csv']],
		                        # [['Luigi20170915_Block-1_eNe1_Offline.csv'],['Luigi20170915_Block-1_eNe2_Offline.csv']], #for neural analysis only (no behavior)
		                        # ['Luigi20170924_Block-1_eNe1_Offline.csv'],['Luigi20170924_Block-1_eNe2_Offline.csv'],#no learning, exclude
		                        [['Luigi20170929_Block-3_eNe1_Offline.csv','Luigi20170929_Block-4_eNe1_Offline.csv'],
								 ['Luigi20170929_Block-3_eNe2_Offline.csv','Luigi20170929_Block-4_eNe2_Offline.csv']],
		                        [['Luigi20171003_Block-1_eNe1_Offline.csv'],['Luigi20171003_Block-1_eNe2_Offline.csv']],
		                        [['Luigi20171015_Block-1_eNe1_Offline.csv'],['Luigi20171015_Block-1_eNe2_Offline.csv']],
		                        [['Luigi20171019_Block-1_eNe1_Offline.csv'],['Luigi20171019_Block-1_eNe2_Offline.csv']],
		                        #[['Luigi20171024_Block-1_eNe1_Offline.csv'],['Luigi20171024_Block-1_eNe2_Offline.csv']], #data not saved completely
		                        [['Luigi20171028_Block-1_eNe1_Offline.csv'],['Luigi20171028_Block-1_eNe2_Offline.csv']]
		                        ]
		
		filenames_stim_spikes = [
								# ['Luigi20170909_Block-3_eNe1_Offline.csv','Luigi20170909_Block-3_eNe2_Offline.csv'],#no learning, exclude
								[['Luigi20170915-2_Block-1_eNe1_Offline.csv'],['Luigi20170915-2_Block-1_eNe2_Offline.csv']],
		                        [['Luigi20170927_Block-3_eNe1_Offline.csv'],['Luigi20170927_Block-3_eNe2_Offline.csv']],
		                        [['Luigi20171001_Block-1_eNe1_Offline.csv'],['Luigi20171001_Block-1_eNe2_Offline.csv']],
		                        [['Luigi20171005_Block-1_eNe1_Offline.csv']],
		                        [['Luigi20171017_Block-1_eNe1_Offline.csv','Luigi20171017_Block-2_eNe1_Offline.csv'],
								 ['Luigi20171017_Block-1_eNe2_Offline.csv','Luigi20171017_Block-2_eNe2_Offline.csv']],
		                        #[['Luigi20171026_Block-1_eNe1_Offline.csv','Luigi20171026_Block-2_eNe1_Offline.csv'],
								# ['Luigi20171026_Block-1_eNe2_Offline.csv','Luigi20171026_Block-2_eNe2_Offline.csv']], #for behvaior analysis only, exclude for neural analysis
		                        # [['Luigi20171031_Block-1_eNe1_Offline.csv'],['Luigi20171031_Block-1_eNe2_Offline.csv']], #data not saved completely
								# [['Luigi20171108_Block-1_eNe1_Offline.csv'],['Luigi20171108_Block-1_eNe2_Offline.csv']]#for neural analysis only (no behavior)
		                        ]
		
		#### syncHDF files
		
		# Paths to Where the syncHDF files are:
		path_syncHDF = "E:\\Value Stimulation\\Data\\Luigi\\syncHDF\\"
		
		# (some files are repeated in the list so that they can match with the offline sorted spike files they correspond to)
		filenames_sham_syncHDF = [
		                        ['Luigi20170822_b2_syncHDF.mat'],
		                        #['Luigi20170824_b1_syncHDF.mat'],
		                        ['Luigi20170831_b2_syncHDF.mat'],
		                        ['Luigi20170902_b1_syncHDF.mat'],
		                        ['Luigi20170907_b1_syncHDF.mat'],
		                        # ['Luigi20170915_b1_syncHDF.mat'],    #for neural analysis only (no behavior)
		                        # ['Luigi20170924_b1_syncHDF.mat'],  #no learning, exclude
# 		                        ['Luigi20170929_b3_syncHDF.mat', skip two part sessions for now for lfp prelim
# 								 'Luigi20170929_b4_syncHDF.mat'],
		                        ['Luigi20171003_b1_syncHDF.mat'],
		                        ['Luigi20171015_b1_syncHDF.mat'],
		                        ['Luigi20171019_b1_syncHDF.mat'],
		                        # ['Luigi20171024_b1_syncHDF.mat'],    #data not saved completely
		                        ['Luigi20171028_b1_syncHDF.mat']
		                        ]
		
		filenames_stim_syncHDF = [
								# ['Luigi20170909_b3_syncHDF.mat'],  #no learning, exclude
								['Luigi20170915-2_b1_syncHDF.mat'],
		                        ['Luigi20170927_b3_syncHDF.mat'],
		                        ['Luigi20171001_b1_syncHDF.mat'],
		                        ['Luigi20171005_b1_syncHDF.mat'],
# 		                        ['Luigi20171017_b1_syncHDF.mat', skip two part sessions for now for lfp prelim
# 								 'Luigi20171017_b2_syncHDF.mat'],
		                        # ['Luigi20171026_b1_syncHDF.mat',  #for behvaior analysis only, exclude for neural analysis
								# 'Luigi20171026_b2_syncHDF.mat'],
		                        # ['Luigi20171031_b1_syncHDF.mat'],    #data not saved completely 
								# ['Luigi20171108_b1_syncHDF.mat'],    #for neural analysis only (no behavior)
		                        ]

		#### tdt files
		
		# Paths to Where the tdt files are:
		path_tdt = "E:\\Value Stimulation\\Data\\Luigi\\tdt\\"

		filenames_sham_tdt = [
		                        ['Luigi20170822\\Block-2'],
		                        #['Luigi20170824\\Block-2'],
		                        ['Luigi20170831\\Block-2'],
		                        ['Luigi20170902\\Block-1'],
		                        ['Luigi20170907\\Block-1'],
								# ['Luigi20170915\\Block-1'],    #for neural analysis only, exclude for behavior analysis
		                        # ['Luigi20170924\\Block-1'],  #no learning, exclude
# 		                        ['Luigi20170929\\Block-3', skip two part sessions for now for lfp prelim
# 		                         'Luigi20170929\\Block-4'],
		                        ['Luigi20171003\\Block-1'],
		                        ['Luigi20171015\\Block-1'],
		                        ['Luigi20171019\\Block-1'],
		                        # ['Luigi20171024\\Block-1'],    #data not saved completely
		                        ['Luigi20171028\\Block-1']
		                        ]
		
		filenames_stim_tdt = [
								# ['Luigi20170909\\Block3'],  #no learning, exclude
								['Luigi20170915-2\\Block-1'],
		                        ['Luigi20170927\\Block-3'],
		                        ['Luigi20171001\\Block-1'],
		                        ['Luigi20171005\\Block-1'],
# 		                        ['Luigi20171017\\Block-1', skip two part sessions for now for lfp prelim
# 		                         'Luigi20171017\\Block-2'],
		                        # ['Luigi20171026\\Block-1',   #for behavior analysis only, exclude for neural analysis
		                        # 'Luigi20171026\\Block-2'],
		                        # ['Luigi20171031\\Block-1'],    #data not saved completely 
								# ['Luigi20171108\\Block-1']     #for neural analysis only, exclude for behavior analysis
		                        ]


	#%% Mario
	if subject == 'Mario':
		
		# Specify Data Files
		
		#### hdf files
		# Paths to Where the hdf files are:
		path_hdf = "D:\\Value Stimulation\\Data\\Mario\\hdf\\"

		filenames_sham_hdf = [
			
		                        ['mari20161220_05_te2795.hdf'],
								
		                        ['mari20170106_03_te2818.hdf'],
								
		                        #['mari20170119_03_te2878.hdf', #excluded until alignment issue fixed
							     #'mari20170119_05_te2880.hdf'],
								
		                        #['mari20170126_03_te2931.hdf', #only half of spikes recorded correctly. recording incomplete. For behavior analysis only (no neural analysis)
 							    # 'mari20170126_05_te2933.hdf'],
								
# 		                        ['mari20170126_07_te2935.hdf',
# 							     'mari20170126_11_te2939.hdf'],
								
								['mari20170204_03_te2996.hdf'],   
 								 
		                        ['mari20170207_05_te3018.hdf',
 		                         'mari20170207_07_te3020.hdf',
 								 'mari20170207_09_te3022.hdf'],
								
		                        ['mari20170207_13_te3026.hdf',
							     'mari20170207_15_te3028.hdf',
 								 'mari20170207_17_te3030.hdf',
 								 'mari20170207_19_te3032.hdf'],
								
# 		                        ['mari20170214_03_te3085.hdf',
# 							     'mari20170214_07_te3089.hdf',
# 								 'mari20170214_09_te3091.hdf',
# 								 'mari20170214_11_te3093.hdf',
# 								 'mari20170214_13_te3095.hdf',
# 								 'mari20170214_16_te3098.hdf'],
								
		                        ['mari20170215_03_te3101.hdf',
							     'mari20170215_05_te3103.hdf',
 								 'mari20170215_07_te3105.hdf']   
 								 
		                        #['mari20170220_07.hdf',
 							    # 'mari20170220_09.hdf',
								# 'mari20170220_11.hdf', #recording incomplete. For behavior analysis only (no neural analysis)
								# 'mari20170220_14.hdf']
	                        ]
		
		filenames_stim_hdf = [
								
								['mari20161221_03_te2800.hdf'],
								
		                        ['mari20161222_03_te2803.hdf'],
								
		                        ['mari20170108_03_te2821.hdf'],
								
# 		                        ['mari20170125_10_te2924.hdf',
# 							     'mari20170125_12_te2926.hdf',
#  								 'mari20170125_14_te2928.hdf'],
# 								
# 	                        	#['mari20170130_12_te2960.hdf',
# 							    # 'mari20170130_13_te2961.hdf'],
# 								
# 								['mari20170131_05_te2972.hdf',
#  								 'mari20170131_07_te2974.hdf'],
								
								['mari20170201_03_te2977.hdf'],
								
# 								['mari20170202_06_te2985.hdf',
#  								 'mari20170202_08_te2987.hdf'],
								
								#['mari20170208_07_te3039.hdf',
								# 'mari20170208_09_te3041.hdf', #recording incomplete. For behavior analysis only (no neural analysis)
								# 'mari20170208_11_te3043.hdf'],
								
								['mari20170209_03_te3047.hdf',
 								 'mari20170209_05_te3049.hdf',
 								 'mari20170209_08_te3052.hdf'],
								
								['mari20170216_03_te3108.hdf',
 								 'mari20170216_05_te3110.hdf',
 								 'mari20170216_08_te3113.hdf',
 								 'mari20170216_10_te3115.hdf'],
								
								['mari20170219_14.hdf',
 								 'mari20170219_16.hdf',
 								 'mari20170219_18.hdf']
	                          
							  ]
		####offline sorted spike files
		
		# Paths to Where the offline sorted spike files are:
		path_spikes = "D:\\Value Stimulation\\Data\\Mario\\offlineSortedSpikes\\" #has good units marked and are not truncated
		
		filenames_sham_spikes = [
			
		                        [['Mario20161220_Block-1_eNe1_Offline.csv'],
							     ['Mario20161220_Block-1_eNe2_Offline.csv']],
								
		                        [['Mario20170106_Block-1_eNe1_Offline.csv'],
							     ['Mario20170106_Block-1_eNe2_Offline.csv']],
								
		                        #[['Mario20170119_Block-1_eNe1_Offline.csv','Mario20170119-2_Block-1_eNe1_Offline.csv'], #excluded until alignment issue fixed
							    # ['Mario20170119_Block-1_eNe2_Offline.csv','Mario20170119-2_Block-1_eNe2_Offline.csv']],
								
								#[['Mario20170126_Block-1_eNe1_Offline.csv'],
								# ['Mario20170126_Block-1_eNe2_Offline.csv']],
								
								[['Mario20170126-2_Block-1_eNe1_Offline.csv','Mario20170126-3_Block-1_eNe1_Offline.csv'],
								 ['Mario20170126-2_Block-1_eNe2_Offline.csv','Mario20170126-3_Block-1_eNe2_Offline.csv']],
								
								[['Mario20170204_Block-1_eNe1_Offline.csv'],
								 ['Mario20170204_Block-1_eNe2_Offline.csv']],
								
								[['Mario20170207_Block-1_eNe1_Offline.csv','Mario20170207_Block-2_eNe1_Offline.csv','Mario20170207_Block-3_eNe1_Offline.csv'],
								 ['Mario20170207_Block-1_eNe2_Offline.csv','Mario20170207_Block-2_eNe2_Offline.csv','Mario20170207_Block-3_eNe2_Offline.csv']],
								
								[['Mario20170207-2_Block-1_eNe1_Offline.csv','Mario20170207-2_Block-2_eNe1_Offline.csv','Mario20170207-2_Block-3_eNe1_Offline.csv','Mario20170207-2_Block-4_eNe1_Offline.csv'],
								 ['Mario20170207-2_Block-1_eNe2_Offline.csv','Mario20170207-2_Block-2_eNe2_Offline.csv','Mario20170207-2_Block-3_eNe2_Offline.csv']],
								
								[['Mario20170214_Block-1_eNe1_Offline.csv','Mario20170214_Block-2_eNe1_Offline.csv','Mario20170214_Block-3_eNe1_Offline.csv','Mario20170214_Block-4_eNe1_Offline.csv','Mario20170214-2_Block-1_eNe1_Offline.csv','Mario20170214-2_Block-2_eNe1_Offline.csv'],
								 ['Mario20170214_Block-1_eNe2_Offline.csv','Mario20170214_Block-2_eNe2_Offline.csv','Mario20170214_Block-3_eNe2_Offline.csv','Mario20170214_Block-4_eNe2_Offline.csv','Mario20170214-2_Block-1_eNe2_Offline.csv','Mario20170214-2_Block-2_eNe2_Offline.csv']],
								
								[['Mario20170215_Block-1_eNe1_Offline.csv','Mario20170215_Block-2_eNe1_Offline.csv','Mario20170215_Block-3_eNe1_Offline.csv'],
								 ['Mario20170215_Block-1_eNe2_Offline.csv','Mario20170215_Block-2_eNe2_Offline.csv','Mario20170215_Block-3_eNe2_Offline.csv']]
								
								#[['Mario20170220_Block-1_eNe1_Offline.csv','Mario20170220_Block-2_eNe1_Offline.csv','Mario20170220_Block-3_eNe1_Offline.csv','Mario20170220_Block-4_eNe1_Offline.csv'],
								# ['Mario20170220_Block-1_eNe2_Offline.csv','Mario20170220_Block-2_eNe2_Offline.csv','Mario20170220_Block-3_eNe2_Offline.csv','Mario20170220_Block-4_eNe2_Offline.csv']],
								
								]
		
		filenames_stim_spikes = [
			
								[['Mario20161221_Block-1_eNe1_Offline.csv'],
								 ['Mario20161221_Block-1_eNe2_Offline.csv']],
								
								[['Mario20161222_Block-1_eNe1_Offline.csv'],
								 ['Mario20161222_Block-1_eNe2_Offline.csv']],
								
								[['Mario20170108_Block-1_eNe1_Offline.csv'],
								 ['Mario20170108_Block-1_eNe2_Offline.csv']],
								
								[['Mario20170125_Block-1_eNe1_Offline.csv','Mario20170125-2_Block-1_eNe1_Offline.csv','Mario20170125-3_Block-1_eNe1_Offline.csv'],
								 ['Mario20170125_Block-1_eNe2_Offline.csv','Mario20170125-2_Block-1_eNe2_Offline.csv','Mario20170125-3_Block-1_eNe2_Offline.csv']],
								
								#[['Mario20170130-2-Block1_eNe1_Offline.csv'],
							    # ['Mario20170130-2-Block1_eNe2_Offline.csv']],
								
								[['Mario20170131_Block-1_eNe1_Offline.csv','Mario20170131-2_Block-1_eNe1_Offline.csv'],
								 ['Mario20170131_Block-1_eNe2_Offline.csv','Mario20170131-2_Block-1_eNe2_Offline.csv']],
								
								[['Mario20170201_Block-1_eNe1_Offline.csv'],
								 ['Mario20170201_Block-1_eNe2_Offline.csv']],
								
								[['Mario20170202_Block-1_eNe1_Offline.csv','Mario20170202-2_Block-1_eNe1_Offline.csv'],
								 ['Mario20170202_Block-1_eNe2_Offline.csv','Mario20170202-2_Block-1_eNe2_Offline.csv']],
								
								#[['Mario20170208-2_Block-1_eNe1_Offline.csv','Mario20170208-2_Block-2_eNe1_Offline.csv','Mario20170208-2_Block-3_eNe1_Offline.csv'],
								# ['Mario20170208-2_Block-1_eNe2_Offline.csv','Mario20170208-2_Block-2_eNe2_Offline.csv','Mario20170208-2_Block-3_eNe2_Offline.csv']],
								
								[['Mario20170209_Block-1_eNe1_Offline.csv','Mario20170209_Block-2_eNe1_Offline.csv','Mario20170209_Block-3_eNe1_Offline.csv'],
								 ['Mario20170209_Block-1_eNe2_Offline.csv','Mario20170209_Block-2_eNe2_Offline.csv','Mario20170209_Block-3_eNe2_Offline.csv']],
								
								[['Mario20170216_Block-1_eNe1_Offline.csv','Mario20170216_Block-2_eNe1_Offline.csv','Mario20170216_Block-3_eNe1_Offline.csv','Mario20170216_Block-4_eNe1_Offline.csv'],
								 ['Mario20170216_Block-1_eNe2_Offline.csv','Mario20170216_Block-2_eNe2_Offline.csv','Mario20170216_Block-3_eNe2_Offline.csv','Mario20170216_Block-4_eNe2_Offline.csv']],
								
								[['Mario20170219_Block-1_eNe1_Offline.csv','Mario20170219_Block-2_eNe1_Offline.csv','Mario20170219_Block-3_eNe1_Offline.csv'],
								 ['Mario20170219_Block-1_eNe2_Offline.csv','Mario20170219_Block-2_eNe2_Offline.csv','Mario20170219_Block-3_eNe2_Offline.csv']]
		                        
								]
		
		
		#### syncHDF files
		
		# Paths to Where the syncHDF files are:
		path_syncHDF = "D:\\Value Stimulation\\Data\\Mario\\syncHDF\\"
		
		# (some files are repeated in the list so that they can match with the offline sorted spike files they correspond to)
		filenames_sham_syncHDF = [
							        ['Mario20161220_b1_syncHDF.mat'],
	
									['Mario20170106_b1_syncHDF.mat'],
									
									#['Mario20170119_b1_syncHDF_new.mat', #excluded until alignment issue fixed
									#'Mario20170119-2_b1_syncHDF_new.mat'],
									
									#['Mario20170126_b1_syncHDF.mat'],
									
# 									['Mario20170126-2_b1_syncHDF.mat',
# 									'Mario20170126-3_b1_syncHDF.mat'],
									
									['Mario20170204_b1_syncHDF.mat'],
									
									['Mario20170207_b1_syncHDF.mat',
									'Mario20170207_b2_syncHDF.mat',
									'Mario20170207_b3_syncHDF.mat'],
									
									['Mario20170207-2_b1_syncHDF.mat',
									'Mario20170207-2_b2_syncHDF.mat',
									'Mario20170207-2_b3_syncHDF.mat',
									'Mario20170207-2_b4_syncHDF.mat'],
									
# 									['Mario20170214_b1_syncHDF.mat',
# 									'Mario20170214_b2_syncHDF.mat',
# 									'Mario20170214_b3_syncHDF.mat',
# 									'Mario20170214_b4_syncHDF.mat',
# 									'Mario20170214-2_b1_syncHDF.mat',
# 									'Mario20170214-2_b2_syncHDF.mat'],
									
									['Mario20170215_b1_syncHDF.mat',
									'Mario20170215_b2_syncHDF.mat',
									'Mario20170215_b3_syncHDF.mat']
									
									#['Mario20170220_b1_syncHDF.mat',
									#'Mario20170220_b2_syncHDF.mat',
									#'Mario20170220_b3_syncHDF.mat',
									#'Mario20170220_b4_syncHDF.mat']
	
		                        ]
		
		filenames_stim_syncHDF = [
			
									['Mario20161221_b1_syncHDF.mat'],
	
									['Mario20161222_b1_syncHDF.mat'],
									
									['Mario20170108_b1_syncHDF.mat'],
									
# 									['Mario20170125_b1_syncHDF.mat',
# 									'Mario20170125-2_b1_syncHDF.mat',
# 									'Mario20170125-3_b1_syncHDF.mat'],
									
									#['Mario20170130-2_b1_syncHDF.mat'],
									
# 									['Mario20170131_b1_syncHDF.mat',
# 									'Mario20170131-2_b1_syncHDF.mat'],
									
									['Mario20170201_b1_syncHDF.mat'],
									
# 									['Mario20170202_b1_syncHDF.mat',
# 									'Mario20170202-2_b1_syncHDF.mat'],
									
									#['Mario20170208-2_b1_syncHDF.mat',
									#'Mario20170208-2_b2_syncHDF.mat',
									#'Mario20170208-2_b3_syncHDF.mat'],
									
									['Mario20170209_b1_syncHDF.mat',
									'Mario20170209_b2_syncHDF.mat',
									'Mario20170209_b3_syncHDF.mat'],
									
 									['Mario20170216_b1_syncHDF.mat',
 									'Mario20170216_b2_syncHDF.mat',
 									'Mario20170216_b3_syncHDF.mat',
 									'Mario20170216_b4_syncHDF.mat'],
 									
 									['Mario20170219_b1_syncHDF.mat',
 									'Mario20170219_b2_syncHDF.mat',
 									'Mario20170219_b3_syncHDF.mat']
	
								]
		
		#### tdt files
		
		# Paths to Where the tdt files are:
		path_tdt = "D:\\Value Stimulation\\Data\\Mario\\tdt\\"

		filenames_sham_tdt = [
			
			                          ['Mario20161220\\Block-1'],
									  
			                          ['Mario20170106\\Block-1'],
									  
#  									  ['Mario20170119\\Block-1',
# 										'Mario20170119-2\\Block-1'],
									  
#  									  ['Mario20170126-2\\Block-1',
# 										'Mario20170126-3\\Block-1'],
									  
									  ['Mario20170204\\Block-1'],
									  
 									  ['Mario20170207\\Block-1',
										'Mario20170207\\Block-2',
										'Mario20170207\\Block-3'],
 									  
 									  ['Mario20170207-2\\Block-1',
										'Mario20170207-2\\Block-2',
										'Mario20170207-2\\Block-3',
										'Mario20170207-2\\Block-4'],
 									  
 									  ['Mario20170215\\Block-1',
										'Mario20170215\\Block-2',
										'Mario20170215\\Block-3']
 									  
		                        ]
		
		filenames_stim_tdt = [
			
 									  ['Mario20161221\\Block-1'],
 									  
 									  ['Mario20161222\\Block-1'],
 									  
 									  ['Mario20170108\\Block-1'],
 									  
 									  ['Mario20170201\\Block-1'],
 									  
 									  ['Mario20170209\\Block-1',
										'Mario20170209\\Block-2',
										'Mario20170209\\Block-3'],
 									  
 									  ['Mario20170216\\Block-1',
										'Mario20170216\\Block-2',
										'Mario20170216\\Block-3',
										'Mario20170216\\Block-4'],
 									  
 									  ['Mario20170219\\Block-1',
										'Mario20170219\\Block-2',
										'Mario20170219\\Block-3']
 									  
		                        ]
		
		
	#%% Output desired list of files 
	
	if stim_or_sham == 'Sham':                             #Sham Data
		filenames_hdf = filenames_sham_hdf
		filenames_spikes = filenames_sham_spikes
		filenames_syncHDF = filenames_sham_syncHDF
		filenames_tdt = filenames_sham_tdt
	
	elif stim_or_sham == 'Stim':                             #Stim Data
		filenames_hdf = filenames_stim_hdf
		filenames_spikes = filenames_stim_spikes
		filenames_syncHDF = filenames_stim_syncHDF
		filenames_tdt = filenames_stim_tdt
	
	
	#Put relevant info into dictionaries for compact export
	paths = {'hdf path': path_hdf, 'spikes path': path_spikes, 
		   'syncHDF path': path_syncHDF, 'tdt path': path_tdt}
	filenames = {'hdf filenames': filenames_hdf, 'spikes filenames': filenames_spikes, 
			   'syncHDF filenames': filenames_syncHDF, 'tdt filenames': filenames_tdt}
	
	return paths, filenames
