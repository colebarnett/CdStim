# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:11:52 2024

@author: coleb
"""
import numpy as np




def GetGoodChans(lfp_sess_name,area):
	'''
	

	Parameters
	----------
	lfp_sess_name : str. Session name. E.g. "LFP_hc_Mario20161220"
	area : 'Cd', 'ACC', 'M1', or 'PMd'. Select set of channels corresponding to specified brain area.

	Returns
	-------
	chs : 1D array. Indexes of uncontaminated ("good") channels for specified brain area
	num_chs : int. =len(chs)
	all_locs : array of lists. Each element is a list containing stereotactic coors for each channel.
		The index of the element in all_locs corresponds to channel number. e.g. locs[0] is the coors for ch 0.
		This array includes locations for all channels, not just good channels. That is, len(all_locs) != len(chs).
		The 1st coordinate is the AP value, the 2nd is the ML value, and the 3rd is the depth from the edge of the cortex.

	'''
	
	
	
	#### Luigi
	if 'igi' in lfp_sess_name:
		all_Cd_channels = set(range(0,64))
		all_ACC_channels = set(range(64,128))
		all_M1_channels = set([])
		all_PMd_channels = set([])
		all_WM_channels = set([])
		
		#all_locs is an array where each element is a list containing stereotactic coors for each channel.
		#The index of the element in all_locs corresponds to channel number. e.g. all_locs[0] is the coors for ch 0.
		all_locs = np.full((128,3),[np.nan,np.nan,np.nan])
		#for luigi, electrode locations vary by session
	
	# Sham
	if 'Luigi20170822' in lfp_sess_name:
		bad_channels = np.array(list(set([15,17,19,20,21,25,26,27,28,29,30,32,34,48,50,62]).union(range(64,128))))
# 		bad_channels = np.array(list(set([15,17,32,34,48,50,62]).union(range(64,96))))
		existsLFP2 = False
		all_locs[:32,:] = [24,10,16] #probe 1
		all_locs[32:64,:] = [27,10,16] #probe 2
	
	elif 'Luigi20170831' in lfp_sess_name:
		bad_channels_1 = np.array([0,2,15,30,47,48,49,50,51,52,53,61,71,88,90,92,94])
		bad_channels_2 = np.array(list(set([0,15,16,17,19,21]).union(range(23,64)))) + 96 #do LFP2 channels separately
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [24,10,16] #probe 1
		all_locs[32:64,:] = [27,10,16] #probe 2
		all_locs[64:96,:] = [26,6,7] #probe 3
		all_locs[96:,:] = [29,6,7] #probe 4
	
	elif 'Luigi20170902' in lfp_sess_name:
		bad_channels = np.array(list(set([0,2,15,22,30,47,48,49,50,51,52,53,54,61,71,78,81]).union(range(96,128))))
		existsLFP2 = True  
		all_locs[:32,:] = [24,10,17] #probe 1
		all_locs[32:64,:] = [27,10,17] #probe 2
		all_locs[64:96,:] = [26,6,7] #probe 3
		all_locs[96:,:] = [29,6,7] #probe 4
	
	elif 'Luigi20170907' in lfp_sess_name:
		bad_channels_1 = np.array([2,15,30,47,48,49,50,51,52,53,54,61,71,79,81,88,90,91,92,93,94,95])
		bad_channels_2 = np.array(list(set([0,15,16,18,24,26]).union(range(28,64)))) + 96 #do LFP2 channels separately
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [24,10,17] #probe 1
		all_locs[32:64,:] = [27,10,17] #probe 2
		all_locs[64:96,:] = [26,6,7] #probe 3
		all_locs[96:,:] = [29,6,7] #probe 4
	
	elif 'Luigi20170929' in lfp_sess_name:
		bad_channels_1 = np.array([0,2,15,30,47,48,49,50,51,52,61,71,79])
		bad_channels_2 = np.array(list(set([0,15,18,20]).union(range(32,64)))) + 96
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [25,10,16] #probe 1
		all_locs[32:64,:] = [28,10,16] #probe 2
		all_locs[64:96,:] = [27,6,7] #probe 3
		all_locs[96:,:] = [30,6,7] #probe 4
	
	elif 'Luigi20171003' in lfp_sess_name:
		bad_channels_1 = np.array([0,2,15,30,47,48,49,50,51,52,54,57,71,79])
		bad_channels_2 = np.array(list(set([0,15,30]).union(range(32,64)))) + 96
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [25,10,15] #probe 1
		all_locs[32:64,:] = [28,10,15] #probe 2
		all_locs[64:96,:] = [27,6,7] #probe 3
		all_locs[96:,:] = [30,6,8] #probe 4
	
	elif 'Luigi20171015' in lfp_sess_name:
		bad_channels_1 = np.array([0,2,15,30,47,48,49,50,51,52,53,54,55,57,61,71])
		bad_channels_2 = np.array(list(set([0,12,15,16,18,20,30]).union(range(32,64)))) + 96
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [25,10,10] #probe 1
		all_locs[32:64,:] = [28,10,10] #probe 2
		all_locs[64:96,:] = [27,6,7] #probe 3
		all_locs[96:,:] = [30,6,7] #probe 4
	
	elif 'Luigi20171019' in lfp_sess_name:
		bad_channels_1 = np.array([0,2,15,30,47,48,49,50,51,52,54,61,71])
		bad_channels_2 = np.array(list(set([0,12,15,18,20,30]).union(range(32,64)))) + 96
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [25,10,10] #probe 1
		all_locs[32:64,:] = [28,10,10] #probe 2
		all_locs[64:96,:] = [27,6,7] #probe 3
		all_locs[96:,:] = [30,6,7] #probe 4
	 
	elif 'Luigi20171024' in lfp_sess_name:
		bad_channels_1 = np.array([0,15,16,18,30,32,34,47,48,49,50,51,52,54,57,61,71,79])
		bad_channels_2 = np.array(list(set([2,14,15,30]).union(range(32,64)))) + 96
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [24,11,11] #probe 1
		all_locs[32:64,:] = [27,11,11] #probe 2
		all_locs[64:96,:] = [29,5,6] #probe 3
		all_locs[96:,:] = [32,5,6] #probe 4
	
	elif 'Luigi20171028' in lfp_sess_name:
		bad_channels_1 = np.array([0,15,16,18,20,30,32,34,47,48,49,50,51,52,53,54,57,61,71,81,83,85,87,88,89,90,91,92,93,94,95])
		bad_channels_2 = np.array(list(set([0,2,15,30]).union(range(32,64)))) + 96
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [24,11,12] #probe 1
		all_locs[32:64,:] = [27,11,12] #probe 2
		all_locs[64:96,:] = [29,5,6.6] #probe 3
		all_locs[96:,:] = [32,5,6.6] #probe 4
	
	# Stim
	elif 'igi20170915-2' in lfp_sess_name:
		bad_channels_1 = np.array([2,15,30,47,48,49,50,51,52,54,57,61,65,67,69,71,79,88,90,92,94,95])
		bad_channels_2 = np.array(list(set([0,8,10,12]).union(range(14,64)))) + 96
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [24,10,16] #probe 1
		all_locs[32:64,:] = [27,10,16] #probe 2
		all_locs[64:96,:] = [26,6,7] #probe 3
		all_locs[96:,:] = [29,6,7] #probe 4
	
	elif 'Luigi20170927' in lfp_sess_name:
		bad_channels_1 = np.array([2,15,30,47,48,49,50,51,52,54,57,61,65,67,69,71,79,88,90,92,94,95])
		bad_channels_2 = np.array(list(set([0,8,10,12,14,16,17,18,19,20,21,22,23,24,25,27]).union(range(29,64)))) + 96
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [25,10,16] #probe 1
		all_locs[32:64,:] = [28,10,16] #probe 2
		all_locs[64:96,:] = [27,6,7] #probe 3
		all_locs[96:,:] = [30,6,7] #probe 4
	
	elif 'Luigi20171001' in lfp_sess_name:
		bad_channels_1 = np.array([2,15,30,34,47,48,49,50,51,52,57,61,71,83,88,90,92,94])
		bad_channels_2 = np.array(list(set([0,14,15,16,17,18,19,20,21,22,23,24,25,26,27]).union(range(29,64)))) + 96
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [25,10,15] #probe 1
		all_locs[32:64,:] = [28,10,15] #probe 2
		all_locs[64:96,:] = [27,6,7] #probe 3
		all_locs[96:,:] = [30,6,8] #probe 4
	
	elif 'Luigi20171005' in lfp_sess_name:
		bad_channels_1 = np.array([0,2,15,30,33,34,47,48,49,50,51,52,54,57,71,81,83,89,90,92,94])
		bad_channels_2 = np.array(list(set([0,15,16,18,20,26,30]).union(range(32,64)))) + 96
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [25,10,10] #probe 1
		all_locs[32:64,:] = [28,10,10] #probe 2
		all_locs[64:96,:] = [27,6,7] #probe 3
		all_locs[96:,:] = [30,6,7] #probe 4
	
	elif 'Luigi20171017' in lfp_sess_name:
		bad_channels_1 = np.array([0,2,15,30,32,33,34,35,36,47,48,50,52,54,57,61,65,67,69,71,82])
		bad_channels_2 = np.array(list(set([0,15,16,18,30]).union(range(32,64)))) + 96
		bad_channels = np.array(list(set(bad_channels_1).union(set(bad_channels_2)))) #combine LFP1 and LFP2 channels
		existsLFP2 = True
		all_locs[:32,:] = [25,10,10] #probe 1
		all_locs[32:64,:] = [28,10,10] #probe 2
		all_locs[64:96,:] = [27,6,7] #probe 3
		all_locs[96:,:] = [30,6,7] #probe 4
		
	
	#### Mario
	if 'rio' in lfp_sess_name:
		all_Cd_channels = set(np.array([1,3,4,17,18,20,40,41,54,56,57,63,64,72,75,81,83,88,89,96,100,112,114,126,130,140,143,146,156,157,159])-1) #from channel_legend.xlsx
		all_ACC_channels = set(np.array([5,6,19,22,30,39,42,43,55,58,59,69,74,77,85,90,91,102,105,121,128])-1) # -1 is to account for idxing from 0 in python
		all_M1_channels = set(np.array([65,67,68,81,82,83,97,98,99,100,101,102,
							   103,104,105,106,108,109,110,111,113,115,116,117,
							   118,119,120,121,122,124,125,126,127,128,136,137,
							   138,139,141,142,144,148,152,153,154,155,156,158])-1)
		all_PMd_channels = set(np.array([6,19,33,35,36,37,38,40,42,43,46,48,49,51,
								52,53,54,56,62,69,70,72,73,74,75,80,84,85,86,
								88,89,91,96])-1)
		all_WM_channels = set(np.array([29,41,57,63,64,112,114,140,143,157,159])-1)
		
		all_locs = _GetMarioLocs()
		#for mario, electrode locations are fixed across all sessions
		
	# Sham
	if 'Mario20161220' in lfp_sess_name:
# 		bad_channels = np.array([0,1,3,4,7,8,24,25,27,28,30,31,41,43,44,48,50,52,65,67,75,79,83,89,92,94,99,107,108,110,121,124,126])
		bad_channels = np.array([0,1,3,4,5,6,7,8,24,25,27,28,30,31,41,48,50,79,92,94,124,126,128,130,143,144,153,158])
		existsLFP2 = True
	
	elif 'Mario20170106' in lfp_sess_name:
# 		bad_channels = np.array([9,10,11,43,44,47,48,49,50,52,57,65,67,83,89,92,101,107,108,110,121,122,124,126])
		bad_channels = np.array([9,47,50,57,92,99,128,130,144,158])
		existsLFP2 = True	
		
# 	elif 'rio20170126-2' in lfp_sess_name: 
# 		bad_channels = np.array([])
# 		
# 		existsLFP2 = True	
	
	elif 'Mario20170204' in lfp_sess_name:
# 		bad_channels = np.array([1,9,10,11,16,30,48,65,67,79,83,107,108,110,111,120,121,123,124,126])
		bad_channels = np.array([9,16,30,49,53,55,67,79,99,111,128,130,144,145])
		existsLFP2 = True	
	
	elif 'Mario20170207' in lfp_sess_name:
# 		bad_channels = np.array([1,2,9,10,11,18,32,48,49,55,56,62,65,67,79,83,86,96,98,99,107,108,109,110,111,119,120,121,122,123,124,126])
		bad_channels = np.array([9,11,32,48,62,79,86,96,98,111,128,130,144,156,158])
		existsLFP2 = True	
		
	elif 'rio20170207-2' in lfp_sess_name:
# 		bad_channels = np.array([1,9,11,32,33,48,62,65,67,79,83,86,96,98,108,109,110,111,124,126])
		bad_channels = np.array([9,32,48,62,79,86,96,98,111,128,130,144,156,158])
		existsLFP2 = True	

	elif 'Mario20170215' in lfp_sess_name:
# 		bad_channels = np.array([1,9,11,30,32,34,47,48,55,64,65,67,79,83,94,96,107,108,109,110,123,124,126])
		bad_channels = np.array([9,11,30,32,34,83,94,96,128,130,143,144,153,157])
		existsLFP2 = True

	# Stim
	elif 'Mario20161221' in lfp_sess_name:
# 		bad_channels = np.array(list(set([1,7,8,9,11,18,25,32,33,34,36,40,48,56,58,59,60,62,63,65,66,67,71,79,83,92,94]).union(range(96,113)).union(range(120,127))))
		bad_channels = np.array([9,11,18,25,29,32,34,36,37,38,39,40,56,58,60,62,63,79,92,94,111,112,128,130,143,144,153])		
		existsLFP2 = True

	elif 'Mario20161222' in lfp_sess_name:
# 		bad_channels = np.array(list(set([1,9,11,15,16,30,32,34,40,47,48,52,63,64,65,66,67,71,79,83,94]).union(range(96,111)).union(range(120,127))))
		bad_channels = np.array([9,11,16,30,32,34,40,47,63,128,130,143,144,148,149])
		existsLFP2 = True

	elif 'Mario20170108' in lfp_sess_name:
# 		bad_channels = np.array([0,1,2,9,10,11,15,30,40,48,52,60,62,63,65,67,79,80,81,83,94,108,110,124,126])
		bad_channels = np.array([0,2,9,11,15,30,39,40,60,62,63,80,94,124,126,128,130,143,144,146,153,157])
		existsLFP2 = True
		
	elif 'Mario20170201' in lfp_sess_name:
# 		bad_channels = np.array([1,9,11,15,16,18,29,40,48,62,63,65,67,79,81,83,110,122,123,124,126])
		bad_channels = np.array([9,11,18,29,40,62,63,128,130,144,145])
		existsLFP2 = True

	elif 'Mario20170209' in lfp_sess_name:
# 		bad_channels = np.array([1,9,11,30,40,48,55,63,64,65,67,79,83,86,96,98,108,110,111,121,122,123,124,126])
		bad_channels = np.array([9,11,30,40,63,64,96,98,111,126,128,130,144])
		existsLFP2 = True		
		
	elif 'Mario20170216' in lfp_sess_name:
# 		bad_channels = np.array([9,17,30,40,48,62,79,82,83,86,89,110,123,126])
		bad_channels = np.array([9,11,17,30,40,48,62,79,82,86,89,126,128,130,143,144])
		existsLFP2 = True		
		
	elif 'Mario20170219' in lfp_sess_name:
# 		bad_channels = np.array([0,1,2,9,11,16,30,40,47,48,55,61,62,63,64,65,66,67,77,78,79,83,86,92,94,99,104,108,109,110,111,114,115,116,117,118,120,121,122,123,124,125,126])
		bad_channels = np.array([0,9,11,16,30,4,47,63,92,94,111,114,115,116,118,121,125,128,130,143,144,158])
		existsLFP2 = True		
	

	#### Finalize good ch list
	try: 
		good_Cd_channels = np.array(list(all_Cd_channels - set(bad_channels))) 
		good_ACC_channels = np.array(list(all_ACC_channels - set(bad_channels)))
		good_M1_channels = np.array(list(all_M1_channels - set(bad_channels)))
		good_PMd_channels = np.array(list(all_PMd_channels - set(bad_channels)))
		good_WM_channels = np.array(list(all_WM_channels - set(bad_channels)))
	except: 
		raise ValueError(f'bad_channels does not exist for {lfp_sess_name}')
		
			
	#get good chs for area of interest
	if area=='Cd':
		chs = good_Cd_channels
		num_chs= len(chs)
	elif area=='ACC':
		chs = good_ACC_channels
		num_chs= len(chs)
	elif area=='M1':
		chs = good_M1_channels
		num_chs= len(chs)
	elif area=='PMd':
		chs = good_PMd_channels
		num_chs= len(chs)	
	elif area=='WM':
		chs = good_WM_channels
		num_chs= len(chs)
			
		
	return chs, num_chs, all_locs


def _GetMarioLocs():
	'''
	Locations of each of Marios electrodes. From Mario_Channel_legend.xlsx.

	'''
	
	#locs is an array where each element is a list containing stereotactic coors for each channel.
	#The index of the element in locs corresponds to channel number. e.g. locs[0] is the coors for ch 0.
	locs = np.array([
		[26.65, 7, 14],
		[28.25, 11, 2],
		[26.65, 4, 15],
		[28.25, 5.5, 14],
		[26.65, 3, 5],
		[28.25, 3, 2.5],
		[26.65, 1, 2],
		[28.25, 2, 2.5],
		[31.35, 8, 17],
		[29.85, 7, 19],
		[31.35, 4, 22],
		[29.85, 3.5, 20.5],
		[31.35, 1.5, 21],
		[29.85, 4.5, 6],
		[31.35, 2.5, 2],
		[29.85, 2.25, 3.5],
		[26.65, 5.5, 14.5],
		[28.25, 7, 13],
		[26.65, 5.5, 2],
		[28.25, 4, 15],
		[26.65, 2.5, 3],
		[28.25, 3, 6],
		[28.25, 13.5, 2],
		[28.25, 1, 2],
		[31.35, 6.5, 15],
		[29.85, 5.5, 18],
		[31.35, 2.5, 22.5],
		[29.85, 2, 22],
		[31.35, 3.5, 2],
		[29.85, 3, 6],
		[31.35, 1, 2],
		[29.85, 1.25, 2],
		[21.85, 14.5, 1.5],
		[21.85, 2.5, 2],
		[21.85, 11.5, 1.5],
		[23.45, 14.5, 1.5],
		[21.85, 8.5, 1.5],
		[23.45, 11.5, 1.5],
		[21.85, 4.5, 6],
		[23.45, 9, 1.5],
		[25.05, 8, 3.5],
		[23.45, 5.5, 2],
		[25.05, 5.5, 1.5],
		[23.45, 2.5, 2.5],
		[25.05, 2.5, 2.5],
		[25.05, 14.5, 2],
		[26.65, 13, 2],
		[25.05, 11, 2],
		[21.85, 13, 1.5],
		[21.85, 1, 1.5],
		[21.85, 10, 1.5],
		[23.45, 13, 1.5],
		[21.85, 7, 1],
		[23.45, 10, 1.5],
		[21.85, 3, 5],
		[23.45, 7, 1.5],
		[25.05, 6.5, 4],
		[23.45, 3, 5.5],
		[25.05, 3, 6],
		[23.45, 1, 2.5],
		[25.05, 1, 2.5],
		[25.05, 13, 2],
		[26.65, 10.5, 4],
		[25.05, 10, 2],
		[17.05, 11.5, 1.5],
		[18.65, 17.5, 1.5],
		[17.05, 8, 1.5],
		[18.65, 14.5, 1.5],
		[17.05, 5.5, 1.5],
		[18.65, 11.5, 1.5],
		[17.05, 2.5, 1.5],
		[18.65, 8.5, 2],
		[20.25, 10, 2],
		[18.65, 5.5, 2],
		[20.25, 7, 2],
		[18.65, 2.5, 2],
		[20.25, 3, 6],
		[20.25, 15.5, 2],
		[20.25, 1, 1.5],
		[20.25, 12.5, 2],
		[17.05, 10, 1.5],
		[18.65, 16, 1.5],
		[17.05, 7, 1.5],
		[18.65, 13, 1.5],
		[17.05, 4, 1.5],
		[18.65, 10, 2],
		[17.05, 1, 1.5],
		[18.65, 7, 2],
		[20.25, 8.5, 2],
		[18.65, 3, 6],
		[20.25, 5.5, 2],
		[18.65, 1, 1],
		[20.25, 2, 2],
		[20.25, 14, 2],
		[21.85, 15, 1.5],
		[20.25, 11.5, 1.5],
		[12.25, 7, 2],
		[13.85, 12, 1],
		[12.25, 4, 1.5],
		[13.85, 8, 1.5],
		[13.85, 17.5, 1.5],
		[13.85, 5.5, 2],
		[13.85, 15, 1.25],
		[13.85, 2.5, 2],
		[15.45, 5, 2],
		[15.45, 18, 1.5],
		[15.45, 2.5, 1.5],
		[15.45, 15, 1.5],
		[17.05, 18, 1.5],
		[15.45, 11.5, 1.5],
		[17.05, 14.5, 1.5],
		[15.45, 8, 2],
		[12.25, 5.5, 1.5],
		[13.85, 9.5, 1.5],
		[13.85, 19, 1.5],
		[13.85, 7, 1.5],
		[13.85, 16.5, 1.5],
		[13.85, 4, 2],
		[13.85, 13.5, 1.5],
		[15.45, 19, 1],
		[15.45, 4, 1.5],
		[15.45, 16.5, 1],
		[15.45, 1, 1.5],
		[15.45, 13.5, 1.5],
		[17.05, 16, 1.5],
		[15.45, 9, 2],
		[17.05, 13, 1.5],
		[15.45, 6.5, 2],
		[0,0,0], #N/A
		[9.85, 11.5, 1.5],
		[0,0,0], #N/A
		[9.85, 8.5, 1.5],
		[9.85, 17.5, 2],
		[10.65, 20.5, 1.5],
		[9.85, 14.5, 2],
		[10.65, 17.5, 2],
		[12.25, 19, 1.5],
		[10.65, 14.5, 2],
		[12.25, 16.5, 1.5],
		[10.65, 11.5, 2],
		[12.25, 13, 1.25],
		[10.65, 8.5, 1.5],
		[12.25, 9.5, 2],
		[10.65, 5.5, 1.5],
		[0,0,0], #N/A
		[9.85, 10, 1.5],
		[9.85, 19, 2],
		[9.85, 7, 2],
		[9.85, 16, 2],
		[10.65, 19, 2],
		[9.85, 13, 2],
		[10.65, 16, 2],
		[12.25, 17.5, 1.5],
		[10.65, 13, 2],
		[12.25, 15, 1.25],
		[10.65, 10, 2],
		[12.25, 11, 2],
		[10.65, 7, 1.5],
		[12.25, 8.5, 2],
		[12.25, 20.5, 1.5],
		])
		
	return locs
	