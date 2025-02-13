import numpy as np
import scipy as sp
#from neo import io
import tdt



def create_syncHDF_orig(TDT_tank):
	#filename = 'Luigi20181027'
# 	TDT_tank = TDT_tank + filename
	filename = TDT_tank[-21:-8]
	#TDT_tank = 'C:/Users/ss45436/Box/UC Berkeley/Stress Task/Luigi - Neural Data/'+filename
	#TDT_tank = './' + filename
	#block_num = 1

	# Load TDT files.
	print(TDT_tank)
	block_num = TDT_tank[-1]
	#r = io.TdtIO(dirname = TDT_tank)				# create a reader
	#bl = r.read_block(lazy=True)#,cascade=True)		# read data
	DIOx = tdt.read_block(TDT_tank, store='DIOx')
	print("File read.")

	# Pull out recorded DIO channels.
	DIOx1 = DIOx.streams.DIOx.data[0,:] #first channel indicates when behavior recording is ON (=1)
	DIOx2 = DIOx.streams.DIOx.data[1,:] #second channel is strobe signal that indicates when DIO message is sent from behavior recording
	DIOx3 = DIOx.streams.DIOx.data[2,:] #third channel indicates DIO message type
	DIOx4 = DIOx.streams.DIOx.data[3,:] #fourth channels has row numbers (mod 256) plus other messages

	# Create dictionary to store synchronization data
	hdf_times = dict()
	hdf_times['row_number'] = [] 			# PyTable row number 
	#hdf_times['tdt_timestamp'] = []
	hdf_times['tdt_samplenumber'] = []		# Corresponding TDT sample number
	hdf_times['tdt_dio_samplerate'] = DIOx.streams.DIOx.fs # Sampling frequency of DIO signal recorded by TDT system
	hdf_times['tdt_recording_start'] = []	# TDT sample number when behavior recording begins

	find_recording_start = np.ravel(np.nonzero(DIOx1))[0]
	find_data_rows = np.logical_and(np.ravel(np.equal(DIOx3,13)),np.ravel(np.greater(DIOx2,0))) 	# samples when data corresponds to a row and strobe is on
	find_data_rows_ind = np.ravel(np.nonzero(find_data_rows))	

	rows = DIOx4[find_data_rows_ind]		# row numbers (mod 256)
	
	prev_row = rows[0] 	# placeholder variable for previous row number
	counter = 0 		# counter for number of cycles (i.e. number of times we wrap around from 255 to 0) in hdf row numbers

	for ind in range(1,len(rows)):
		row = rows[ind]
		cycle = (row < prev_row) # row counter has cycled when the current row number is less than the previous
		counter += cycle
		rows[ind] = counter*256 + row
		prev_row = row
		#print float(ind)/length

	# Load data into dictionary
	hdf_times['row_number'] = rows
	hdf_times['tdt_samplenumber'] = find_data_rows_ind
	hdf_times['tdt_recording_start'] = find_recording_start
	#hdf_times['tdt_timestamp'] = times

	# Save syncing data as .mat file
	mat_filename = filename+'_b'+str(block_num)+'_syncHDF_orig.mat'
	#sp.io.savemat('C:/Users/ss45436/Box/UC Berkeley/Stress Task/syncHDF/'+mat_filename,hdf_times)
	sp.io.savemat(mat_filename,hdf_times)
	print(f'{mat_filename} saved to current directory.')

	# Note:
	# If DIO sample num is x, then data sample number is R*(x-1) + 1 where
	# R = data_sample_rate/dio_sample_rate
	return mat_filename


def create_syncHDF_TDTloaded(filename, TDT_tank, bl):
	'''
	Method for when TDT tank has already been loaded in another method - the bl data class (resulting from bl = r.read_block command)
	can be passed directly into this method in order to generate a syncHDF file.

	Inputs:
	- filename: string; name of TDT tank parent folder
	- TDT_tank: string; folder location for TDT tank parent folder
	- bl: class object; data resulting from loading TDT data, i.e. bl = r.read_block 

	'''
	for block_num in range(1,len(bl.segments)+1):
		print("Block %i" % (block_num))
		analogsig = bl.segments[block_num-1].analogsignals

		# Create dictionary to store synchronization data
		hdf_times = dict()
		hdf_times['row_number'] = [] 			# PyTable row number 
		#hdf_times['tdt_timestamp'] = []
		hdf_times['tdt_samplenumber'] = []		# Corresponding TDT sample number
		hdf_times['tdt_dio_samplerate'] = []	# Sampling frequency of DIO signal recorded by TDT system
		hdf_times['tdt_recording_start'] = []	# TDT sample number when behavior recording begins

		row_number = []
		tdt_timestamp = []
		tdt_samplenumber = []

		# Pull out recorded DIO channels.
		for sig in analogsig:
			if (sig.name == "b'DIOx' 1"):					# first channel indicates when behavior recording is ON (=1)
				DIOx1 = np.array(sig)
			if (sig.name == "b'DIOx' 2"):					# second channel is strobe signal that indicates when DIO message is sent from behavior recording
				DIOx2 = np.array(sig)
			if (sig.name == "b'DIOx' 3"): 					# third channel indicates DIO message type
				DIOx3 = np.array(sig)
				hdf_times['tdt_dio_samplerate'] = sig.sampling_rate
			if (sig.name == "b'DIOx' 4"): 					# fourth channels has row numbers (mod 256) plus other messages
				DIOx4 = np.array(sig)
				#DIOx4 = [sig[ind].item() for ind in range(0,sig.size)]
				##DIOx4_times = sig.times


		find_recording_start = np.ravel(np.nonzero(DIOx1))[0]
		find_data_rows = np.logical_and(np.ravel(np.equal(DIOx3,13)),np.ravel(np.greater(DIOx2,0))) 	# samples when data corresponds to a row and strobe is on
		find_data_rows_ind = np.ravel(np.nonzero(find_data_rows))	

		rows = DIOx4[find_data_rows_ind]		# row numbers (mod 256)
		
		prev_row = rows[0] 	# placeholder variable for previous row number
		counter = 0 		# counter for number of cycles (i.e. number of times we wrap around from 255 to 0) in hdf row numbers

		for ind in range(1,len(rows)):
			row = rows[ind]
			cycle = (row < prev_row) # row counter has cycled when the current row number is less than the previous
			counter += cycle
			rows[ind] = counter*256 + row
			prev_row = row
			#print float(ind)/length

		# Load data into dictionary
		hdf_times['row_number'] = rows
		hdf_times['tdt_samplenumber'] = find_data_rows_ind
		hdf_times['tdt_recording_start'] = find_recording_start
		#hdf_times['tdt_timestamp'] = times

		# Save syncing data as .mat file
		mat_filename = filename+'_b'+str(block_num)+'_syncHDF.mat'
		sp.io.savemat('C:/Users/ss45436/Box/UC Berkeley/Stress Task/syncHDF/'+mat_filename,hdf_times)
		#sp.io.savemat('./'+mat_filename,hdf_times)

	# Note:
	# If DIO sample num is x, then data sample number is R*(x-1) + 1 where
	# R = data_sample_rate/dio_sample_rate
	return