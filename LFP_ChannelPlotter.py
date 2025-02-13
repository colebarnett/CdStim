# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:07:22 2025

@author: coleb

Plot processed LFP data from the caudate stimulation experiment (Santacruz et al. 2017) 

Plot time-aligned LFP snippets, spectrograms, and PSDS by trial and/or channel.
"""
import pandas as pd
from SortedFiles import GetFileList
import os
from scipy.stats import pearsonr
from matplotlib import cm
from GoodChans import GetGoodChans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class ChannelPlotter():
	'''
	Class for plotting LFP voltage traces, spectrograms, or power spectra from data saved out by LFP_CdStim_Processing 
	Is able to plot all channels, all channels within a brain area, or specific channels or trials.
	
	Parameters
	----------
		- subject: 'Mario' or 'Luigi'
		- epoch: 'hc', 'rp', or 'mp'
				Choose which part of a trial to focus on. 
				hc=hold_center, rp=reward period, mp=movement period
		- data_type: 'PSD', 'Sxx', or 'LFP'
				Choose types of plots.
				PSD=power vs freq, Sxx=power vs freq vs time (spectrogram), LFP=voltage vs time
		- stop_or_pause: 'stop', 'pause', or 'continue'
				Choose whether to advance the code after data for one session has been plotted
				stop=only plot the first session, pause=pause plotting after each session, continue=plot all sessions without stopping 
		- file_names: list of str or None
				Can manually pass through files to plot
				If this is left empty or is None, then all files will be plotted by default
		'''
		
	def __init__(self,subject,epoch,data_type,stop_or_pause,file_names=None):
		
		self.subject = subject
		self.epoch = epoch
		self.data_type = data_type
		self.stop_or_pause = stop_or_pause
		
		if self.data_type == 'PSD':
			self.data_type_str = 'PSDs'
			
		if self.data_type == 'Sxx':
			self.data_type_str = 'Spectrograms'
			
		if self.data_type == 'LFP':
			self.data_type_str = 'LFP_snippets' 
			
		if file_names:	
			self.file_names = file_names
		else:
			self.file_names = self._GetFileNames()

		if self.subject=='Mario':
			self.scale_bar_voltage = 2e-4 #V
		if self.subject=='Luigi':
			self.scale_bar_voltage = 1e-4 #V
		self.scale_bar_voltage_label = f"{self.scale_bar_voltage*1000}\nmV"
		self.scale_bar_power = 1e-9 #V**2
		self.scale_bar_power_label = f'{self.scale_bar_power}\nV^2'
		self.scale_bar_time = 0.1 #s
		self.scale_bar_time_label = f'{self.scale_bar_time} s'
	
		
	#%% Loading methods and scale bar methods		
	def _GetFileNames(self):
		
		file_names =[]
		for cond_counter,stim_or_sham in enumerate(['Sham','Stim']):
			paths, filenames = GetFileList(self.subject,stim_or_sham)
			num_sessions = len(filenames['hdf filenames'])
			for session in range(0,num_sessions):
				tdt_files = [paths['tdt path'] + filename for filename in filenames['tdt filenames'][session]]
				filestr = tdt_files[0][-21:-8]
				
				# to allow flexibility of whether data is located on drive D: or E:
				if os.path.exists(f"D:\\Value Stimulation\\Data\\{self.subject}\\LFP\\{self.data_type_str}_{self.epoch}_{filestr}.pkl"):
					file_names.append(f"D:\\Value Stimulation\\Data\\{self.subject}\\LFP\\{self.data_type_str}_{self.epoch}_{filestr}.pkl")
				elif os.path.exists(f"E:\\Value Stimulation\\Data\\{self.subject}\\LFP\\{self.data_type_str}_{self.epoch}_{filestr}.pkl"):
					file_names.append(f"E:\\Value Stimulation\\Data\\{self.subject}\\LFP\\{self.data_type_str}_{self.epoch}_{filestr}.pkl")
		
		if len(file_names)==0:
			raise ValueError('Hard drive not found OR wrong subject.')
					
		return file_names
				

			
	def _LoadData(self,file_name):	
		
		print('-'*10 + '\nLoading data')
		with open(file_name,'rb') as f:
			file = pd.read_pickle(f)
		print(f'\t{file_name} loaded')	
		
		return file
	
	def _ScaleBar_Power(self,ax,t):
		
		self._PowerBar(ax,t)
		self._PowerLabel(ax,t)
		self._TimeBar(ax,t)
		self._TimeLabel_Power(ax,t)
		
	def _ScaleBar_Voltage(self,ax,t):
		
		self._VoltageBar(ax,t)
		self._VoltageLabel(ax,t)
		self._TimeBar(ax,t)
		self._TimeLabel_Voltage(ax,t)

	def _VoltageBar(self,ax,t):

		a = ax.vlines(t[-1]+0.05,0,self.scale_bar_voltage,color='r')
		return a
	
	def _VoltageLabel(self,ax,t):

		a = ax.text(t[-1]+0.1,self.scale_bar_voltage,
		 self.scale_bar_voltage_label,fontsize='small',color='r')
		return a
	
	def _PowerBar(self,ax,t):

		a = ax.vlines(t[-1]+0.05,0,self.scale_bar_power,color='r')
		return a
	
	def _PowerLabel(self,ax,t):

		a = ax.text(t[-1]+0.085,self.scale_bar_power,
		 self.scale_bar_power_label,fontsize='small',color='r')
		return a
	
	def _TimeBar(self,ax,t):

		a = ax.hlines(0,t[-1]-0.05,t[-1]+self.scale_bar_time-0.05,color='r')
		return a
	
	def _TimeLabel_Power(self,ax,t):

		a = ax.text(t[-1]-0.05,-1*self.scale_bar_power,
		 self.scale_bar_time_label,fontsize='small',color='r')
		return a		

	def _TimeLabel_Voltage(self,ax,t):

		a = ax.text(t[-1],-self.scale_bar_voltage,
		 self.scale_bar_time_label,fontsize='small',color='r')
		return a	
		
		
	#%% Plotting methods
	def ChCorrMatrix(self,area):
		'''
		Plots a correlation matrix of all channels averaged over all trials for 
		chosen epoch for a specified brain area.

		Parameters
		----------
		area : 'Cd' or 'ACC'
		'''
		
		print(f'\nPlotting ch corr matrix for {area}\n')
		
		if self.subject != 'Luigi':
			print('This method is only capable of plotting Luigi data so far')
			return
		
		for file_name in self.file_names:
			
			file = self._LoadData(file_name)
			
			if self.data_type == 'LFP':
				
				#set good channels
				chs, num_chs, _ = GetGoodChans(file['name'],area)
				data = file['LFP']
				
				corrs = np.full((64,64),np.nan)
				
				for ch1 in chs:
					
					x = np.nanmean(np.squeeze(data[ch1,:,:]),axis=0) #get lfp for channel avgd across trials
					
					for ch2 in chs:

						y = np.nanmean(np.squeeze(data[ch2,:,:]),axis=0) #get lfp for channel avgd across trials
						
						if area=='Cd':
							corrs[ch1,ch2], pval = pearsonr(x,y)
						else:
							corrs[ch1-64,ch2-64], pval = pearsonr(x,y) #-64 is to convert from ACC ch# to matrix index
				
				if area=='Cd':
					X,Y = np.meshgrid(np.arange(64),np.arange(64))
				else:
					X,Y = np.meshgrid(np.arange(64,128),np.arange(64,128))
				fig, ax = plt.subplots()
				surf = ax.pcolor(X,Y,corrs,cmap=cm.viridis,vmin=0.3,vmax=1.0)		
				ax.set_xlabel('Channel')
				ax.set_ylabel('Channel')
				ax.set_title(file['name'] + '\nLFP Channel Correlation matrix')
				cbar = fig.colorbar(surf)
				cbar.ax.set_ylabel('Pearson Correlation')
				ax.minorticks_on()
				plt.show()
				
			if self.data_type == 'PSD':
				
				#set good channels
				chs, num_chs = GetGoodChans(file['name'],area)
				data = file['psds']
				
				corrs = np.full((64,64),np.nan)
				
				
				for ch1 in chs:
					
					x = np.nanmean(np.squeeze(data[ch1,:,:]),axis=0) #get psds for channel avgd across trials
					
					for ch2 in chs:

						y = np.nanmean(np.squeeze(data[ch2,:,:]),axis=0) #get psds for channel avgd across trials
						
						if area=='Cd':
							corrs[ch1,ch2], pval = pearsonr(x,y)
						else:
							corrs[ch1-64,ch2-64], pval = pearsonr(x,y)
						
					#print(f'Channel {ch1+1}/{num_chs} done.',end='\r')
						
				
				if area=='Cd':
					X,Y = np.meshgrid(np.arange(64),np.arange(64))
				else:
					X,Y = np.meshgrid(np.arange(64,128),np.arange(64,128))
				fig, ax = plt.subplots()
				surf = ax.pcolor(X,Y,corrs,cmap=cm.viridis,vmin=0.3,vmax=1.0)		
				ax.set_xlabel('Channel')
				ax.set_ylabel('Channel')
				ax.set_title(file['name'] + '\npsds Channel Correlation matrix')
				cbar = fig.colorbar(surf)
				cbar.ax.set_ylabel('Pearson Correlation')
				ax.minorticks_on()
				plt.show()

			if self.stop_or_pause == 'stop':
				return #to stop after first session
			if self.stop_or_pause == 'pause':
				input('Press enter to continue') #to pause after each session
			
		
	def PlotChs(self):
		'''
		Plots all channels. 
		Plots are averaged over all trials.
		'''
		
		print('\nPlotting all chs\n')
		
		for file_name in self.file_names:
			
			file = self._LoadData(file_name)
			
			if self.data_type == 'LFP':
				
				fig,ax=plt.subplots()
				ax.set_title(file['name'])
				
				fig,axs=plt.subplots(4,1)
				row=0
				
				data = file['LFP']
				t = file['t']
				pwr = data**2 
				
				chs = range(160)
				
				for ch in chs:
					
					try:
						avg= np.nanmean(pwr[ch,:,:],axis=0) #avg power over trials
						sem= np.nanstd(pwr[ch,:,:],axis=0) / np.sqrt(np.size(pwr[ch,:,:],axis=0))
						axs[row].plot(t,avg,color='black',linewidth=1)
						axs[row].fill_between(t, avg-sem,avg+sem, color='black',alpha=0.4)
						axs[row].vlines(0,np.min(avg),np.max(avg),'red') #t=0 bar
						axs[row].text(-0.7,np.mean(avg),f'Ch {ch}',fontsize='small')
						self._ScaleBar_Power(axs[row],t)
						
						if row==3:
							axs[row].axis("off")
			# 				axs[row].spines[['left', 'right', 'top']].set_visible(False)
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/file['fs'],11),2))
							fig.tight_layout()
							
							fig,axs=plt.subplots(4,1)
							row=0
						else:
							axs[row].axis('off')
							row+=1
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
				plt.show()
		
		
			if self.data_type == 'Sxx':
				
				fig,ax=plt.subplots()
				ax.set_title(file['name'])
				
				fig,axs=plt.subplots(3,4)
				row=0
				col=0
				
				data = file['Sxx']
				data = np.array(data)
				
				chs = range(160)
				
				
				for ch in chs:
					
					try:
						avg= np.nanmean(data[ch,:,:,:],axis=0) #avg over trials
						c = axs[row,col].pcolormesh(file['t'], file['f'], avg, cmap='plasma')
						axs[row,col].vlines(0,np.min(file['f']),np.max(file['f']),'black',linestyle='--') 
						axs[row,col].set_title(f'Ch {ch}',fontsize='small')
						
						if row==2 and col==3:
							axs[row,col].axis("off")
			# 				axs[row].spines[['left', 'right', 'top']].set_visible(False)
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/file['fs'],11),2))
							fig.tight_layout()
							fig,axs=plt.subplots(3,4)
							row=0
							col=0
						else:
							if not (row==0 and col==0):
								axs[row,col].axis('off')
							if col<3:
								col+=1
							else:
								row+=1
								col=0
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
						
				plt.show()
			
				
			if self.data_type == 'PSD':
				
				fig,ax=plt.subplots()
				ax.set_title(file['name'])
				
				fig,axs=plt.subplots(8,1)
				row=0
				
				data = file['psd']
				
				chs = range(160)
				
				for ch in chs:
					
					try:
						avg= np.nanmean(data[ch,:,:],axis=0) #avg over trials
						sem= np.nanstd(data[ch,:,:],axis=0) / np.sqrt(np.size(data[ch,:,:],axis=0))
						axs[row].plot(file['f'],avg,color='black')
						axs[row].fill_between(file['f'], avg-sem,avg+sem, color='black',alpha=0.4)
						axs[row].text(-10,0,f'Ch {ch}',fontsize='small')
						
						if row==7:
			# 				axs[row].axis("off")
							axs[row].spines[['left', 'right', 'top']].set_visible(False)
							axs[row].set_yticks([])
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/psds['fs'],11),2))
							fig.tight_layout()
							fig,axs=plt.subplots(8,1)
							row=0
						else:
							axs[row].axis('off')
							row+=1
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
						
				plt.show()
			
			if self.stop_or_pause == 'stop':
				return #to stop after first session
			if self.stop_or_pause == 'pause':
				input('Press enter to continue') #to pause after each session
				
	
	def PlotGoodChs(self,area):
		'''
		Plots all good channels for specified area.
		Plots are averaged over all trials.
		Good channels defined in GoodChans.py

		Parameters
		----------
		area : 'Cd', 'ACC', 'M1', or 'PMd'
		'''
		
		print(f'\nPlotting good chs for {area}\n')
		
		
		for file_name in self.file_names:
			
			file = self._LoadData(file_name)
		
		
			if self.data_type == 'LFP':
				
				fig,ax=plt.subplots()
				ax.set_title(f'{file["name"]}\n{area}')
				
				fig,axs=plt.subplots(4,1)
				row=0
				
				data = file['LFP']
				
				from scipy.signal import butter
				from scipy.signal import filtfilt
				
# 				b,a = butter(5,[62,100],'bandpass',fs=file['fs'])
# # 				b,a = butter(5,62,'highpass',fs=file['fs'])
# 				pwr = filtfilt(b,a,data,axis=-1)**2
# 				self.scale_bar_power = 1e-11 #V**2
				pwr = data**2
				t = file['t']
				
				chs,_,_ = GetGoodChans(file['name'],area) #get good channels
				
				for ch in chs:
					
					try:
						avg= np.nanmean(pwr[ch,:,:],axis=0) #avg over trials
						sem= np.nanstd(pwr[ch,:,:],axis=0) / np.sqrt(np.size(pwr[ch,:,:],axis=0))
						axs[row].plot(t,avg,color='black',linewidth=1)
						axs[row].fill_between(t, avg-sem,avg+sem, color='black',alpha=0.4)
						axs[row].vlines(0,np.min(avg),np.max(avg),'red')
						axs[row].text(-0.7,np.mean(avg),f'Ch {ch}',fontsize='small')
						self._ScaleBar_Power(axs[row],t)
						
						if row==0:
							axs[row].text(-0.1,np.max(avg),'t=0',color='red')
	
						if row==3:
							axs[row].axis("off")
			# 				axs[row].spines[['left', 'right', 'top']].set_visible(False)
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/lfp['fs'],11),2))
							fig.tight_layout()
							fig,axs=plt.subplots(4,1)
							row=0
						else:
							axs[row].axis('off')
							row+=1
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
				plt.show()
		
		
			if self.data_type == 'Sxx':
				
				fig,ax=plt.subplots()
				ax.set_title(file['name'] + '\n' + area)
				
				from scipy.stats import zscore
				
				fig,axs=plt.subplots(3,4)
				row=0
				col=0
				
				data = file['Sxx']
				data = np.array(data)
				
				chs,_,_ = GetGoodChans(file['name'],area) #get good channels
				
				for ch in chs:
					
					try:
# 						avg= zscore(np.nanmean(data[ch,:,:,:],axis=0),axis=1) #avg over trials
						avg= np.nanmean(data[ch,:,:,:],axis=0) #avg over trials
						c = axs[row,col].pcolormesh(file['t'], file['f'], avg, cmap='plasma')
						axs[row,col].vlines(0,np.min(file['f']),np.max(file['f']),'black',linestyle='--')
						axs[row,col].set_title(f'Ch {ch}',fontsize='small')
# 						cbar = fig.colorbar(c, ax=axs[row,col])
						
						if row==2 and col==3:
							axs[row,col].axis("off")
			# 				axs[row].spines[['left', 'right', 'top']].set_visible(False)
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/file['fs'],11),2))
							fig.tight_layout()
							fig,axs=plt.subplots(3,4)
							row=0
							col=0
						else:
							if not (row==0 and col==0):
								axs[row,col].axis('off')
							if col<3:
								col+=1
							else:
								row+=1
								col=0
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
						
				plt.show()
			
				
			if self.data_type == 'PSD':
				
				
				
				fig,ax=plt.subplots()
				ax.set_title(f'{file["name"]}\n{area}')
				
				fig,axs=plt.subplots(8,1)
				row=0
				
				data = file['psd']
				
				chs,_,_ = GetGoodChans(file['name'],area) #get good channels
				
				for ch in chs:
					
					try:
						avg= np.nanmean(data[ch,:,:],axis=0) #avg over trials
						sem= np.nanstd(data[ch,:,:],axis=0) / np.sqrt(np.size(data[ch,:,:],axis=0))
						axs[row].plot(file['f'],avg,color='black')
						axs[row].fill_between(file['f'], avg-sem,avg+sem, color='black',alpha=0.4)
						axs[row].text(-10,0,f'Ch {ch}',fontsize='small')
						
						if row==7:
			# 				axs[row].axis("off")
							axs[row].spines[['left', 'right', 'top']].set_visible(False)
							axs[row].set_yticks([])
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/psds['fs'],11),2))
							fig.tight_layout()
							fig,axs=plt.subplots(8,1)
							row=0
						else:
							axs[row].axis("off")
							row+=1
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
				
				

				plt.show()
	
	
			if self.stop_or_pause == 'stop':
				return #to stop after first session
			if self.stop_or_pause == 'pause':
				input('Press enter to continue') #to pause after each session
				




	def PlotGoodChs_Var(self,area):
		'''
		Plots all good channels for specified area.
		Plots are averaged over all trials.
		Good channels defined in GoodChans.py

		Parameters
		----------
		area : 'Cd', 'ACC', 'M1', or 'PMd'
		'''
		
		print(f'\nPlotting good chs for {area}\n')
		
		
		for file_name in self.file_names:
			
			file = self._LoadData(file_name)
		
		
			if self.data_type == 'Sxx':
				
				fig,ax=plt.subplots()
				ax.set_title(file['name'] + '\n' + area)
				
				from scipy.stats import zscore
				
				fig,axs=plt.subplots(3,4)
				row=0
				col=0
				
				data = file['Sxx']
				data = np.array(data)
				
				chs,_,_ = GetGoodChans(file['name'],area) #get good channels
				
				for ch in chs:
					
					try:
						sem= np.nanstd(data[ch,:,:,:],axis=0) / np.sqrt(np.size(data[ch,:,:,:],axis=0)) #sem over trials
						c = axs[row,col].pcolormesh(file['t'], file['f'], sem, cmap='plasma')
						axs[row,col].vlines(0,np.min(file['f']),np.max(file['f']),'black',linestyle='--')
						axs[row,col].set_title(f'Ch {ch}',fontsize='small')
						cbar = fig.colorbar(c, ax=axs[row,col])
						
						if row==2 and col==3:
							axs[row,col].axis("off")
			# 				axs[row].spines[['left', 'right', 'top']].set_visible(False)
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/file['fs'],11),2))
							fig.tight_layout()
							fig,axs=plt.subplots(3,4)
							row=0
							col=0
						else:
							if not (row==0 and col==0):
								axs[row,col].axis('off')
							if col<3:
								col+=1
							else:
								row+=1
								col=0
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
						
				plt.show()

			if self.stop_or_pause == 'stop':
				return #to stop after first session
			if self.stop_or_pause == 'pause':
				input('Press enter to continue') #to pause after each session
				
				
				
	def PlotChs_SingleTrial(self,trial_num):
		'''
		Plots all channels for a single specified trial.

		Parameters
		----------
		trial_num : int
		'''
		
		print(f'\nPlotting all chs for trial {trial_num}\n')
		
		for file_name in self.file_names:
			
			file = self._LoadData(file_name)
		
		
			if self.data_type == 'LFP':
				
				fig,ax=plt.subplots()
				ax.set_title(file['name'] + f'\ntrial num: {trial_num}')
				
				fig,axs=plt.subplots(8,1)
				row=0
				
				data = file['LFP']
				t = file['t']
				
				chs = range(160)
				
				for ch in chs:
					
					try:
						axs[row].plot(t,data[ch,trial_num,:],color='black')
						axs[row].vlines(0,np.min(data[ch,trial_num,:]),np.max(data[ch,trial_num,:]),'red')
						axs[row].text(-0.7,0,f'Ch {ch}',fontsize='small')
						axs[row].set_ylim([0,self.scale_bar_voltage])
						
						if row==7:
							axs[row].axis("off")
			# 				axs[row].spines[['left', 'right', 'top']].set_visible(False)
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/lfp['fs'],11),2))
							fig.tight_layout()
							self._ScaleBar_Voltage(axs[0],t)
							fig,axs=plt.subplots(8,1)
							row=0
						else:
							axs[row].axis('off')
							row+=1
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
				self._ScaleBar_Voltage(axs[0],t)	
				plt.show()		
		
		
			if self.data_type == 'Sxx':
				
				fig,ax=plt.subplots()
				ax.set_title(file['name'] + f'\nTrial #{trial_num}')
				
				fig,axs=plt.subplots(3,4)
				row=0
				col=0
				
				data = file['Sxx']
				
				chs = range(160)
				
				for ch in chs:
					
					try:
						c = axs[row,col].pcolormesh(file['t'], file['f'], data[ch,trial_num,:,:], cmap='plasma')
						axs[row,col].vlines(0,np.min(file['f']),np.max(file['f']),'black',linestyle='--')
						axs[row,col].set_title(f'Ch {ch}',fontsize='small')
						
						if row==2 and col==3:
							axs[row,col].axis("off")
			# 				axs[row].spines[['left', 'right', 'top']].set_visible(False)
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/file['fs'],11),2))
							fig.tight_layout()
							fig,axs=plt.subplots(3,4)
							row=0
							col=0
						else:
							if not (row==0 and col==0):
								axs[row,col].axis('off')
							if col<3:
								col+=1
							else:
								row+=1
								col=0
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
						
				plt.show()	
			
				
			if self.data_type == 'PSD':
				
				
				fig,ax=plt.subplots()
				ax.set_title(file['name'] + f'\ntrial num: {trial_num}')
				
				fig,axs=plt.subplots(8,1)
				row=0
				
				data = file['psd']
				
				chs = range(160)
				
				for ch in chs:
					
					try:
						axs[row].plot(file['f'],data[ch,trial_num,:],color='black')
						axs[row].text(-10,0,f'Ch {ch}',fontsize='small')
						
						if row==7:
			# 				axs[row].axis("off")
							axs[row].spines[['left', 'right', 'top']].set_visible(False)
							axs[row].set_yticks([])
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/file['fs'],11),2))
							fig.tight_layout()
							fig,axs=plt.subplots(8,1)
							row=0
						else:
							axs[row].axis('off')
							row+=1
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
						
				plt.show()		
	
	
			if self.stop_or_pause == 'stop':
				return #to stop after first session
			if self.stop_or_pause == 'pause':
				input('Press enter to continue') #to pause after each session
				

		
	def PlotTrials_SingleCh(self,ch):
		'''
		Plots all trials for a single specified channel.

		Parameters
		----------
		ch : int
		'''
		
		print(f'\nPlotting all trials for ch {ch}\n')
		
		for file_name in self.file_names:
			
			file = self._LoadData(file_name)
		
		
			if self.data_type == 'LFP':
				
				fig,ax=plt.subplots()
				ax.set_title(file['name'] + f'\nch: {ch}')
				
				fig,axs=plt.subplots(8,1)
				row=0
				
				data = file['LFP']
				t = file['t']
				
				trials = range(np.shape(data)[1])
				
				for trial in trials:
					
					try:
						axs[row].plot(t,data[ch,trial,:],color='black')
						axs[row].vlines(0,np.min(data[ch,trial,:]),np.max(data[ch,trial,:]),'red')
						axs[row].text(-0.7,0,f'Trial {trial}',fontsize='small')
						
						if row==7:
							axs[row].axis("off")
			# 				axs[row].spines[['left', 'right', 'top']].set_visible(False)
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/file['fs'],11),2))
							fig.tight_layout()
							self._ScaleBar_Voltage(axs[0],t)
							fig,axs=plt.subplots(8,1)
							row=0
						else:
							axs[row].axis('off')
							row+=1
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
						
				self._ScaleBar_Voltage(axs[0],t)
				plt.show()	
		
		
			if self.data_type == 'Sxx':
				
				fig,ax=plt.subplots()
				ax.set_title(file['name'] + f'\nch: {ch}')
				
				fig,axs=plt.subplots(3,4)
				row=0
				col=0
				
				data = np.array(file['Sxx'])
				
				
				trials = range(len(file['behavior']['TrialType']))
				
				for trial in trials:			
						
					try:
						c = axs[row,col].pcolormesh(file['t'], file['f'], data[ch,trial,:,:], cmap='plasma')
						axs[row,col].vlines(0,np.min(file['f']),np.max(file['f']),'black',linestyle='--')
						axs[row,col].set_title(f'Trial {trial}',fontsize='small')
						
						if row==2 and col==3:
							axs[row,col].axis("off")
			# 				axs[row].spines[['left', 'right', 'top']].set_visible(False)
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/file['fs'],11),2))
							fig.tight_layout()
							fig,axs=plt.subplots(3,4)
							row=0
							col=0
						else:
							if not (row==0 and col==0):
								axs[row,col].axis('off')
							if col<3:
								col+=1
							else:
								row+=1
								col=0
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
						
				plt.show()	
			
				
			if self.data_type == 'PSD':
				
				
				fig,ax=plt.subplots()
				ax.set_title(file['name'] + f'\nch: {ch}')
				
				fig,axs=plt.subplots(8,1)
				row=0
				
				data = file['psd']
				
				trials = range(len(file['behavior']['TrialType']))
				
				for trial in trials:
					
					try:
						axs[row].plot(file['f'],data[ch,trial,:],color='black')
						axs[row].text(-10,0,f'Trial {trial}',fontsize='small')
						
						if row==7:
			# 				axs[row].axis("off")
							axs[row].spines[['left', 'right', 'top']].set_visible(False)
							axs[row].set_yticks([])
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/file['fs'],11),2))
							fig.tight_layout()
							fig,axs=plt.subplots(8,1)
							row=0
						else:
							axs[row].axis('off')
							row+=1
						
					except IndexError:
						print(f'Ch {ch} out of bounds for {file["name"]}')
						
				plt.show()	
	
	
			if self.stop_or_pause == 'stop':
				return #to stop after first session
			if self.stop_or_pause == 'pause':
				input('Press enter to continue') #to pause after each session
		
	def PlotTrialBatches_SingleCh(self,ch,batch_size):
		'''
		Plots a single channel averaged over batches of trials.

		Parameters
		----------
		ch : int
		batch_size: int
		'''
		
		print(f'\nPlotting trial batches for ch {ch}\n')
		
		if self.data_type != 'Sxx':
			print('This method is only capable of plotting Sxx data so far')
			return
		
		for file_name in self.file_names:
			
			file = self._LoadData(file_name)
		
		
			if self.data_type == 'Sxx':
				
				fig,ax=plt.subplots()
				ax.set_title(file['name'] + f'\nch: {ch}')
				
				fig,axs=plt.subplots(3,4)
				row=0
				col=0
				
				data = file['Sxx']
				
				trials = np.arange(0,len(file['behavior']['TrialType']),step=batch_size)
				
				for i in range(len(trials)-1):			
						
					try:
						avg= np.nanmean(data[ch,trials[i]:trials[i+1],:,:],axis=0) #avg over trials
						c = axs[row,col].pcolormesh(file['t'], file['f'], avg, vmin=0,vmax=2.5e-5, cmap='plasma')
						axs[row,col].vlines(0,np.min(file['f']),np.max(file['f']),'black',linestyle='--')
						axs[row,col].set_title(f'Trials {trials[i]}:{trials[i+1]}',fontsize='small')
			# 			print(np.max(avg),np.min(avg))
						
						if row==2 and col==3:
							axs[row,col].axis("off")
			# 				axs[row].spines[['left', 'right', 'top']].set_visible(False)
			# 				axs[row].set_xticks(np.linspace(0,len(avg),11),np.round(np.linspace(0,len(avg)/file['fs'],11),2))
							fig.tight_layout()
							fig,axs=plt.subplots(3,4)
							row=0
							col=0
						else:
							if not (row==0 and col==0):
								axs[row,col].axis('off')
							if col<3:
								col+=1
							else:
								row+=1
								col=0
						
					except IndexError:
						print(f'out of bounds for {file["name"]}')
						
				plt.show()		
			
				
			
			if self.stop_or_pause == 'stop':
				return #to stop after first session
			if self.stop_or_pause == 'pause':
				input('Press enter to continue') #to pause after each session
		