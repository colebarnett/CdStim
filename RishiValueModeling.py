# -*- coding: utf-8 -*-
"""
ValueModelingClass - Clean interface for Q-learning value extraction

This class provides a simple interface for extracting Q-values from HDF behavioral data
using various Q-learning models. Designed for easy integration into analysis pipelines.

Usage:
    from ValueModelingClass import ValueModelingClass
    
    # Get Q-values from HDF file
    vmc = ValueModelingClass()
    value_dict = vmc.get_values(hdf_file, num_trials_A, num_trials_B, method='PersDecay')
    
    # Returns: {'Q_low': array, 'Q_high': array}
    
    # Compare multiple models
    vmc.plot_model_comparison(hdf_file, num_trials_A, num_trials_B, 
                             models=['Baseline', 'PersDecay', 'KernelDecay'])

Author: HeyEVA (for Cole Barnett)
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from DecisionMakingBehavior import ChoiceBehavior_TwoTargets_Stimulation

# Import all Q-learning model variants
from TwoStageQLearning_Baseline import DistrQlearning_2Targs_TwoStage_Baseline
from TwoStageQLearning_BaselineDecay import DistrQlearning_2Targs_TwoStage_BaselineDecay
from TwoStageQLearning_Pers import DistrQlearning_2Targs_TwoStage_Pers
from TwoStageQLearning_PersDecay import DistrQlearning_2Targs_TwoStage_PersDecay
from TwoStageQLearning_PersSingle import DistrQlearning_2Targs_TwoStage_PersSingle
from TwoStageQLearning_PersSingleDecay import DistrQlearning_2Targs_TwoStage_PersSingleDecay
from TwoStageQLearning_Kernel import DistrQlearning_2Targs_TwoStage_Kernel
from TwoStageQLearning_KernelDecay import DistrQlearning_2Targs_TwoStage_KernelDecay
from TwoStageQLearning_KernelSingle import DistrQlearning_2Targs_TwoStage_KernelSingle
from TwoStageQLearning_KernelSingleDecay import DistrQlearning_2Targs_TwoStage_KernelSingleDecay
from TwoStageQLearning_AdaptiveOnly import DistrQlearning_2Targs_TwoStage_AdaptiveOnly
from TwoStageQLearning_AdaptiveDecay import DistrQlearning_2Targs_TwoStage_AdaptiveDecay
from TwoStageQLearning_PersAdaptive import DistrQlearning_2Targs_TwoStage_PersAdaptive
from TwoStageQLearning_PersAdaptiveDecay import DistrQlearning_2Targs_TwoStage_PersAdaptiveDecay
from TwoStageQLearning_KernelAdaptive import DistrQlearning_2Targs_TwoStage_KernelAdaptive
from TwoStageQLearning_KernelAdaptiveDecay import DistrQlearning_2Targs_TwoStage_KernelAdaptiveDecay
from TwoStageQLearning_DualBaseline import DistrQlearning_2Targs_TwoStage_DualBaseline
from TwoStageQLearning_DualBaselineDecay import DistrQlearning_2Targs_TwoStage_DualBaselineDecay


class ValueModelingClass:
    """
    Main class for extracting Q-values from HDF behavioral data.
    
    This class wraps the Q-learning model pipeline to provide a clean interface
    for getting Q-values and comparing different models.
    """
    
    def __init__(self):
        """Initialize the ValueModelingClass with available model methods."""
        
        # Dictionary mapping model names to their functions
        self.model_functions = {
            'Baseline': DistrQlearning_2Targs_TwoStage_Baseline,
            'BaselineDecay': DistrQlearning_2Targs_TwoStage_BaselineDecay,
            'Pers': DistrQlearning_2Targs_TwoStage_Pers,
            'PersDecay': DistrQlearning_2Targs_TwoStage_PersDecay,
            'PersSingle': DistrQlearning_2Targs_TwoStage_PersSingle,
            'PersSingleDecay': DistrQlearning_2Targs_TwoStage_PersSingleDecay,
            'Kernel': DistrQlearning_2Targs_TwoStage_Kernel,
            'KernelDecay': DistrQlearning_2Targs_TwoStage_KernelDecay,
            'KernelSingle': DistrQlearning_2Targs_TwoStage_KernelSingle,
            'KernelSingleDecay': DistrQlearning_2Targs_TwoStage_KernelSingleDecay,
            'AdaptiveOnly': DistrQlearning_2Targs_TwoStage_AdaptiveOnly,
            'AdaptiveDecay': DistrQlearning_2Targs_TwoStage_AdaptiveDecay,
            'PersAdaptive': DistrQlearning_2Targs_TwoStage_PersAdaptive,
            'PersAdaptiveDecay': DistrQlearning_2Targs_TwoStage_PersAdaptiveDecay,
            'KernelAdaptive': DistrQlearning_2Targs_TwoStage_KernelAdaptive,
            'KernelAdaptiveDecay': DistrQlearning_2Targs_TwoStage_KernelAdaptiveDecay,
            'DualBaseline': DistrQlearning_2Targs_TwoStage_DualBaseline,
            'DualBaselineDecay': DistrQlearning_2Targs_TwoStage_DualBaselineDecay,
        }
        
        # Model display properties for plotting
        self.model_properties = {
            'Baseline': {'color': 'blue', 'linestyle': '-', 'name': 'Baseline'},
            'BaselineDecay': {'color': 'cyan', 'linestyle': '-', 'name': 'Baseline+Decay'},
            'Pers': {'color': 'green', 'linestyle': '-', 'name': 'Perseverance'},
            'PersDecay': {'color': 'lime', 'linestyle': '-', 'name': 'Pers+Decay'},
            'PersSingle': {'color': 'darkgreen', 'linestyle': '-', 'name': 'Pers (Single-α)'},
            'PersSingleDecay': {'color': 'lightgreen', 'linestyle': '-', 'name': 'Pers+Decay (Single-α)'},
            'Kernel': {'color': 'orange', 'linestyle': '-', 'name': 'Choice Kernel'},
            'KernelDecay': {'color': 'red', 'linestyle': '-', 'name': 'Kernel+Decay'},
            'KernelSingle': {'color': 'darkorange', 'linestyle': '-', 'name': 'Kernel (Single-α)'},
            'KernelSingleDecay': {'color': 'salmon', 'linestyle': '-', 'name': 'Kernel+Decay (Single-α)'},
            'AdaptiveOnly': {'color': 'purple', 'linestyle': '--', 'name': 'Adaptive Only'},
            'AdaptiveDecay': {'color': 'violet', 'linestyle': '--', 'name': 'Adaptive+Decay'},
            'PersAdaptive': {'color': 'magenta', 'linestyle': '--', 'name': 'Pers+Adaptive'},
            'PersAdaptiveDecay': {'color': 'orchid', 'linestyle': '--', 'name': 'Pers+Adaptive+Decay'},
            'KernelAdaptive': {'color': 'brown', 'linestyle': '--', 'name': 'Kernel+Adaptive'},
            'KernelAdaptiveDecay': {'color': 'tan', 'linestyle': '--', 'name': 'Kernel+Adaptive+Decay'},
            'DualBaseline': {'color': 'navy', 'linestyle': ':', 'name': 'Dual Baseline'},
            'DualBaselineDecay': {'color': 'skyblue', 'linestyle': ':', 'name': 'Dual Baseline+Decay'},
        }
        
        # Parameter counts for AIC calculation (two-stage models have params for Block A and A')
        self.model_param_counts = {
            'Baseline': 6,  # 2 alphas + beta for A, same for A'
            'BaselineDecay': 8,  # 2 alphas + decay + beta for A, same for A'
            'Pers': 8,  # 2 alphas + rho + beta for A, same for A'
            'PersDecay': 10,  # 2 alphas + decay + rho + beta for A, same for A'
            'PersSingle': 6,  # alpha + rho + beta for A, same for A'
            'PersSingleDecay': 8,  # alpha + decay + rho + beta for A, same for A'
            'Kernel': 10,  # 2 alphas + ck_weight + ck_decay + beta for A, same for A'
            'KernelDecay': 12,  # 2 alphas + decay + ck_weight + ck_decay + beta for A, same for A'
            'KernelSingle': 8,  # alpha + ck_weight + ck_decay + beta for A, same for A'
            'KernelSingleDecay': 10,  # alpha + decay + ck_weight + ck_decay + beta for A, same for A'
            'AdaptiveOnly': 4,  # Just adaptive alpha + beta for A, same for A'
            'AdaptiveDecay': 6,  # Adaptive alpha + decay + beta for A, same for A'
            'PersAdaptive': 6,  # Adaptive alpha + rho + beta for A, same for A'
            'PersAdaptiveDecay': 8,  # Adaptive alpha + decay + rho + beta for A, same for A'
            'KernelAdaptive': 8,  # Adaptive alpha + ck_weight + ck_decay + beta for A, same for A'
            'KernelAdaptiveDecay': 10,  # Adaptive alpha + decay + ck_weight + ck_decay + beta for A, same for A'
            'DualBaseline': 6,  # 2 alphas + beta for A, same for A'
            'DualBaselineDecay': 8,  # 2 alphas + decay + beta for A, same for A'
        }
    
    def get_values(self, hdf_file, num_trials_A, num_trials_B, method='PersSingle'):
        """
        Extract Q-values from HDF file using specified Q-learning model.
        
        Parameters:
        -----------
        hdf_file : str or list of str
            Path to HDF file(s). Can be a single file or list of files for sessions
            split across multiple files.
        num_trials_A : int
            Number of trials in Block A (baseline block)
        num_trials_B : int
            Number of trials in Block B (stimulation block)
        method : str, optional (default='PersDecay')
            Q-learning model to use. Options include:
            - 'Baseline': Basic dual-alpha Q-learning
            - 'BaselineDecay': Baseline + decay/forgetting
            - 'Pers': Baseline + perseverance
            - 'PersDecay': Baseline + perseverance + decay (RECOMMENDED)
            - 'Kernel': Baseline + choice kernel
            - 'KernelDecay': Baseline + choice kernel + decay
            - 'AdaptiveOnly': Adaptive learning rates only
            - 'AdaptiveDecay': Adaptive learning rates + decay
            - 'PersAdaptive': Perseverance + adaptive learning
            - 'PersAdaptiveDecay': Perseverance + adaptive + decay
            And more (see self.model_functions for complete list)
        
        Returns:
        --------
        dict with keys:
            - 'Q_low': numpy array of Q-values for low-value target
            - 'Q_high': numpy array of Q-values for high-value target
            - 'prob_choice_low': numpy array of choice probabilities for low-value target
            - 'prob_choice_high': numpy array of choice probabilities for high-value target
            - 'chosen_target': numpy array of actual choices (1=low, 2=high)
            - 'rewards': numpy array of rewards received (0 or 1)
            - 'num_trials_A': number of Block A trials
            - 'num_trials_B': number of Block B trials
        """
        
        # Validate method
        if method not in self.model_functions:
            raise ValueError(f"Unknown method '{method}'. Available methods: {list(self.model_functions.keys())}")
        
        # Convert single file to list for consistency
        if isinstance(hdf_file, str):
            hdf_files = [hdf_file]
        else:
            hdf_files = hdf_file
        
        # Load behavioral data from HDF file(s)
        behavior = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, num_trials_A, num_trials_B)
        
        # Extract choices, rewards, and trial types
        chosen_target, rewards, instructed_or_freechoice = behavior.GetChoicesAndRewards()
        
        # Check if we got any trials
        if len(chosen_target) == 0:
            raise ValueError(
                f"No trials extracted from HDF file!"
            )
        
        # Run the Q-learning model
        model_func = self.model_functions[method]
        results = model_func(chosen_target, rewards, instructed_or_freechoice, 
                           num_trials_A, num_trials_B)
        
        # Unpack results (format varies slightly by model)
        Q_low = results[0]
        Q_high = results[1]
        prob_choice_low = results[2]
        prob_choice_high = results[3]
        accuracy = results[4]
        log_likelihood = results[5]  # Total log-likelihood across all blocks
        
        # Create output dictionary
        value_dict = {
            'Q_low': Q_low,
            'Q_high': Q_high,
            'prob_choice_low': prob_choice_low,
            'prob_choice_high': prob_choice_high,
            'chosen_target': chosen_target,
            'rewards': rewards,
            'instructed_or_freechoice': instructed_or_freechoice,
            'num_trials_A': num_trials_A,
            'num_trials_B': num_trials_B,
            'method': method,
            'log_likelihood': log_likelihood,
            'accuracy': accuracy,
        }
        
        
        print(f"Q_low range: [{np.min(Q_low):.3f}, {np.max(Q_low):.3f}]")
        print(f"Q_high range: [{np.min(Q_high):.3f}, {np.max(Q_high):.3f}]")
        
        
        return value_dict
    
    def plot_model_comparison(self, hdf_file, num_trials_A=100, num_trials_B=100, 
                             models=None, window_size=10, save_path=None):
        """
        Compare multiple Q-learning models against actual behavior WITH AIC comparison.
        
        Parameters:
        -----------
        hdf_file : str or list of str
            Path to HDF file(s)
        num_trials_A : int
            Number of trials in Block A
        num_trials_B : int
            Number of trials in Block B
        models : list of str, optional
            List of model names to compare. If None, uses ALL available models.
        window_size : int, optional (default=10)
            Sliding window size for smoothing actual behavior
        save_path : str, optional
            If provided, saves the figure to this path
        
        Returns:
        --------
        fig, axes : matplotlib figure and axes objects
        aic_values : dict of AIC values for each model
        """
        
        # Default to ALL models if not specified
        if models is None:
            models = list(self.model_functions.keys())
        # Get Q-values for each model
        model_results = {}
        aic_values = {}
        
        for model in models:
            print(f"\nRunning model: {model}")
            model_results[model] = self.get_values(hdf_file, num_trials_A, num_trials_B, method=model)
            
            # Calculate AIC: AIC = 2k - 2*log(L)
            k = self.model_param_counts.get(model, 6)  # Default to 6 if not found
            log_L = model_results[model]['log_likelihood']
            aic = 2 * k - 2 * log_L
            aic_values[model] = aic
            
            print(f"  Log-likelihood: {log_L:.2f}")
            print(f"  Parameters: {k}")
            print(f"  AIC: {aic:.2f}")
        
        # Extract actual behavior from first model (same for all)
        first_model = list(model_results.values())[0]
        chosen_target = first_model['chosen_target']
        
        # Convert choices to binary arrays for each target
        # chosen_target: 1=low, 2=high
        actual_choices_low = (chosen_target == 1).astype(float)
        actual_choices_high = (chosen_target == 2).astype(float)
        
        # Extract Block A' only (post-stimulation return)
        block_Ap_start = num_trials_A + num_trials_B
        actual_choices_low_Ap = actual_choices_low[block_Ap_start:]
        actual_choices_high_Ap = actual_choices_high[block_Ap_start:]
        
        # Smooth actual behavior
        behavior_smoothed_low = self._sliding_average(actual_choices_low_Ap, window_size)
        behavior_smoothed_high = self._sliding_average(actual_choices_high_Ap, window_size)
        
        # Create figure with THREE subplots
        fig = plt.figure(figsize=(24, 7))
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        ax3 = plt.subplot(1, 3, 3)
        
        trials = np.arange(len(actual_choices_low_Ap))
        
        # ========== LEFT PLOT: LOW VALUE TARGET ==========
        # Plot actual behavior (RED)
        ax1.plot(trials, behavior_smoothed_low, 'r-', linewidth=2.5, label='Actual Behavior', zorder=10)
        
        # Plot each model
        for model in models:
            if model in model_results:
                prob_low = model_results[model]['prob_choice_low'][block_Ap_start:]
                
                # Get model properties or use defaults
                props = self.model_properties.get(model, 
                    {'color': 'gray', 'linestyle': '-', 'name': model})
                
                ax1.plot(trials, prob_low, 
                        color=props['color'], 
                        linestyle=props['linestyle'],
                        linewidth=2.0,
                        label=props['name'],
                        alpha=0.8)
        
        ax1.set_xlabel('Trial Number (Block A\' only)', fontsize=14)
        ax1.set_ylabel('P(Choose Low Value Target)', fontsize=14)
        ax1.set_title('Low Value Target Selection', fontsize=16, fontweight='bold')
        ax1.legend(loc='best', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([-0.05, 1.05])
        
        # ========== MIDDLE PLOT: HIGH VALUE TARGET ==========
        # Plot actual behavior (RED)
        ax2.plot(trials, behavior_smoothed_high, 'r-', linewidth=3.0, label='Actual Behavior', zorder=10)
        
        # Plot each model
        for model in models:
            if model in model_results:
                prob_high = model_results[model]['prob_choice_high'][block_Ap_start:]
                
                # Get model properties or use defaults
                props = self.model_properties.get(model,
                    {'color': 'gray', 'linestyle': '-', 'name': model})
                
                ax2.plot(trials, prob_high,
                        color=props['color'],
                        linestyle=props['linestyle'],
                        linewidth=2.0,
                        label=props['name'],
                        alpha=0.8)
        
        ax2.set_xlabel('Trial Number (Block A\' only)', fontsize=14)
        ax2.set_ylabel('P(Choose High Value Target)', fontsize=14)
        ax2.set_title('High Value Target Selection', fontsize=16, fontweight='bold')
        ax2.legend(loc='best', fontsize=9, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.05, 1.05])
        
        # ========== RIGHT PLOT: AIC COMPARISON ==========
        # Sort models by AIC (lower is better)
        sorted_models = sorted(aic_values.items(), key=lambda x: x[1])
        model_names = [self.model_properties.get(m, {'name': m})['name'] for m, _ in sorted_models]
        aic_vals = [aic for _, aic in sorted_models]
        
        # Create bar chart
        bars = ax3.barh(range(len(model_names)), aic_vals, 
                       color=[self.model_properties.get(m, {'color': 'gray'})['color'] 
                             for m, _ in sorted_models],
                       alpha=0.7)
        
        ax3.set_yticks(range(len(model_names)))
        ax3.set_yticklabels(model_names, fontsize=10)
        ax3.set_xlabel('AIC (lower is better)', fontsize=14)
        ax3.set_title('Model Comparison (AIC)', fontsize=16, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Highlight best model (lowest AIC)
        best_idx = 0
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(3)
        
        # Add AIC values as text
        for i, (model, aic) in enumerate(sorted_models):
            ax3.text(aic + (max(aic_vals) - min(aic_vals)) * 0.01, i, 
                    f'{aic:.1f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n{'='*60}")
            print(f"Figure saved to: {save_path}")
            print(f"{'='*60}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Best model (lowest AIC): {sorted_models[0][0]} (AIC = {sorted_models[0][1]:.2f})")
        print(f"Worst model (highest AIC): {sorted_models[-1][0]} (AIC = {sorted_models[-1][1]:.2f})")
        print(f"{'='*60}\n")
        
        return fig, (ax1, ax2, ax3), aic_values
    
    def _sliding_average(self, data, window):
        """Apply sliding window average for smoothing."""
        smoothed = np.zeros(len(data))
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            smoothed[i] = np.mean(data[start:end])
        return smoothed
    
    def _get_num_parameters(self, model_name):
        """
        Get the number of free parameters for each model.
        
        Two-stage models fit Block A and A' separately, so they have
        roughly 2x the parameters (except beta which may be shared).
        
        Returns:
        --------
        int : number of free parameters
        """
        # Parameter counts for each model
        # Format: (params_per_block, shared_across_blocks)
        # Two-stage models fit A and A' separately
        param_counts = {
            # Baseline models: pos_alpha, neg_alpha, beta per block
            'Baseline': 3 * 2,  # 6 total (A and A' fitted separately)
            'BaselineDecay': 4 * 2,  # pos_alpha, neg_alpha, decay, beta per block
            
            # Perseverance models: add rho parameter
            'Pers': 4 * 2,  # pos_alpha, neg_alpha, rho, beta per block
            'PersDecay': 5 * 2,  # pos_alpha, neg_alpha, decay, rho, beta per block
            'PersSingle': 3 * 2,  # alpha, rho, beta per block (single alpha)
            'PersSingleDecay': 4 * 2,  # alpha, decay, rho, beta per block
            
            # Choice kernel models: add ck_weight, ck_decay
            'Kernel': 5 * 2,  # pos_alpha, neg_alpha, ck_weight, ck_decay, beta per block
            'KernelDecay': 6 * 2,  # pos_alpha, neg_alpha, decay, ck_weight, ck_decay, beta
            'KernelSingle': 4 * 2,  # alpha, ck_weight, ck_decay, beta per block
            'KernelSingleDecay': 5 * 2,  # alpha, decay, ck_weight, ck_decay, beta
            
            # Adaptive models: adaptive learning rates
            'AdaptiveOnly': 4 * 2,  # pos_alpha, neg_alpha, adaptive_param, beta per block
            'AdaptiveDecay': 5 * 2,  # pos_alpha, neg_alpha, decay, adaptive_param, beta
            
            # Combined models
            'PersAdaptive': 5 * 2,  # pos_alpha, neg_alpha, rho, adaptive_param, beta
            'PersAdaptiveDecay': 6 * 2,  # pos_alpha, neg_alpha, decay, rho, adaptive_param, beta
            'KernelAdaptive': 6 * 2,  # pos_alpha, neg_alpha, ck_weight, ck_decay, adaptive_param, beta
            'KernelAdaptiveDecay': 7 * 2,  # all of the above + decay
            
            # Dual baseline models
            'DualBaseline': 3 * 2,  # Same as Baseline
            'DualBaselineDecay': 4 * 2,  # Same as BaselineDecay
        }
        
        return param_counts.get(model_name, 6)  # Default to 6 if unknown
    
    def plot_aic_comparison(self, hdf_file, num_trials_A, num_trials_B, 
                           models=None, save_path=None):
        """
        Create AIC comparison bar chart across different Q-learning models.
        AIC = 2k - 2·log(L)
        where k = number of parameters, L = likelihood
        
        Parameters:
        -----------
        hdf_file : str or list of str
            Path to HDF file(s)
        num_trials_A : int
            Number of trials in Block A
        num_trials_B : int
            Number of trials in Block B
        models : list of str, optional
            List of model names to compare. If None, uses all available models.
        save_path : str, optional
            If provided, saves the figure to this path
        
        Returns:
        --------
        fig, ax : matplotlib figure and axes objects
        aic_values : dict
            Dictionary mapping model names to their AIC values
        """
        
        if models is None:
            models = list(self.model_functions.keys())
        print(f"\n{'='*60}")
        print(f"Computing AIC for {len(models)} models...")
        print(f"{'='*60}")
        
        # Get results for each model
        aic_values = {}
        bic_values = {}
        log_likelihoods = {}
        
        for model in models:
            print(f"\nRunning model: {model}")
            
            # Get model results
            result = self.get_values(hdf_file, num_trials_A, num_trials_B, method=model)
            
            # Extract log likelihood from the full model results
            # We need to re-run to get the log_prob_total
            if isinstance(hdf_file, str):
                hdf_files = [hdf_file]
            else:
                hdf_files = hdf_file
            
            behavior = ChoiceBehavior_TwoTargets_Stimulation(hdf_files, num_trials_A, num_trials_B)
            chosen_target, rewards, instructed_or_freechoice = behavior.GetChoicesAndRewards()
            
            model_func = self.model_functions[model]
            results = model_func(chosen_target, rewards, instructed_or_freechoice, 
                               num_trials_A, num_trials_B)
            
            # Extract log likelihood (usually index 5 in results tuple)
            log_likelihood = results[5]
            
            # Get number of parameters
            k = self._get_num_parameters(model)
            
            # Calculate AIC and BIC
            # AIC = 2k - 2·log(L)
            aic = 2 * k - 2 * log_likelihood
            
            # BIC = k·log(n) - 2·log(L)
            n = len(chosen_target)
            bic = k * np.log(n) - 2 * log_likelihood
            
            aic_values[model] = aic
            bic_values[model] = bic
            log_likelihoods[model] = log_likelihood
            
            print(f"  Log-likelihood: {log_likelihood:.2f}")
            print(f"  Parameters: {k}")
            print(f"  AIC: {aic:.2f}")
            print(f"  BIC: {bic:.2f}")
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort models by AIC (best to worst)
        sorted_models = sorted(models, key=lambda m: aic_values[m])
        sorted_aic = [aic_values[m] for m in sorted_models]
        
        # Get colors from model properties
        colors = [self.model_properties.get(m, {'color': 'gray'})['color'] 
                 for m in sorted_models]
        
        # Create bar chart
        x_pos = np.arange(len(sorted_models))
        bars = ax.bar(x_pos, sorted_aic, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Highlight the best model (lowest AIC)
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(3)
        
        # Add value labels on bars
        for i, (bar, aic) in enumerate(zip(bars, sorted_aic)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{aic:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('AIC (lower is better)', fontsize=14, fontweight='bold')
        ax.set_title('Model Comparison: Akaike Information Criterion', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        # Add text box with best model
        best_model = sorted_models[0]
        best_aic = sorted_aic[0]
        textstr = f'Best Model: {best_model}\nAIC = {best_aic:.2f}'
        props = dict(boxstyle='round', facecolor='gold', alpha=0.3)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=props, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nAIC comparison plot saved to: {save_path}")
        
        print(f"\n{'='*60}")
        print(f"Best model: {best_model} (AIC = {best_aic:.2f})")
        print(f"{'='*60}\n")
        
        return fig, ax, aic_values

if __name__ == "__main__":

    # Example: Extract Q-values from a session
    vmc = ValueModelingClass()
    
    hdf_file = '/Users/rishichapati/Documents/SantaCruzLab/RishiMacData/Luigi/hdf/luig20170927_07_te361.hdf'

    num_trials_A = 100
    num_trials_B = 100
    
    value_dict = vmc.get_values(hdf_file, num_trials_A, num_trials_B, method='PersDecay')
    print(f"\nQ-values extracted! Shape: {value_dict['Q_low'].shape}")
    
    vmc.plot_model_comparison(
        hdf_file, num_trials_A, num_trials_B,
        
        save_path='model_comparison.png'
    )
    plt.show()