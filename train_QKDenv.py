"""
This script aims to train the RL agent that acts in the customized environment (QKDEnv()).

Note:
Environment: defined in environment.py, based on the physics engine defined in QKDengine_advanced.py

The agent ACTS by choosing a bias for AB, then OBSERVES the resulting distance, qber and gain. 
This rewards the agent either positively or negatively, depending on the qber and gain.
Then the episode is either terminated or continues in a new session (in this case the distance is randomized
again, but the agent can keep learning to choose the bias in a smarter way, based on the new distance and 
the previous experience).
"""

import gymnasium as gym
from stable_baselines3 import PPO #Proximal Policy Optimization, currently trending RL algoritmh
from stable_baselines3.common.callbacks import CheckpointCallback #just a check to save the model @ regular intervals
from stable_baselines3.common.monitor import Monitor #Monitor=class used to log the training progress
from stable_baselines3.common.callbacks import EvalCallback #to avoid trusting just the logs during the learning process 
from stable_baselines3.common.callbacks import BaseCallback #(blocco 4) serve per la callback diagnostica qui sotto
import numpy as np
from environment import QKDEnv
import json
import time
import os
from datetime import datetime

class ActionDistributionLogger(BaseCallback):
    def __init__(self, check_freq=1000, verbose=0):
        super().__init__(verbose)
        self.check_freq=check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            obs=self.locals.get("new_obs") #extracting the last observation without calling reset
            if obs is not None:
                obs_tensor,_=self.model.policy.obs_to_tensor(obs) #returns a tuple already
                dist=self.model.policy.get_distribution(obs_tensor) #(distribution not-squashed: directly picking mean/std from policy)
                mean_raw=dist.distribution.mean.detach().cpu().numpy().flatten()
                std_raw=dist.distribution.stddev.detach().cpu().numpy().flatten()
                clipped=np.clip(mean_raw, self.model.action_space.low, self.model.action_space.high)
                print(f"[step {self.num_timesteps}] action (bias) unrefined (==raw): mean={mean_raw[0]:.3f} std={std_raw[0]:.3f} | dopo clip=[0.1,0.9]: {clipped[0]:.3f}")
                self.logger.record("diagnostics/raw_action_mean", float(mean_raw[0]))
                self.logger.record("diagnostics/raw_action_std", float(std_raw[0]))
        return True

def train_agent(total_tmstpd):
    # path handling:
    base_dir=os.path.dirname(os.path.abspath(__file__))
    # logs/modls directories
    logs_dir=os.path.join(base_dir, "logs_tpm")
    models_dir=os.path.join(base_dir, "models_tpm")
    best_model_dir=os.path.join(logs_dir, "best_model")
    eval_log_dir=os.path.join(logs_dir, "eval")
    for path in [logs_dir, models_dir, best_model_dir, eval_log_dir]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Successfully created directory: {path}")
    env=QKDEnv()
    eval_env=Monitor(QKDEnv()) #evaluation environment, to be used in the EvalCallback, to evaluate the model during training
    
    #OLD version: before tpm (useful for names of old graphs)
    #env=Monitor(env, 'C:/Users/rorag/VS_Cpp/thesis proj/logs/')  #wrap of the environment, to be passed to the model as a monitored object
    env=Monitor(env, logs_dir)
    model=PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, ent_coef=0.05)
    #learning_rate of PPO to see how fast the agent learns
    #n_steps the number of steps to run for each environment before the agent updates the NN
    #batch_size extent of the data pool to use for each update
    action_logger=ActionDistributionLogger(check_freq=1000) #DIAGNOSTIC related (statistics for the ation distribution)
    #checkpoint_callback=CheckpointCallback(save_freq=5000, save_path='C:/Users/rorag/VS_Cpp/thesis proj/models/', name_prefix='ppo_qkd')
    #eval_callback=EvalCallback(eval_env=eval_env, best_model_save_path='C:/Users/rorag/VS_Cpp/thesis proj/logs/best_model/', log_path='C:/Users/rorag/VS_Cpp/thesis proj/logs/eval', eval_freq=5000, deterministic=True, render=False)
    checkpoint_callback=CheckpointCallback(save_freq=5000, save_path=models_dir, name_prefix='ppo_qkd_tpm')
    eval_callback=EvalCallback(eval_env=eval_env, best_model_save_path=best_model_dir, log_path=eval_log_dir, eval_freq=1000, deterministic=True, render=False)

    print('Starting training...')
    ti=time.time()
    model.learn(total_timesteps=total_tmstpd, callback=[checkpoint_callback, eval_callback, action_logger]) #total timesteps for training: 20-30k to start spotting intelligent behaviours; action_logger=diagnostica boundary-collapse
    #deterministic to AVOID having casual exploration thrg eevolution
    #callback passed as LIST (args: callback points)
    #saves the ebst models yet found during explorastion
    #admitting we do not rely solely onto the model logs, but we want to check SEPARATEDLY to get te actual learning curve behaviour
    tf=time.time()
    duration=tf-ti #training duration in seconds
    report = {
    "current_experiment": "Optimization Sifting QKD",
    "total_timesteps": total_tmstpd,
    "duration_in_s": duration,
    "fps_medi": 116, #from training logs
    #"model_saved": "ppo_qkd_optimized_sifting.zip"
    "model_saved": "ppo_qkd_tpm_optimized.zip"
    }

    # to prevent WINDOWS permission ERROR
    report_path=os.path.join(base_dir, "report_train_QKD.json")
    written_successfully=False
    for attempt in range(3):
        try:
            with open(report_path, "a") as f:
                json.dump(report, f)
                f.write("\n")
            written_successfully=True
            break
        except PermissionError:
            time.sleep(1) #WAIT for external application to release the file handle
    # fallback to unique timestamped file if Windows refuses to release main report
    if not written_successfully:
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path=os.path.join(base_dir, f"report_train_QKD_{timestamp}.json")
        with open(fallback_path, "w") as f:
            json.dump(report, f)
            f.write("\n")
        print(f"Warning: Primary report was locked. Wrote to: {fallback_path}")

    final_model_path=os.path.join(models_dir, "ppo_qkd_tpm_final")
    model.save(final_model_path)
    print('Training completed and model saved:)')

if __name__ == "__main__":
    train_agent(50000)