import time
import numpy as np
import gymnasium as gym
import pandas as pd
from scipy.fft import dct
from stable_baselines3 import PPO
from environment import QKDEnv
import os

#OLD version before tpm: local
base_dir=os.path.dirname(os.path.abspath(__file__))
model_path=os.path.join(base_dir, "models_tpm", "ppo_qkd_tpm_final.zip")
model=PPO.load(model_path)

env=QKDEnv()
distances_test=np.linspace(0,150,16) #test su 16 distanze da 0 a 150 km
n_sessions=100 #number of sessions simulated for each distance

def test_agent(distances_test, n_sessions):
    reslts=[]
    for d in distances_test:
        #observation, info=env.reset()
        #env.unwrapped.engine.L=d #sets the distance for the test SUPER WRONG UNWRAPPING
        #env.set_distance(d)
        
        for i in range(n_sessions):
            observation, info=env.reset(options={'distance': d})
            
            for step_idx in range(20): #20 steps per session, then reset, since n_max steps=20 di là
                ti=time.time()
                action=model.predict(observation, deterministic=True)[0] #determ=True to say NOT TO GO EXPLORING, just use agent's policy
                next_observation, reward, terminated, truncated, info=env.step(action)
                tf=time.time()
                t=tf-ti
                riga={
                    'distance': d,
                    'session': i, 
                    'step': step_idx,
                    'bias': env.engine.bias, #bias chosen by the agent
                    'reward': reward, #reward obtained from the session
                    'qber': info['qber'], #qber obtained from the session
                    'gain': info['gain'], #gain obtained from the session
                    'key_length': info['key_length'], #sifted key length obtained from the session
                    'tpm_synced': info['tpm_synced'], #whether the TPMs were synchronized
                    'time': t #time taken for the step
                }
                reslts.append(riga)
                observation=next_observation
                if terminated or truncated:
                    #observation, info=env.reset() #reset the env for the next session
                    break #go to the next session

    df=pd.DataFrame(reslts)
    
    #df.to_csv(f'test_results.csv', index=False) #@local
    output_csv_path=os.path.join(base_dir, "test_results_tpm.csv")
    df.to_csv(output_csv_path, index=False)
    
    print('Test completed and dataframe saved!')

if __name__ == "__main__":
    test_agent(distances_test, n_sessions)