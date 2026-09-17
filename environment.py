"""
Theoretical background:
A and B exchanging photons: to prepare them, use two basis, computational (Z) and Hadamar (X).
Distinguish them by setting Z codified by 0 and 1, while H by + and -.
In terms of polarization, this means Z gets either 0/90 and X gets 45/135.
BB84 protocol: A prepares them chosing the basis randomly, so does B once he measures them.
Biased basis selection INSTEAD relies on a biased coin: one basis gets chosen more often:
This way, Eve's evadesdropping the channel gets easily detected, as she will be more likely 
to choose the WRONG basis, and thus introduce ERRORS in the key.

https://gymnasium.farama.org/introduction/create_custom_env/

Since in the new version there are a's and B's tpms added, their weights distances are meant to
be compared: instead of changing the rumber of steps per episode (set initially @20), we try here
to vary the metrics of the env, using the Hamming distannce: 
$$v = \frac{1}{K \cdot N} \sum_{i=1}^{K} \sum_{j=1}^{N} \left( 1 - \delta(W_{ij}^A, W_{ij}^B) \right)$$
Old version commented below.
"""

import gymnasium as gym
from gymnasium import spaces
from QKDengine_advanced import QKDEngineAdvanced
from tpm_neural import TreeParityMachine as tpm
import numpy as np
from scipy.spatial import distance

class QKDEnv(gym.Env):
    def __init__(self):
        self.max_steps=20 #NOTEForUser: max number of sessions before truncation MIGHT be chsnged later
        self.tpm_A=tpm(K=3, N=4, L=6, M=3, B=2) #initialize the TPM for Alice
        self.tpm_B=tpm(K=3, N=4, L=6, M=3, B=2) #initialize the TPM for Bob
        #Action space:
        self.action_space=spaces.Box(low=0.1, high=0.9, shape=(1,)) #cfr: np.random.rand() for the ia to comfortably pick a bias in the range 0.1-0.9
        #Observation space:
        self.observation_space=spaces.Box(low=np.array([0.0,0.0,0.0], dtype=np.float32), high=np.array([1.0, 1.0, 1.0], dtype=np.float32), dtype=np.float32) #L=distance, qber=quantum bit error rate, gain=ratio of sifted key to initial bits NORMALIZED to be fed to the agent
        #physicial engine init from QKDAdvancedblabla
        self.engine=QKDEngineAdvanced(bits=10000, bias=0.5, bit_flip=0, p_eve=0.1, p_multiphoton=0.15, L=50, alpha=0.2, p_dark=1e-4, sigma=0.1)
        pass

    def reset(self, seed=None, options=None):
        #@each reset the agent experiences a new casual scenario (=>different distance, raANFOM)
        self.current_step=0 #reset the step counter
        super().reset(seed=seed) #standard for the env
        self.tpm_A= tpm(K=3, N=4, L=6, M=3, B=2) #reset the TPM for Alice
        self.tpm_B= tpm(K=3, N=4, L=6, M=3, B=2) #reset the TPM for Bob
        if options is not None and isinstance(options, dict) and 'distance' in options:
                    new_L=float(options['distance'])
        else:
            #new_L=np.random.uniform(0, 100) #OLD version
            if hasattr(self, 'current_L'):
                step_choice=np.random.choice([-10.0, 0.0, 10.0]) #discrete grid choices: -10, stayssame, +10
                new_L=float(np.clip(self.current_L + step_choice, 0.0, 150.0))
            else:
                new_L=10 #absolute first choice for first sim
        self.engine.L=new_L
        self.current_L=new_L
        # RECOMPUTING chloss based on new L
        if hasattr(self.engine, 'alpha'):
            self.engine.chloss=1-10**((-new_L*self.engine.alpha)/10)
        else:
            self.engine.chloss=1-10**((-new_L*0.2)/10)
        scaled_L=np.float32(new_L/150.0)
        observation=np.array([scaled_L, 0.0, 0.0], dtype=np.float32) #random distance, qber and gain are 0 AS at the beginning
        info={} #requested from gymnasium, empty currently 
        return observation, info

    def step(self, action):
        self.current_step+=1 #saving this for later
        bias=float(np.squeeze(action)) #about bias=action[0], to get the bias from action
        self.engine.bias=bias #update the bias in the ACTUAL engine
        # running the physicsss
        A_bits, A_bases = self.engine.generate_Alice()
        survived_bases, survived_bits, whistledown_mask = self.engine.channel_simulation(A_bases, A_bits)
        B_bases, B_bits = self.engine.generate_Bob(survived_bases, survived_bits)
        A_sifted, B_sifted, sifted_whistledown = self.engine.sifting(survived_bases, B_bases, survived_bits, B_bits, whistledown_mask)
        raw_qber, key_lenght=self.engine.calculate_metrics(A_sifted, B_sifted)
        # ERROR RECONCILIATION anfd ...:
        A_final, B_final=A_sifted, B_sifted
        final_key_len=0
        key_lenght_pre_privamp=key_lenght
        leaked_fraction=0.0 #setting this as default value if qber >= 0.11
        if raw_qber<0.11 and key_lenght>0: #if qber is too high, we don't even try to reconcile
            A_reconciled, B_reconciled, final_key_len=self.engine.reconciliation(A_sifted, B_sifted, raw_qber)
            key_lenght_pre_privamp=final_key_len #after reconciliation (leakage per error correction) but before privacy amplification
            key_lenght=final_key_len #update the key length after reconciliation
            A_final, B_final, final_key_len=self.engine.priv_ampl(A_reconciled, B_reconciled, final_key_len, sifted_whistledown)
            key_lenght=final_key_len #update the key length after priv ampl
            # ... PRIV amplification, covering fpr what Eve knows (leaked)
            if key_lenght_pre_privamp>0:
                leaked_fraction=1.0-(final_key_len/key_lenght_pre_privamp)
            else:
                leaked_fraction=0.0
        # TPMs sync session core:
        bits_per_tpm_input=self.tpm_A.K*self.tpm_A.N*self.tpm_A.B
        tpm_steps=0
        sync_success=0
        # using extracted key by A and B to feed TPMs: 
        while len(A_final)>=bits_per_tpm_input:
            current_bits_A=A_final[:bits_per_tpm_input]
            current_bits_B=B_final[:bits_per_tpm_input]
            A_final=A_final[bits_per_tpm_input:]
            B_final=B_final[bits_per_tpm_input:]
            X_A=self.tpm_A.convert_bits_to_input(current_bits_A)
            X_B=self.tpm_B.convert_bits_to_input(current_bits_B)
            if X_A is None or X_B is None: #no input vector for TPMs
                break
            tau_A=self.tpm_A.compute_output(X_A)
            tau_B=self.tpm_B.compute_output(X_B)
            if tau_A==tau_B:
                self.tpm_A.update_weights(X_A, tau_A, tau_B)
                self.tpm_B.update_weights(X_B, tau_B, tau_A)
            tpm_steps+=1
            if np.array_equal(self.tpm_A.weights, self.tpm_B.weights):
                sync_success=1
                break #holymoly succesfulll sync => halt
        # progressive-rewarding based on log (Hamming distance btwn Ws) => carrot
        diff_weights=np.sum(self.tpm_A.weights != self.tpm_B.weights)
        total_weights=self.tpm_A.K*self.tpm_A.N
        hamming_ratio=float(diff_weights/total_weights)
        qber=hamming_ratio #mapping qber perceived by agent
        SECURITY_PENALTY_WEIGHT=8.0 #=> stick
        real_gain=float(final_key_len/self.engine.bits)
        # evaluation of reward MUST be based on DISTANCE (agents choices does not entirely depend on his policy, but channel as well)
        distance_block = np.floor(self.engine.L / 10.0)  #like 0, 1, 2... 9
        distance_weight = np.exp(-0.2 * distance_block) #equally spaced distances weights from 1.0 to 0.1
        if sync_success==1:
            #gain=1.0
            #NOTEforUser: the synch bonus gets eaten by security penality (speeding up with extreme bias that gives away half of the key should be OPPOSED by agent)
            speed_bonus=(self.max_steps-self.current_step)*0.5 #to encourage faster syncs (less sessions)
            reward=(10.0 + speed_bonus)/(distance_weight + 1e-5)-(SECURITY_PENALTY_WEIGHT * leaked_fraction)
            terminated=True #IMPORTABT: terminating episode immediately if sync
        elif final_key_len==0: #failure penalty decreases with distance increasing
            #gain=0.0 
            reward=-2.0*distance_weight  # @ 10 km = -1.6, or @ 90 km = -0.3
            terminated=False
        else:
            #gain=0.0 
            # carrotting progressively if Hamm distance is DIMINISHED
            reward=(-5.0*hamming_ratio+(final_key_len/5000.0)-SECURITY_PENALTY_WEIGHT*leaked_fraction)*distance_weight
            terminated=False

        # if o bits were extracted (channel with too destructive asymmetric bias or smthg)
        #if final_key_lenght==0:
            #reward=-2.0  #IDLING penalty reduced slightly

        reward=float(np.clip(reward, -10.0, 15.0)) #used to stabilize PPO gradient updates :)

        #gain=key_lenght/self.engine.bts
        #update the observation: what we get
        #observation=np.array([self.engine.L, qber, gain], dtype=np.float32) #distance, qber, gain (new all)
        #CLAUSOLE (old VERSION):
        #terminated=(qber>0.45) or (gain==0) #terminate if qber is too high or no survivors
        #truncated=False #no time limit YET: OLD VERSION
        #terminated=np.array_equal(self.tpm_A.weights, self.tpm_B.weights) #ep concluded if and only if sync completed
        #hd=distance.hamming(self.tpm_A.weights.flatten(), self.tpm_B.weights.flatten()) #compute the hamming distance between the two TPMs
        #terminated=hd==0.0 #terminate if the two TPMs are synchronized (hamming distance=0)
        #truncated=self.current_step>=self.max_steps #truncated if we reach a certain number of steps (sessions)
        #reward:
        #if qber<0.11:
            #reward=gain*(1-2*qber) #reward is higher for higher gain and lower qber
        #else:
            #reward=-0.1-(qber-0.11) #penalize LESS heavily if qber is too high (insecure)

        #if terminated==True:
        #    reward=100+(self.max_steps-self.current_step)*2.0 #VALORE PESATO IN BASE A QUANTI POCHI STEP HA IMPIEGATO
        #elif not tpm_updated:
        #    reward = -2.0 - (qber * 5.0) #penalize if the TPMs couldn't be updated (not enough sifted bits)
        #else:
                #OLD version:
                #if qber<0.11:
                    #reward=gain*(1-2*qber) #reward is higher for higher gain and lower qber
        #    reward = -1.0 + (gain * 10.0) - (qber * 5.0) #new: shifted towards negative region (we'r supposedly guiding the agent)
                #else:
                    #reward=-0.1-(qber-0.11) #penalize LESS heavily if qber is too high (insecure)
        scaled_L=np.float32(self.engine.L/150.0)
        observation=np.array([scaled_L, np.float32(qber), np.float32(real_gain)], dtype=np.float32)

        truncated=self.current_step>=self.max_steps
        
        info={
            'raw_qber': raw_qber,
            'gain': real_gain,
            'key_length': final_key_len,
            'hamming_ratio': hamming_ratio,
            'tpm_synced': sync_success,
            'leaked_fraction': leaked_fraction
        }
        return observation, reward, terminated, truncated, info

#NOTEForUsers cmd used was:
# pip install stable-baselines3 shimmy
# stable_baselines3 is the library for the RL algorithm
# shimmy serves the purpuse of linking the custom env to sb3 (provides a wrapper to make the env compatible)