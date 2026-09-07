"""
TREE PARITY MACHINEs

Theory:
https://www.emergentmind.com/topics/tree-parity-machines-tpm

The TPM synchronization protocol enables two parties (typically termed Alice and Bob) to agree on a 
common secret key over a public channel. Both instantiate identical TPMs with random initial weights,
sharing only input vectors and output bits (tau) at each iteration. The protocol proceeds as:
1. One party samples and sends a random input vector.
2. Both compute their outputs and exchange  τA and τB
3. if those two taus are equal, both parties update only those weights for which 
σ(k)=τ (where sigma function is theactivation function σ(h(k)) using an 
agreed rule (Hebbian, anti-Hebbian, or random-walk):
4. Updated weights are clipped to [−L,L].


Implementation:
The class takes the survived sifted bits (with an acceptable qber), then groups/maps them 
into kxn arrays with values into a finite range [-L,L].
-> chaotic synchronization acts as a shared secret key generator
-> non-binary inputs should lower the synch time drastically, by increasing variance of the scalar product

TPM security is based on the fact that the weight configuration space is discrete and limited.
The interval -L,L indeed allows the system to operate like a random walk :)

-Integrate the tpm into the loop of the reconciliation phase!!!!!
DA FARE

"""

import numpy as np
import random


class TreeParityMachine:
    def __init__(self, K, N, L, M, B):
        self.K=K #number of hidden units
        self.N=N #length of each input vector
        #each of the input vector elements (x(k,n)) belongs to a range (-M,..,-1,1,..,M)
        #M>1 is the input range
        self.L=L #weight range (NB L>1)
        self.M=M #input range (NB M>1)
        self.B=B #lenght of the blocks through which the conversion procedure runs
        #TO AVOID binary random input vectors using [-M,M] instead of [-1,1]
        #OLD version: we avoided them so much sync was never actually acheived...
        #self.weights=np.random.uniform(-self.L, self.L, (self.K, self.N)) #KxN array
        self.weights=np.random.randint(-self.L, self.L + 1, size=(self.K, self.N))
        self.sigma = np.zeros(K) # Output dei neuroni nascosti: hint di gemini

    def compute_output(self, input_vector): 
        """
        This function is built to resamble a parity function of sorts, as -given input vectors
        shaped like input_vector=np.random.choice([-1,1], (self.K, self.N))- the output is evaluated
        by computing scalar product between weights and input.
        Returns:
        - tau: the global output of the TPM, which is the product of the outputs of the hidden units.
        """
        for k in range (self.K):
            self.sigma[k]=np.sign(np.dot(self.weights[k], input_vector[k]))
            if self.sigma[k]==0: #if the scalar product is zero, we can assign it randomly to either -1 or 1 DETERMINISTICALLY OBBV
                self.sigma[k]=1 #or -1, but we need to be deterministic for BOTH them parties to update the same weights
            #NB sigma is the hidden state
            tau=np.prod(self.sigma) #global tau output
        return tau
        
    def update_weights(self, input_vector, tau_A, tau_B, rule='hebbian'):
    #NOTEforUser: the computation might be initialized with other rules as well, maybe anti-hebbian or random-walk
        """
        Updating of weights encompassed in this function workflow: the logic states that for case 
        tauA==tauB, the weights gets directed towards the sign of the tau. In the event that computed weights
        exceed the range L, they get clipped to fit that range. However, different taus result in no
        weights updating.
        Returns:
        - weights: the updated weights after the update step.
        """
        if tau_A==tau_B:
            for k in range(self.K):
                if self.sigma[k]==tau_A: #update only those weights for which sigma(k)=tau
                    if rule=='hebbian':
                        self.weights[k]+=np.sign(input_vector[k]) #hebbian update
                    elif rule=='anti-hebbian':
                        self.weights[k]-=np.sign(input_vector[k]) #anti-hebbian update
                    elif rule=='random-walk':
                        self.weights[k]+=random.choice([-1,1])*np.sign(input_vector[k]) #random walk update
                    #clip weights to [-L,L]
                    self.weights[k]=np.clip(self.weights[k], -self.L, self.L)
            if tau_A!=tau_B:
                pass
        return self.weights
        
    def synchronize(self, other_party, max_iterations=10000):
        """
        The synchronization step followsthe idea of the article: 2104.11105v1%20(1).pdf. Due to that, 
        the input vector must NOT be random binary, so that synchronization can be achieved in a 
        reasonable number of iterations.
        Returns:
        - True if synchronization is achieved within the maximum number of iterations.
        """
            #other_party is the other TPM instance (Alice or Bob)
        for iteration in range(max_iterations):
            input_vector=np.random.randint(-self.M, self.M+1, (self.K, self.N)) #random input vector with values in the range [-L,L]
            tau_self=self.compute_output(input_vector)
            tau_other=other_party.compute_output(input_vector)
            self.update_weights(input_vector, tau_self, tau_other)
            other_party.update_weights(input_vector, tau_other, tau_self)
            if np.array_equal(self.weights, other_party.weights):
                print(f'Synchronization achieved in {iteration} iterations!')
                return True
            else:
                print(f'No synchronization achieved after {max_iterations} iterations.')

    def convert_bits_to_input(self, sifted_bits):
        """
        This function handles the conversion: it's employed to bridge the gap between the 
        qkd engines and the current class. It takes the sifted bits and maps them into 
        input vectors formatted (by splitting them into blocks and creating KxM arrays) 
        to adapt easily to the TPM architechture.
        NOTEToSelf: @the moment we generate randomly-> take bits from channel simulation and feed
        them to the machines :) 
        Returns:
        - input_vectors: the KxN array of input vectors for the TPM.
        """
        bits_needed=self.K*self.N*self.B #number of bits needed to create the input vectors
        if len(sifted_bits)<bits_needed:
            print(f'Not enough sifted bits to create input vectors. Needed: {bits_needed}, but got: {len(sifted_bits)}')
            return None
        #for mortals: @this step the tpm weights are not updated
        else:
            used_bits=sifted_bits[:bits_needed] #take only the needed bits
            blocks=np.split(used_bits, self.K*self.N) #split the sifted bits into K*N blocks
            input_vectors=np.zeros((self.K, self.N), dtype=int)
            for k in range(self.K):
                for n in range(self.N):
                    block_index=k*self.N+n          
                    block=blocks[block_index]
                    #map the block to a value in the range [-M,M]
                    #turning 0s into -1s:
                    transformed_block=2*block-1
                    value=int(np.sum(transformed_block)) #sum of bits in the block as a simple mapping
                    value=np.clip(value, -self.M, self.M) #clip to the range [-M,M]
                    input_vectors[k][n]=value
            return input_vectors
#taste test:
    def check_synchronization(self, tpm_B):
        return np.array_equal(self.W, tpm_B.W)