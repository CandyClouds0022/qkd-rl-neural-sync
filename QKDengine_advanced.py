"""
Class for an advanced variation of the BB84 algorithm. In this scenario, Alice and Bob agreed on a
bias to be used for communication: before transmission, A proposed to use a certain amount of times
a base with respect to the other, for cofdification. 
The class inherits the methods from the physicalengine.py file, operating some modifications to 
certain methods (Alice generation stays the same, since the induced bias can be passed as an argument
upon class building initialization).

Actions:
- channel simulation: simulates the channel losses, the dark counts (false positives detected), the AWGN noise and Eve's attacks.
Eve gets smarter, since WCP can send multiphoton pulses, so that E might intercept one of the photons,
letting the other proceed undisturbed (=> Eve does NOT introduce errors, getting info);
- Bob generation: Bob generates his bases according to the bias, and guesses the bits where his bases
mismatch with Alice's;
- Sifting: the sifting procedure is performed, discarding the bits where the bases present mismatch;
- calculation of the QBER and the final key length (metrics for the model);
- Reconciliation step: if the QBER is below 11%, Bob may change his bits to resamble ALice's,
reducing the key amount of usable information by doing so;
- Privacy Amplification: the final key is hashed using a pseudo hashing encryption, but SHA256 
can be used for a more realistic approach (final key len penalized as well).

Notes for user are included as comments in the code.
"""

import numpy as np
import random
from hashlib import sha256

from physicalengine import QKDEngine

class QKDEngineAdvanced(QKDEngine):

    def __init__(self, bits, bias, bit_flip, p_eve, p_multiphoton, L, alpha, p_dark, sigma):
        super().__init__(bits, bias, L, bit_flip, p_eve)
        self.L=L #lenght of the fiber (distance in Km)
        self.alpha=alpha #absorption coefficient (typically around 0.2 dB/Km)
        #Lambert Beer Law for light absorption in media (dependance on distance and absorption coefficient)
        self.p_dark=p_dark #probability for Bob to perform DARK COUNTS
        self.sigma=sigma #standard deviation of the AWGN: NOISE, for mortals
        self.p_multiphoton=p_multiphoton #probability for a pulse to be multiphoton (sent by WCP source)

    def channel_simulation(self, crypted_bases, Abits):
        """
        This function simulates channel losses, dark counts, AWGN effects and Eve's non-detectable attacks.
        Returns:
        - survived_crypted_bases: the bases that survived the channel losses;
        - survived_A_bits: the bits that survived the channel losses, with noise and dark counts applied;
        - eve_leak_mask: a boolean mask indicating which bits were leaked to Eve 
        (multiphoton pulses she attacked succesfully).
        """
        Abits=Abits.astype(float) #to allow NaN values for the lost bits, which will be handled in the sifting phase
        N=int(self.bts)
        survived_mask=np.zeros(N) #0=dead, 1=survived

        # channel losses new:
        rand_mask=np.random.rand(N)
        #OLD VERSION: survived_mask[loss_mask>self.chloss]=1
        T=10**(-self.alpha * self.L/10) #new loss probability based on actual, real fiber components: computes the Transmittance
        #survived_mask[rand_mask<T]=1
        survived_mask=rand_mask<T #boolean mask: True if survives, False if lost
        survived_A_bits=np.where(survived_mask==1, Abits, np.nan)
        survived_crypted_bases=np.where(survived_mask==1, crypted_bases, np.nan)

        # smarter Eve: only attacking where ther's a photon (@survivedmask==true):
        eve_attack_roll=np.random.random(N)
        attacked_mask=(survived_mask) & (eve_attack_roll<self.p_eve)
        mpmask=np.random.random(N)<self.p_multiphoton #multiphoton mask
        whistledown_mask=(attacked_mask)&(mpmask) #amongst the attacked mask bools take the ones thatr multiphoton
        eve_leak_mask=np.copy(whistledown_mask) #to keep track of the bits that eve got
        for i in range(N):
            if whistledown_mask[i]: #Eve attacks a multiphoton pulse
                #Eve takes one photon, lets the other travel unperturbed
                #she does not introduce errors, but she gets the info after sifting
                pass
            elif attacked_mask[i]: #Eve attacks a single photon pulse
                #OLDER VERSION. no real biased bb84, Eve chooses a random base and measures the bit, then resends it to Bob
                #eve_base=np.random.randint(0,2)
                eve_base=0 if np.random.random()<self.bias else 1
                if eve_base==crypted_bases[i]: #Eve chooses the right base
                    #no more pass just because no errors were introduced
                    eve_leak_mask[i]=True #E non ha introdotto errore (but got the bit)
                else: #Eve chooses the wrong base
                    survived_A_bits[i]=np.random.randint(0,2) #introduces an error with 50% probability
        
        # AWGN (follow-up of the bit flip, more realistic for channels, theoretically)
        #to make them feasible => need to turn the bits into actual e-signals (bits 0:+1, bits 1:-1)
        masky=~np.isnan(survived_A_bits) #booleans to hit where there are no NaNs == mask to apply noise only to the survived bits
        Asignal=np.where(masky, 1-2*survived_A_bits, 0) #turn bits into signals: 0->+1, 1->-1, and 0 for the dead bits
        Anoise=np.random.normal(0, self.sigma, N)
        noisy_Asignal=Asignal+Anoise
        survived_A_bits=np.where(masky, np.where(noisy_Asignal > 0, 0, 1), np.nan) #converting back to bits, after noise, Bob's pov
        #reminder: in reality when we have huge lines (L) and a noisy channel, the real signal drowns in noise (fun thing, huh? PROVA)
        
        # Dark Counts (only for Bob):
        # logic behind: "dead=np.where(survived_A_bits==np.nan, 1, 0)" for def NaN
        dead=np.isnan(survived_A_bits)
        dark_trigger=dead & (np.random.rand(N) < self.p_dark) # p_dark deve essere tipo 1e-5
        survived_A_bits[dark_trigger]=np.random.randint(0, 2, np.sum(dark_trigger))

        return survived_crypted_bases, survived_A_bits, eve_leak_mask #returning the mask of the 
            #multiphoton pulses that Eve attacked, for later use in the sifting phase
    
    def generate_Bob(self, survived_crypted_bases, survived_A_bits):
        """
        Bob's generation of bases and bits (accounting for the bias), accounting for dark counts,
        receiving in input the survived bases and bits from the channel simulation.
        Returns: 
        - Bob_bases: the bases generated by Bob according to the bias;
        - B_bits: the bits generated by Bob, with his guesses.
        """
        Bob_bases = np.where(np.random.random(self.bts) < self.bias, 0, 1)
        B_bits=np.copy(survived_A_bits) #Bob counts the bits gotten out of the channel
        B_guesses_mask=(survived_crypted_bases != Bob_bases)&(~np.isnan(survived_A_bits)) #Bob guesses the bits where his bases mismatch with Alice's, but only for the survived bits
        B_bits[B_guesses_mask]=np.random.randint(0,2, np.sum(B_guesses_mask)) #Bob guesses the bits where his bases mismatch with Alice's, but only for the survived bits
        return Bob_bases, B_bits
    
    def sifting(self, survived_crypted_bases, B_bases, survived_A_bits, B_bits, whistledown_mask):
        """
        This advanced version of sifting discards if bases are different, OR if photon is lost
        (bit is NaN). Consider the MULTIPHOTON case (E uses her BS).
        Returns:
        - Asifted: the sifted bits for Alice;
        - Bsifted: the sifted bits for Bob;
        - sifted_whistledown: the mask for the sifted multiphoton bits.
        """
        match_mask=(survived_crypted_bases==B_bases) & (~np.isnan(B_bits)) #& without_multiphoton_mask
        Asifted=survived_A_bits[match_mask]
        Bsifted=B_bits[match_mask]
        sifted_whistledown=whistledown_mask[match_mask]  #mask for multiphoton bits that actually survived
        return Asifted, Bsifted, sifted_whistledown

    def calculate_metrics(self, Asifted, Bsifted):
        """
        This function calculates the QBER and the final key length after the sifting phase.
        Returns:
        - qber: the Quantum Bit Error Rate (QBER) between Alice and Bob's sifted bits;
        - key_len: the length of the sifted key.
        """
        if len(Asifted)==0:
            return 0.5, 0 # QBER max if no bits got to this point; key len is consequently 0
        errors=np.sum(Asifted != Bsifted)
        qber=errors/len(Asifted)
        key_len=len(Asifted)
        return qber, key_len

    def reconciliation(self, Asifted, Bsifted, qber):
        """
        This function simulates a reconciliation step.
        Needed implementation so that, if qber under 0.11 Bob may change his bits to align to 
        Alice's, resulting in a change of the key lenght. To do so, we evaluate the binary entropy
        associated to the QBER computed, as in: BINARY ENTROPY: H_2(e) = -e*log2(e) - (1-e)*log2(1-e).\
        Returns:
        - A_final: the reconciled bits for Alice;
        - B_final: the reconciled bits for Bob;
        - final_key_lenght: the length of the reconciled key.
        """
        if len(Asifted)==0 or len(Bsifted)==0:
            return Asifted, Bsifted, 0  #== NO bits to reconcile
        if qber<0.11:
            Bsifted_reconciled=np.copy(Asifted)
            if qber>0:
            #case in which we compute binary entropy
            #https://www.sigmaaldrich.com/IT/it/support/calculators-and-apps/absorbance-transmittance-conversion?srsltid=AfmBOooLsk_kHUpAgHha4xEqDDEH5vyOodQBaFEiFac--L8B_hUjZvBv
                h2=-qber*np.log2(qber) - (1-qber)*np.log2(1-qber)
                #al secolo, sindrome dell'errore simulato
            else:
                h2=0
            # real efficiencly factor:
            f=1.22 #SHOULD BE REALISTIC WRT the implemented PROTOCOL: ORIGINAL version
            #f=1.3 #for high error regimes (near QBER threshold)
            #f=1.16 #for cascade protocols perfectly implemented
            leakage=int(f*h2*len(Asifted)) #number of bits to be leaked to Eve
            #bits leaked onto the public channel REDUCE the lenght of the key
            #so that, moral of the story:
            final_key_lenght=max(0, (len(Asifted)-leakage))
            A_final=Asifted[:final_key_lenght]
            B_final=Bsifted[:final_key_lenght]
            return A_final, B_final, final_key_lenght
        else:
            return Asifted, Bsifted, len(Asifted) #highly unlikely to happen..

    def priv_ampl(self, A_final, B_final, final_key_lenght, sifted_whistledown):
        """
        This function simulates a privacy amplification step, where the final key is hashed using a pseudo hashing encryption.
        NEWS: penalize the final key lenght by the number of bits leaked to Eve (see reconciliation)
        Returns:
        - A_hashed: the hashed bits for Alice;
        - B_hashed: the hashed bits for Bob;
        - final_key_lenght: the length of the hashed key.
        """
        effective_whistled=sifted_whistledown[:final_key_lenght] #we only consider the bits that survived the priv ampl
        whistledown_count=np.sum(effective_whistled) #number of bits that Eve got for free
        final_key_lenght=max(0, final_key_lenght-whistledown_count) #penalizing the final key lenght by the number of bits leaked to Eve
        A_hashed=A_final[:final_key_lenght]
        B_hashed=B_final[:final_key_lenght]
        # older version: required splitting
        #A_bytes_hashed=sha256(A_final.tobytes()).digest()
        #B_Bytes_hashed=sha256(B_final.tobytes()).digest()
        return A_hashed, B_hashed, final_key_lenght