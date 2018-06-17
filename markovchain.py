import numpy as np
import random

class MarkovChain():
    
    def __init__(self, states, transition=None, initial=None):
        self.states = states
        self.n = len(states)
        self.model = {}
        if transition is not None:
            self.transition = transition
        else:
            self.transition = np.zeros((self.n, self.n))
        if initial is not None:
            self.initial = initial
        else:
            self.initial = np.zeros((self.n, 1))
            
    def _train(self, seq):
        for i in range(0, len(seq)-1):
            state = seq[i]
            next_state = seq[i+1]
            if state not in self.model:
                self.model[state] = {}
            if next_state not in self.model[state]:
                self.model[state][next_state] = 0
            self.model[state][next_state] += 1
        self._cal_transition()
    
    def train(self, sequences):
        for seq in sequences:
            for i in range(0, len(seq)-1):
                state = seq[i]
                next_state = seq[i+1]
                if state not in self.model:
                    self.model[state] = {}
                if next_state not in self.model[state]:
                    self.model[state][next_state] = 0
                self.model[state][next_state] += 1
        self._cal_transition()
    
    def _cal_transition(self):
        for i in range(0, self.n):
            total = 0
            for j in range(0, self.n):
                try:
                    total += self.model[i][j]
                except:
                    continue
            for j in range(0, self.n):
                try:
                    self.transition[i][j] = self.model[i][j]/total
                except:
                    self.transition[i][j] = 0
                    
    def retrain(self, sequences):
        self.model = {}
        self.train(sequences)
        
    def get_probability(self, seq):
        prob = 1
        for i in range(0, len(seq)-1):
            state = seq[i]
            next_state = seq[i+1]
            prob *= self.transition[state][next_state]
        return prob