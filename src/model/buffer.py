class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_ = []
        self.next_values = []
        self.weights = []
        self.weights_ = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.states_[:]
        del self.next_values[:]
        del self.weights[:]
        del self.weights_[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def __len__(self):
        return len(self.actions)