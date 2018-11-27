
class Explorer:
    """
    Epsilon-greedy with linearyly decayed epsilon

    Args:
      start_epsilon: max value of epsilon
      end_epsilon: min value of epsilon
      decay_steps: how many steps it takes for epsilon to decay

    """
    def __init__(self, start_eps, end_eps, decay_steps=100000):
        assert 0 <= start_eps <= 1, 'invalid start_eps'
        assert 0 <= end_eps <= 1, 'invalid end_eps'
        assert decay_steps >= 0

        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_steps = decay_steps
        self.eps = start_eps

    def value(self, t):
        if t >= self.decay_steps:
            return self.end_eps
        else:
            eps_diff = self.end_eps - self.start_eps
            return self.start_eps + eps_diff * (t / self.decay_steps)
