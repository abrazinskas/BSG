from libraries.simulators.bsg_simulator import BsgSimulator
from libraries.simulators.support import KL


class BsgSimulatorInterface:
    """
    Interface class to access simulators.

    """
    def __init__(self, vocab, model_file_path):
        self.simulator = BsgSimulator(vocab=vocab, model_file_path=model_file_path)

    def score(self, target, left_context, right_context, candidates, **kwargs):
            scores = {}
            mu_q, sigma_q = self.simulator.encode(context_words=left_context+right_context, center_word=target)
            for cand in candidates:
                mu_p, sigma_p = self.simulator.get_representation(cand)
                scores[cand] = -1*KL(mu_q, sigma_q, mu_p, sigma_p)
            return scores