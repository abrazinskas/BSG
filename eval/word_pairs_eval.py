# a console application to evaluate word pairs
from support import KL, cosine_sim, read_vectors_to_dict
import argparse


def word_pairs_eval(word_pairs_path, mu_vectors_path, sigma_vectors_path):
    """
    :param word_pairs_path: file path that contains lines of the form word1 word2 (space separated)
    :param mu_vectors_path: path to the learned mu vectors
    :type mu_vectors_path: str
    :param sigma_vectors_path: path to the learned sigma vectors
    :type sigma_vectors_path: str

    """
    mus_and_sigmas = read_vectors_to_dict(mu_vectors_path, sigma_vectors_path, log_sigmas=False)

    with open(word_pairs_path) as f:
        for line in f:
            word1, word2 = line.strip().split()

            mu_w1, sigma_w1 = mus_and_sigmas[word1]
            mu_w2, sigma_w2 = mus_and_sigmas[word2]
            kl1 = KL(mu_w1, sigma_w1, mu_w2, sigma_w2)
            kl2 = KL(mu_w2, sigma_w2, mu_w1, sigma_w1)

            print "cos_sim(%s, %s) = %f" % (word1, word2, cosine_sim(mu_w1, mu_w2))
            print "kl(%s, %s) = %f" % (word1, word2, kl1)
            print "kl(%s, %s) = %f" % (word2, word1, kl2)

            my_str = "%s entails %s"
            if kl1 < kl2:
                print my_str % (word1, word2)
            else:
                print my_str % (word2, word1)
            print '---------------------------------------'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Word pairs evaluation using BSG learned representations.')
    parser.add_argument('-wpp', '--word_pairs_path', type=str)
    parser.add_argument('-mup', '--mu_vectors_path', type=str)
    parser.add_argument('-sigmap', '--sigma_vectors_path', type=str)
    args = parser.parse_args()
    word_pairs_eval(args.word_pairs_path, args.mu_vectors_path, args.sigma_vectors_path)
