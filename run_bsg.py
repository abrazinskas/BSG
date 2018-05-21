# this file contains an example on how to run the bayesian skip-gram model
import os

from interfaces.interface_configurator import InterfaceConfigurator
from evaluation.support import evaluate
from evaluation.lexsub.run_lexsub import run_lexsub


train_data_path = '../data/training/2M'
vocab_file_path = '../data/vocabulary/2M.txt'
output_folder_path = "output/2M/"
model_file_path = None

# obtain the interface to interact with the model. If one wants to change hyper-param the manual modification of the below class's method will be necessary!
i_model = InterfaceConfigurator.get_interface(train_data_path, vocab_file_path, output_folder_path, model_file_path)

i_model.train_workflow()

# store the temporary vocab, because it can be different from the original one(e.g. smaller number of words)
vocab = i_model.vocab
temp_vocab_file_path = os.path.join(i_model.output_path, "vocab.txt")
vocab.write(temp_vocab_file_path)

mu_vecs = [os.path.join(i_model.output_path, "mu.vectors")]
sigma_vecs = [os.path.join(i_model.output_path, "sigma.vectors")]

# a complex of word embedding evaluations(word similarity, entailment, directional entailment)
evaluate(mu_vectors_files=mu_vecs, sigma_vectors_files=sigma_vecs, vocab_file=temp_vocab_file_path, log_sigmas=False,
         full_sim=True, vocab=vocab)

# run lexical substitution evaluation
run_lexsub(input_folder=i_model.output_path, output_path=i_model.output_path)
