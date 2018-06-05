import os, sys
sys.path.append(os.path.join(os.getcwd(), "../../"))
from libraries.tools.vocabulary import Vocabulary
from main.simulators_interfaces.bsg_simulator_interface import BsgSimulatorInterface
from main.skipgram_embeddings import Skipgram_Embeddings
from main.support import read_vectors
from main.lex_sub import lex_sub


def run_lexsub(input_folder, output_path, half_window_size=5, input_type="normal", embeddings_type="bsg", sg_files_prefix="",
                   arithm_type="add"):
    assert embeddings_type in ['bsg', 'sg']
    assert input_type in ["normal", "dependency"]
    assert arithm_type in ["add", "mult"]
    # data paths
    candidates_file = os.path.dirname(os.path.realpath(__file__)) + "/datasets/lst.gold.candidates"
    test_file = os.path.dirname(os.path.realpath(__file__)) + "/datasets/lst_all.preprocessed"
    conll_file = os.path.dirname(os.path.realpath(__file__)) + "/datasets/lst_all.conll"
    gold_file = os.path.dirname(os.path.realpath(__file__)) + "/datasets/lst_all.gold"

    vocab_file_path = os.path.join(input_folder, 'vocab.txt')
    model_file_path = os.path.join(input_folder, 'model.pkl')
    vocab = Vocabulary()
    vocab.load(vocab_file_path=vocab_file_path)
    if embeddings_type == "bsg":
        embeddings = BsgSimulatorInterface(model_file_path=model_file_path, vocab=vocab)
    else:
        # default skipgram
        # two cases : either we have mu.vectors or prefix_input.vectors and prefix_output.vectors
        input_file_name = "_".join([sg_files_prefix, 'input'])+'.vectors' if sg_files_prefix != "" else "input.vectors"
        output_file_name = "_".join([sg_files_prefix, 'output'])+'.vectors' if sg_files_prefix != "" else "output.vectors"
        input_vectors_file_path = os.path.join(input_folder, input_file_name)
        output_vectors_file_path = os.path.join(input_folder, output_file_name)
        if not os.path.exists(input_vectors_file_path) or not os.path.exists(output_vectors_file_path):
            input_vectors_file_path = os.path.join(input_folder,  'mu.vectors')
            output_vectors_file_path = input_vectors_file_path
        embeddings = Skipgram_Embeddings(target_word_embeddings=read_vectors(input_vectors_file_path),
                                         context_embeddings=read_vectors(output_vectors_file_path))

    # extract word_to_index vectors from vocab
    word_to_index = {obj.token:obj.id for obj in vocab}
    target_words_vocab = word_to_index
    context_words_vocab = word_to_index

    print ' running  lexical substitution evaluation'
    # run lexical substitution
    lex_sub(embeddings=embeddings, input_type=input_type, target_words_vocab=target_words_vocab,
            context_words_vocab=context_words_vocab, candidates_file=candidates_file, conll_file=conll_file, test_file=test_file,
            gold_file=gold_file, half_window_size=half_window_size, output_path=output_path, arithm_type=arithm_type)


