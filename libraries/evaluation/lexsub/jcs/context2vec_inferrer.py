'''
Context-sensitive inferrer based on context2vec (bidirectional lsmt)
Used in the paper:
context2vec: Learning Generic Context Embedding with Bidirectional LSTM. CoNLL, 2016.
'''

from cs_inferrer import CsInferrer
from jcs.jcs_io import vec_to_str
import numpy as np

# from context2vec.common.model_reader import ModelReader
    
#
# class Context2vecInferrer(CsInferrer):
#
#     def __init__(self, lstm_model_params_filename, ignore_target, context_math, top_inferences_to_analyze):
#
#         CsInferrer.__init__(self)
#         self.ignore_target = ignore_target
#         self.context_math = context_math
#         self.top_inferences_to_analyze = top_inferences_to_analyze
#
#         model_reader = ModelReader(lstm_model_params_filename)
#         self.context_model = model_reader.model
#         self.target_words = model_reader.w
#         self.word2index = model_reader.word2index
#         self.index2word = model_reader.index2word
#
#     def represent_target_and_context(self, lst_instance, tfo):
#
#         sent_words = lst_instance.words
#         target_ind = lst_instance.target_ind
#
#         ignore_target = self.ignore_target and len(sent_words) > 1 # if there's only the target word in the sentence then we don't ignore it...
#
#         if not ignore_target:
#             if lst_instance.target not in self.word2index:
#                 tfo.write("ERROR: %s not in word embeddings.Trying lemma.\n" % lst_instance.target)
#                 if lst_instance.target_lemma not in self.word2index:
#                         tfo.write("ERROR: lemma %s also not in word embeddings.\n" % lst_instance.target_lemma)
#                 else:
#                     sent_words[target_ind] = lst_instance.target_lemma
#
#         target_v = self.target_words[self.word2index[sent_words[target_ind]]] if not ignore_target else None
#
#         if len(sent_words) > 1:
#             context_v = self.context_model.context2vec(sent_words, target_ind)
#             context_v = context_v / np.sqrt((context_v * context_v).sum())
#         else:
#             context_v = None # just target with no context
#
#         return target_v, context_v
#
#
#     def find_inferred(self, lst_instance, tfo):
#
#         target_v, context_v = self.represent_target_and_context(lst_instance, tfo)
#
#         if target_v is not None and context_v is not None:
#
#         #This is not working very well at the moment. Requires more research.
#
#             # ZERO-TO-HALF
#             target_sim = (self.target_words.dot(target_v)+1.0)/2
#             context_sim = (self.target_words.dot(context_v)+1.0)/2
#             similarity = target_sim*context_sim
#
#             # RANKS
# #            target_sim = self.target_words.dot(target_v)
# #            context_sim = self.target_words.dot(context_v)
# #            for rank, i in enumerate(target_sim.argsort()):
# #                target_sim[i] = float(rank)
# #            for rank, i in enumerate(context_sim.argsort()):
# #                context_sim[i] = float(rank)
# #
# #            similarity = (target_sim*context_sim)/(len(target_sim)**2)
#
#             # POSITIVE SCORES
# #            target_sim = self.target_words.dot(target_v)
# #            context_sim = self.target_words.dot(context_v)
# #            target_sim[target_sim<0.0] = 0.0
# #            context_sim[context_sim<0.0] = 0.0
# #            similarity = target_sim*context_sim
#
#             # EXP
# #            target_sim = self.target_words.dot(target_v)
# #            target_sim = np.exp(target_sim)
# #            context_sim = self.target_words.dot(context_v)
# #            context_sim = np.exp(context_sim)
#
#
#             # NORMALIZE
# #            target_sim = self.target_words.dot(target_v)
# #            target_sim_mean = np.mean(target_sim)
# #            target_sim_std = np.sqrt(np.var(target_sim))
# #            target_sim = (target_sim - target_sim_mean)/target_sim_std
# ##            target_sim[target_sim<0.0] = 0.0
# #            context_sim = self.target_words.dot(context_v)
# #            context_sim_mean = np.mean(context_sim)
# #            context_sim_std = np.sqrt(np.var(context_sim))
# #            context_sim = (context_sim - context_sim_mean)/context_sim_std
# ##            context_sim[context_sim<0.0] = 0.0
# #
# #            similarity = target_sim + context_sim
#
#         else:
#             if target_v is not None:
#                 similarity = (self.target_words.dot(target_v)+1.0)/2
#             elif context_v is not None:
#                 similarity = (self.target_words.dot(context_v)+1.0)/2
#             else:
#                 raise Exception("Can't find a target nor context.")
#
#         result_vec = sorted(zip(self.index2word, similarity), reverse=True, key=lambda x: x[1])
#
#         tfo.write("Top most similar embeddings: " + vec_to_str(result_vec, self.top_inferences_to_analyze) + '\n')
#
#         return result_vec
        


