import numpy as np
from support import cosine_sim, pos_cosine_sim_normed

class Skipgram_Embeddings():

    def __init__(self, target_word_embeddings, context_embeddings):
        self.w_emb = target_word_embeddings
        self.c_emb = context_embeddings

    # performs the scoring of candidates embeddings given
    # returns a dictionary of scores
    def score(self, target, context, candidates, ar_type="add", i=-1):
        assert ar_type in ["add", "mult"]
        scores = {}
        if ar_type == "add":
            cont_repr = self.__repr_context_add(target, context, avg_flag=True)
        for cand in candidates:
            if ar_type == "add":
                scores[cand] = self.__add(cont_repr, cand=self.w_emb[cand])
            elif ar_type == "mult":
                scores[cand] = self.__mult(target, context, cand, geo_mean_flag=True)
        return scores

    # Oren's strange add operation
    def __add(self, repr, cand):
        return np.dot(repr, cand)

    def __repr_context_add(self, target, deps, avg_flag=True):
        target_vec = None if target is None else np.copy(self.w_emb[target])
        dep_vec = None
        deps_found = 0
        for dep in deps:
            if dep in self.c_emb:
                deps_found += 1
                if dep_vec is None:
                    dep_vec = np.copy(self.c_emb[dep])
                else:
                    dep_vec += self.c_emb[dep]

        ret_vec = None
        if target_vec is not None:
            ret_vec = target_vec
        if dep_vec is not None:
            if avg_flag:
                dep_vec /= deps_found
            if ret_vec is None:
                ret_vec = dep_vec
            else:
                ret_vec += dep_vec

        norm = (ret_vec.dot(ret_vec.transpose()))**0.5
        ret_vec /= norm

        return ret_vec

    def __mult(self, target, deps, subsitute, geo_mean_flag=True):
        target_vec = self.w_emb[target]
        subs_vec = self.w_emb[subsitute]
        score = pos_cosine_sim_normed(target_vec, subs_vec)
        for dep in deps:
            if dep in self.c_emb:
                dep_vec = self.c_emb[dep]
                mult_scores = pos_cosine_sim_normed(dep_vec, subs_vec)
                if geo_mean_flag:
                    mult_scores = mult_scores**(1.0/len(deps)) # TODO: think if you need to fix it because len(deps) +1 should be here
                score = np.multiply(score, mult_scores)
        return score



    # def __add(self, center_word, context, substitute):
    #     seen = 0
    #     scr = 0
    #     if center_word in self.w_emb:
    #         scr += cosine_sim(self.w_emb[center_word], self.w_emb[substitute])
    #         seen += 1
    #     for c in context:
    #         if c in self.c_emb:
    #             scr += cosine_sim(self.c_emb[c], self.w_emb[substitute])
    #             seen+=1
    #     return scr/(seen)
