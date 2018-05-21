# Lexical Substitution Evaluation

This code was used to perform the lexical substitution evaluation described in the following papers:

**[1] A Simple Word Embedding Model for Lexical Substitution**
Oren Melamud, Omer Levy, Ido Dagan.  Workshop on Vector Space Modeling for NLP (VSM), 2015 [[pdf]](http://u.cs.biu.ac.il/~melamuo/publications/melamud_vsm15.pdf).

**[2] context2vec: Learning Generic Context Embedding with Bidirectional LSTM**  
Oren Melamud, Jacob Goldberger, Ido Dagan. CoNLL, 2016 [[pdf]](http://u.cs.biu.ac.il/~melamuo/publications/context2vec_camera_ready.pdf).


## Requirements

* Python 2.7
* [NLTK 3.0](http://www.nltk.org/))  - optional (only required for the AWE baseline and MSCC evaluation)
* Numpy
* [context2vec](https://github.com/orenmel/context2vec) - for the context2vec evaluation

## Datasets

This repository contains preprocessed data files based on the datasets introduced by the following papers:

**[3] Semeval-2007 task 10: English lexical substitution task**
Diana McCarthy, Roberto Navigli, SemEval 2007.  
(files with the prefix 'lst' under the 'dataset' directory)

**[4] What substitutes tell us-analysis of an ”all-words” lexical substitution corpus.**
Gerhard Kremer,Katrin Erk, Sebastian Pado,  Stefan Thater. EACL, 2014.  
(files with the prefix 'coinco' under the 'dataset' directory)

## Evaluating the word embedding model [1]

* Download the word embeddings, context embeddings from [[here]](http://u.cs.biu.ac.il/~nlp/resources/downloads/lexsub_embeddings/)
* Preprocess the embedding files:
```
python jcs/text2numpy.py <word-embeddings-filename> <word-embeddings-filename>
python jcs/text2numpy.py <context-embeddings-filename> <context-embeddings-filename>
```
* To perform the lexical substitution evaluation run (replace the example datasets files and params below as you wish):
```
python jcs/jcs_main.py --inferrer emb -vocabfile datasets/ukwac.vocab.lower.min100 -testfile datasets/lst_all.preprocessed -testfileconll datasets/lst_all.conll -candidatesfile datasets/lst.gold.candidates -embeddingpath <word-embeddings-filename> -embeddingpathc <context-embeddings-filename> -contextmath mult --debug -resultsfile <result-file>
```
* This will create the following output files: 
	- \<result-file\>
	- \<result-file\>.ranked
	- \<result-file\>.generate.oot
	- \<result-file\>.generate.best
* Run the following to compute the candidate ranking GAP score. The results will be written to \<gap-score-file\>.
```
python jcs/evaluation/lst/lst_gap.py ~/datasets/lst_all.gold <result-file>.ranked <gap-score-file> no-mwe
```
* Run the following to compute the OOT and BEST substitute prediction scores. The results will be written to \<xxx-score-file\>. score.pl was distributed in [3].
```
perl dataset/score.pl \<result-file\>.generate.oot datasets/lst_all.gold -t oot > \<oot-score-file\>
```
```
perl dataset/score.pl \<result-file\>.generate.best datasets/lst_all.gold -t best > \<best-score-file\>
```

## Evaluating the context2vec model [2]

* See [context2vec](https://github.com/orenmel/context2vec) for how to download or train a \<context2vec-model\>.
* To perform the lexical substitution evaluation run (replace the example datasets files and params below as you wish):
```
python jcs/jcs_main.py --inferrer lstm -lstm_config \<context2vec-model\>.params -testfile datasets/lst_all.preprocessed -testfileconll datasets/lst_all.conll -candidatesfile datasets/lst.gold.candidates -contextmath mult -resultsfile <result-file> --ignoretarget --debug
```
* From here, follow the same instructions as in the previous section.


## License

Apache 2.0





