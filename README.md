# Bayesian Skip-gram(BSG)

This repository contains Theano code for the Bayesian version of Skip-gram model presented in:
 
 [1] **Embedding Words as Distributions with a Bayesian Skip-gram Model**, Arthur Bražinskas, Serhii Havrylov, Ivan Titov, [arxiv](https://arxiv.org/abs/1711.11027)
 
The model represents words are Gaussian distributions instead of point estimates, and is capable of learning addition word properties, such as generality that is
encoded in variances. The instructions below provide a guide on how to install and run the model, also how to evaluate word pairs. 


## Requirements
- Python 2.7
- Theano 0.9.0
- numpy 1.14.2
- nltk 3.2.2
- scipy 0.18.1
- Lasagne 0.2.dev1

## Installation

First of all, install the dependency Python modules, such as Theano and nltk. 

```
pip install requirements.txt
```

Afterwards, install the necessary NLTK sub-packages. 

```
python -m nltk.downloader wordnet

python -m nltk.downloader punkt
```

## Runing the model 
In order to run the model, please refer to **run_bsg.py** file that contains an example code on how to train and evaluate the model. Upon completion of training,
word representations will be saved to the *output* folder. For example, one can use trained word Gaussian representations(mus and sigmas) as input to word pairs evaluation. 

### Data
A small dataset consisting of [15 million tokens](https://drive.google.com/open?id=1QWC2x6qq8KyHFUCgyvVJJoGHexZrw7gO) dataset is available for smoke tests of the setup. Alternatively, a dataset consisting of approximately [1 billion tokens](http://www.statmt.org/lm-benchmark/) is also available for the public use.
The dataset that was used originally in the research is not publicly available, but can be (requested)[http://wacky.sslmit.unibo.it/doku.php?id=corpora]. 

## Word pairs evaluation

One can use the **eval/word_pairs_eval.py** console application as a playground for word pairs evaluation in terms of similarity, Kullback-Leibler divergence,
and entailment directionality. The console application expects paths for word pairs, mu and sigma vectors(i.e. representations of word).
A word pairs file should contain two words(order does not matter) per line separated by space. The latter two files are obtained from a trained BSG model.
Alternative, pre-trained on the 3B tokens dataset [word representations](https://drive.google.com/open?id=1YQQHFV215YjKLlvxpxsKWLm__TlQMw1Q).

The example command below will evaluate pairs stored in **eval/example_word_pairs.txt**, and output results to the console. 
```
python eval/word_pairs_eval.py -wpp eval/example_word_pairs.txt -mup vectors/mu.vectors -sigmap vectors/sigma.vectors
```



## Additional resourced used in the project
Lexical substitution benchmark is a modified version of https://github.com/orenmel/lexsub 
