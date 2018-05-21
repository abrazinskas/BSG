# Bayesian Skip-gram(BSG)

This repository contains Theano code for the Bayesian version of Skip-gram model presented in [Embedding Words as Distributions with a Bayesian Skip-gram Model](https://arxiv.org/abs/1711.11027).
The model represents words are Gaussian distributions instead of point estimates, and is capable of learning addition word properties, such as generality. 
The instructions below provide a guide on how to install and run the model, also how to evaluate word pairs. 


## Installation

First of all, install the dependency Python modules, such as Theano and nltk. 

```
pip install requirements.txt
```

Afterwards, one needs to install necessary NLTK sub-packages. 

```
python -m nltk.downloader wordnet

python -m nltk.downloader punkt
```

## Runing the model 
In order to run the model, please refer to **run_bsg.py** file that contains an example code on how to train and evaluate the model. Upon completion of training,
word representations will be saved to the *output* folder. For example, one can use trained word Gaussian representations(mus and sigmas) as input to word pairs evaluation. 

## Word pairs evaluation

One can use the **eva/word_pairs_eval.py** console application as a playground for word pairs evaluation in terms of similarity, Kullback-Leibler divergence,
and entailment directionality. The console application expects paths for word pairs, mu and sigma vectors(i.e. representations of word).
A word pairs file should contain two words(order does not matter) per line separated by space. The latter two files are obtained from a trained BSG model.
Alternative, pre-trained on a 3B tokens dataset [word representations](https://drive.google.com/open?id=1YQQHFV215YjKLlvxpxsKWLm__TlQMw1Q).

A run example is provided below. 

```
python eval/word_pairs_eval.py my_word_pairs.txt mu.vectors sigma.vectors
```


Lexical substitution benchmark is a modified version of https://github.com/orenmel/lexsub 
