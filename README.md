
#amal #tme7 #pos-tagging #gru

## POS-Tagging with a GRU RNN.

In this practical work, we try to solve the NLP task of part-of-speech tagging (POS-tagging). It consists of assigning to each word in a sentence its grammatical nature or category.

The dataset is the [French-GSD dataset](https://github.com/UniversalDependencies/UD_French-GSD).

We use a RNN with a GRU cell.

Without trying a bidirectional or a stacked RNN for improvement, we are quickly able to obtain over 90% accuracy on the development set in just a few epochs.

Tensorboard runs with hyperparameters values are stored under the `runs/` directory.

## Usage

To train the GRU RNN on the dataset:
```
$ python pos_tagging_gsd.py <options>
```
A bunch of options are available (see the source file).

## Todo

- [ ] Manage out-of-vocabulary (OOV) words during inference. Here they just all have code one in the vocab. and no meaningful embedding.
- [ ] Evaluate on the test set
