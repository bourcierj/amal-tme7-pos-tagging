
#amal #tme7 #pos-tagging #gru #pytorch

## POS-Tagging with a GRU RNN.

In this practical work, we try to solve the NLP task of part-of-speech tagging (POS-tagging). It consists of assigning to each word in a sentence its grammatical nature, or category (like a verb or a noun).

The dataset is the [French-GSD dataset](https://github.com/UniversalDependencies/UD_French-GSD).

The model is a simple RNN with a GRU cell. It is implemented in PyTorch.

Without trying a bidirectional or a stacked RNN for improvement, we quickly obtain over 90% accuracy on the validation set in just a few epochs.

TensorBoard runs with hyperparameters values are stored under the `runs/` directory.

## Results

| Set        | Cross-entropy-loss | Accuracy |
| ---------- | ------------------ | -------- |
| Train      | 4.2645e-02         | 0.9872   |
| Validation | 4.1209e-01         | 0.9193   |
| Test       | 3.6465e-01         | 0.9262   |

Hyperparameters values:
batch-size=128 - lr=0.01 - epochs=20 - patience=5 - clip=None - embedding-size=30 - hidden-size=30 - num-layers=1 - dropout=0 - bidirectional=False


## Usage


### Visualize TensorBoard runs

`tensorboard --logdir ./runs` and go to `localhost:6006` in your web browser.

### Training

To train the GRU RNN POS-tagger on the French-GSD dataset:
```
$ python pos_tagging_train.py --savepath PATH_TO_CHECKPOINT.PT
```
Full list of hyperparameters is accessible via `--help`. A new TensorBoard run with all parameters values is created under the `runs/` directory.

### Inference

Asks for a sentence as input and predict the POS-tags for each word using an already trained model:
```
$ python pos_tagging_inference.py --saved_path PATH_TO_CHECKPOINT.pt
```
You can pass the option `--text TEXT` to pass an optional text input directly, else you can pass text in interactive mode.
