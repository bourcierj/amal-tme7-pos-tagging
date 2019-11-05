
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tagger(nn.Module):
    """Maps each word in sequences to their tag."""

    def __init__(self, vocab_size, num_tags, embedding_size, hidden_size):

        super(Tagger, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size)
        self.lin = nn.Linear(hidden_size, num_tags)

    def forward(self, x, lengths):
        """
        Args:
            x (Tensor): tensor of padded sequences of dim (T, B, 1)
            lengths (list): lengths of sentences, for packing.
        """
        embed = self.embedding(x)  # (T, B, embedding_size)
        # pack sequences for the RNN
        packed = pack_padded_sequence(embed, lengths, enforce_sorted=False)
        out, _ = self.rnn(packed)
        # unpack output of RNN
        out, out_lengths = pad_packed_sequence(out)

        assert(torch.all(lengths == out_lengths))
        out = self.lin(out)
        return out

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from datamaestro import prepare_dataset
    ds = prepare_dataset('org.universaldependencies.french.gsd')

    BATCH_SIZE = 100

    from pos_tagging_data import VocabularyTagging, TaggingDataset

    words = VocabularyTagging(True)
    tags = VocabularyTagging(False)
    train_dataset = TaggingDataset(ds.files["train"], words, tags, True)
    loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              collate_fn=train_dataset.collate)

    data, lengths, target = next(iter(loader))
    print(f"Input batch: {tuple(data.size())}, with lengths: {tuple(lengths.size())}")
    print(f"Target batch: {tuple(target.size())}\n")

    vocab_size = len(words)
    num_tags = len(tags)
    net = Tagger(vocab_size, num_tags, embedding_size=10, hidden_size=5)
    output = net(data, lengths)
    print(f"Output batch: {tuple(output.data.size())}\n")

    print(net)

