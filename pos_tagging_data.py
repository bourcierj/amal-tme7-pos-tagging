"""Utilities for handling GSD datasets from datamaestro"""

from tqdm import tqdm
import unicodedata
import string

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch


class VocabularyTagging:
    """Helper class to manage a vocabulary.
    Args:
        oov (bool): if True, allow out of vocabulary tokens
    """

    OOV_ID = 1  # out of vocabulary code
    NULL_ID = 0  # not a word (empty token)

    def __init__(self, oov: bool):
        self.oov = oov
        self.word2id = {'': 0}
        self.id2word = ['']
        if oov:
            self.word2id['__OOV__'] = 1
            self.id2word.append('__OOV__')

    def __getitem__(self, idx):
        return self.id2word[idx]

    def get(self, word: str, adding=True):
        """Maps a word to its id.
        Args:
            word (str): word for which to get the id
            adding (bool): if True, adds the word to vocab if it does not exists
        """
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return 1
            raise

    def __len__(self):
        return len(self.id2word)

    def decode(self, sentence):
        """Decodes a sentence.
        Args:
            sentence (list): a list or Tensor of ids
        Returns:
            list: a list of words
        """
        return [self.id2word[w] for w in sentence]


class TaggingDataset(Dataset):
    """Dataset for Part-Of-Speech tagging."""

    def __init__(self, data, words: VocabularyTagging, tags: VocabularyTagging,
                 adding=True):

        self.sentences = []
        for s in data:
            self.sentences.append((torch.tensor([words.get(token["form"], adding)
                                                 for token in s]),
                                   torch.tensor([tags.get(token["upostag"], adding)
                                                 for token in s])))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Args:
            idx: the index
        """
        return self.sentences[idx]

    @staticmethod
    def collate(batch):
        """Collate function (pass to DataLoader's collate_fn arg).
        Args:
            batch (list): list of examples returned by __getitem__
        Returns:
            tuple: Three tensors: batch of padded sequences, lengths of senquences,
            and targets (ie POS tags for each word)
        """
        #@todo: Change padding to 0 to be able to use CrossEntropyLoss with ignore_index=0
        # We have to ensure text and tags vocabulary codes start at 1.

        text, target = list(zip(*batch))
        lengths = torch.tensor([len(s) for s in text], dtype=torch.int)

        # pad the sequences with 0
        text = pad_sequence(text)
        target = pad_sequence(target)

        return (text, lengths, target)


if __name__ == '__main__':

    from datamaestro import prepare_dataset
    ds = prepare_dataset('org.universaldependencies.french.gsd')

    BATCH_SIZE = 100

    # format de sortie: https://pypi.org/project/conllu/

    print("Loading French GSD datasets...")
    words = VocabularyTagging(True)
    tags = VocabularyTagging(False)
    train_data = TaggingDataset(ds.files["train"], words, tags, True)
    dev_data = TaggingDataset(ds.files["dev"], words, tags, True)
    test_data = TaggingDataset(ds.files["test"], words, tags, False)

    print("Vocabulary size:", len(words))
    print("Number of tags:", len(tags))

    print(f"Train sentences: {len(train_data)}, test sentences: {len(test_data)}, "
          f"dev sentences: {len(dev_data)}")

    print("Train samples:")
    for i in range(3):
        sentence, target = train_data[i]
        print(f"Input: {' '.join(words.decode(sentence))}")
        print(f"Target: {' '.join(tags.decode(target))}")
        # print(f"Target: {[tags.id2word[t] for t in target]}")

    # print('Test of collate')
    # batch = train_data[:16]
    # text, lengths, target = train_data.collate(batch)
    # print(f'Text: {text}')
    # print(f'Lengths: {lengths}')
    # print(f'Target {target}')
