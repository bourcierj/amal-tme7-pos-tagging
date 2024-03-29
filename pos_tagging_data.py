"""Utilities for handling GSD dataset from datamaestro"""

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

from datamaestro import prepare_dataset


class VocabularyTagging:
    """Helper class to manage a vocabulary.
    Args:
        oov (bool): if True, allow out of vocabulary (OOV) tokens
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
            adding (bool): if True, adds word to vocabulary if it was not
                encountered before.
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

    def decode(self, sen):
        """Decodes a sentence.
        Args:
            sen (list): a list of Tensor of ids
        Returns:
            list: a list of string tokens
        """
        if isinstance(sen, torch.Tensor):
            sen = sen.tolist()
        return [self.id2word[w] for w in sen]


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
        text, target = list(zip(*batch))
        lengths = torch.tensor([len(s) for s in text], dtype=torch.int)
        # pad the sequences with 0 (0 maps to empty ('') character)
        text = pad_sequence(text)
        target = pad_sequence(target)
        return (text, lengths, target)

    def num_oov_words(self, words: VocabularyTagging):
        """Returns the number of OOV words"""
        n_oov = 0
        for s, t in self.sentences:
            n_oov += (s == words.OOV_ID).sum().item()
        return n_oov


def get_dataloaders_and_vocabs(batch_size):

    ds = prepare_dataset('org.universaldependencies.french.gsd')

    words = VocabularyTagging(True)
    tags = VocabularyTagging(False)
    train_dataset = TaggingDataset(ds.files['train'], words, tags, True)
    val_dataset = TaggingDataset(ds.files['dev'], words, tags, False)
    test_dataset = TaggingDataset(ds.files['test'], words, tags, False)

    kwargs = dict(collate_fn=TaggingDataset.collate,
                  pin_memory=(torch.cuda.is_available()),
                  num_workers=torch.multiprocessing.cpu_count()
                  )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             **kwargs)
    return train_loader, val_loader, test_loader, words, tags


if __name__ == '__main__':

    ds = prepare_dataset('org.universaldependencies.french.gsd')

    BATCH_SIZE = 100

    # format de sortie: https://pypi.org/project/conllu/

    print("Loading French GSD datasets...")
    words = VocabularyTagging(True)
    tags = VocabularyTagging(False)
    train_data = TaggingDataset(ds.files['train'], words, tags, True)
    dev_data = TaggingDataset(ds.files['dev'], words, tags, False)
    test_data = TaggingDataset(ds.files['test'], words, tags, False)

    print("Vocabulary size:", len(words))
    print("Number of tags:", len(tags))

    print(f"Train sentences: {len(train_data)}, test sentences: {len(test_data)}, "
          f"dev sentences: {len(dev_data)}\n")

    print("Train samples:")
    for i in range(3):
        sentence, target = train_data[i]
        print(f"Input: {' '.join(words.decode(sentence))}")
        print(f"Target: {' '.join(tags.decode(target))}")
        # print(f"Target: {[tags.id2word[t] for t in target]}")
    print()
    print(f"Number of OOV words in dev set: {dev_data.num_oov_words(words)}")
    print(f"Number of OOV words in test set: {test_data.num_oov_words(words)}")

    # print('Test of collate')
    # batch = train_data[:16]
    # text, lengths, target = train_data.collate(batch)
    # print(f'Text: {text}')
    # print(f'Lengths: {lengths}')
    # print(f'Target {target}')
