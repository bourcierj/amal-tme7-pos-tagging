"""
Asks for a sentence as input and predict the POS-tags for each word using an already
trained model.
"""

import argparse
import spacy  # spaCy is used only to tokenize text.
import torch
import torch.nn.functional as F

nlp = spacy.load('fr_core_news_sm')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(net, text, words, tags):
    """Predict the tags for every word in a raw sentence.
    Args:
        net (torch.nn.Module): the trained model
        text (str): the raw sentence
        words (TaggingVocabulary)
        tags (TaggingVocablary)
    """
    # tokenize the text using spacy
    doc = nlp(text)
    data = torch.tensor([words.get(token.text, False) for token in doc]).unsqueeze(1)
    length = torch.tensor([len(data)])
    net.eval()
    with torch.no_grad():
        data, length = data.to(device), length.to(device)
        output = net(data, length)
        output = output.view(length, -1)  # squeeze batch dim
        # predict the tags
        _, pred = F.log_softmax(output, 1).max(1)

    # return dict of words to tags
    word2tag = [(w, t) for w, t in zip(words.decode(data.squeeze(0)), tags.decode(pred))]
    return word2tag

if __name__ == '__main__':

    def parse_args():
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Predicts the POS-tags for an input sentence.")
        parser.add_argument('--saved-path', default='./checkpoints/saved/checkpt.pt',
                            type=str)
        parser.add_argument('--text', default=None, type=str)
        return parser.parse_args()

    torch.manual_seed(42)
    args = parse_args()

    from tagger import Tagger
    from pos_tagging_data import *
    from utils import CheckpointState

    batch_size = 128
    _, _, _, words, tags = \
        get_dataloaders_and_vocabs(batch_size=128)

    # load model from saved checkpoint
    net = Tagger(len(words), len(tags), embedding_size=30, hidden_size=30)
    checkpoint = CheckpointState(net, savepath=args.saved_path)
    checkpoint.load()
    net = net.to(device)

    if args.text is not None:
        word2tag = predict(net, args.text, words, tags)
        print("=>")
        print(word2tag)
    else:
        try:
            while True:
                # ask for input text
                text = input("Enter your text:\n")
                # predict the tags and print them
                word2tag = predict(net, text, words, tags)
                print("=>")
                print(word2tag)
        except KeyboardInterrupt:
            pass
