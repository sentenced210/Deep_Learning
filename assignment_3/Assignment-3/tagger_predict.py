# python3.7 tagger_predict.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import sys
import torch
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class POS_tag_net(nn.Module):
    def __init__(self, vocab_size, char_size, target_size,
                 word_emb_dim=32, char_emb_dim=8,
                 out_channels=16, kernel_size=3,
                 lstm_hidden_dim=64):

        super().__init__()
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.target_size = target_size
        self.word_emb_dim = word_emb_dim
        self.char_emb_dim = char_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, word_emb_dim)
        self.char_embedding = nn.Embedding(char_size, char_emb_dim)
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(self.char_emb_dim, out_channels=self.out_channels, kernel_size=self.kernel_size,
                               padding=self.kernel_size // 2)

        self.lstm = nn.LSTM(self.word_emb_dim + self.out_channels,
                            self.lstm_hidden_dim,
                            bidirectional=True)

        self.fc1 = nn.Linear(self.lstm_hidden_dim * 2, self.target_size)

    def forward(self, words_batch, chars_batch, max_len_word, max_len_sentence):

        right_words_batch = []
        for words in words_batch:
            embedding = self.word_embedding(torch.tensor(words))

            if embedding.shape[0] != max_len_sentence:
                embedding = torch.cat(
                    (embedding, torch.zeros(max_len_sentence - embedding.shape[0], self.word_emb_dim)))

            right_words_batch.append(embedding)

        right_words_batch = torch.stack(right_words_batch)

        #         print(right_words_batch.shape)

        right_chars_batch = []

        for words in chars_batch:
            sentence = []
            for chars in words:
                c_embedding = self.char_embedding(torch.tensor(chars))

                if c_embedding.shape[0] != max_len_word:
                    c_embedding = torch.cat(
                        (c_embedding, torch.zeros(max_len_word - c_embedding.shape[0], self.char_emb_dim)))

                sentence.append(torch.t(c_embedding))

            while len(sentence) != max_len_sentence:
                sentence.append(torch.t(torch.zeros(max_len_word, self.char_emb_dim)))

            sentence = torch.stack(sentence)
            right_chars_batch.append(sentence)

        right_chars_batch = torch.stack(right_chars_batch)

        right_chars_batch = right_chars_batch.view(-1, self.char_emb_dim, max_len_word)

        right_chars_batch = F.relu(self.conv1(right_chars_batch))

        right_chars_batch, _ = torch.max(right_chars_batch, dim=-1)

        right_chars_batch = right_chars_batch.view(-1, max_len_sentence, self.out_channels)

        lstm_input = torch.cat((right_words_batch, right_chars_batch), dim=-1)

        lstm_output, _ = self.lstm(lstm_input)

        out = F.log_softmax(self.fc1(lstm_output), dim=-1)

        return out

    def loss(self, Y_hat, Y):
        Y = Y.view(-1)
        Y_hat = Y_hat.view(-1, self.target_size)
        mask = (Y > -1).float()
        nb_tokens = int(torch.sum(mask).item())
        Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask
        cross_entropy_loss = -torch.sum(Y_hat) / nb_tokens

        return cross_entropy_loss


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
    # use torch library to load model_file

    model_meta_parameters, word2idx, tag2idx, idx2tag, char2idx, model_state_dict = torch.load(model_file)

    my_net = POS_tag_net(vocab_size=model_meta_parameters['vocab_size'],
                             char_size=model_meta_parameters['char_size'],
                             target_size=model_meta_parameters['target_size'],
                             word_emb_dim=model_meta_parameters['word_emb_dim'],
                             char_emb_dim=model_meta_parameters['char_emb_dim'],
                             out_channels=model_meta_parameters['out_channels'],
                             kernel_size=model_meta_parameters['kernel_size'],
                             lstm_hidden_dim=model_meta_parameters['lstm_hidden_dim'])

    my_net.load_state_dict(model_state_dict)

    out = open(out_file, 'w')
    with open(test_file, 'r') as f:
        for line in f:

            X = [line.rstrip("\n").rstrip("\r").lower().split(" ")]
            tokens = line.rstrip("\n").rstrip("\r").split(" ")

            max_word_len = -1
            max_sentence_len = -1

            batch_char_X = []
            batch_words_X = []

            for sentence in X:
                if len(sentence) > max_sentence_len:
                    max_sentence_len = len(sentence)
                for word in sentence:
                    if len(word) > max_word_len:
                        max_word_len = len(word)

            for sentence in X:

                words_ids = []
                words_chars_ids = []

                for word in sentence:

                    if word not in word2idx:
                        words_ids.append(word2idx["<UNKNOWN>"])
                    else:
                        words_ids.append(word2idx[word])
                    char_ids = []

                    for c in word:
                        if c not in char2idx:
                            char_ids.append(char2idx['\r'])
                        else:
                            char_ids.append(char2idx[c])

                    words_chars_ids.append(char_ids)

                batch_char_X.append(words_chars_ids)
                batch_words_X.append(words_ids)

            with torch.no_grad():
                tags = []
                output = my_net(batch_words_X, batch_char_X, max_word_len, max_sentence_len)
                output = output.view(max_sentence_len, -1)
                for e in output:
                    tags.append(idx2tag[torch.argmax(e).item()])

            for token, tag in zip(tokens, tags):
                out.write(token + "/" + tag + " ")
            out.write('\n')

    out.close()


    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
