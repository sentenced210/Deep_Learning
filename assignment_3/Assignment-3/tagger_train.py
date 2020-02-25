# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>

import pickle
import sys
from random import uniform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time


class CustomDataset(Dataset):
    def __init__(self, filename):
        X, y = [], []
        with open(filename, "r") as file:

            for line in file:
                pairs = line.rstrip("\n").rstrip("\r").split(" ")

                X1, y1 = [], []

                for pair in pairs:
                    spliting_pair = pair.split("/")

                    word = "/".join(spliting_pair[:len(spliting_pair) - 1]).lower()
                    tag = spliting_pair[len(spliting_pair) - 1]

                    X1.append(word)
                    y1.append(tag)

                X.append(" ".join(X1))
                y.append(" ".join(y1))

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file

    st = time.time()

    tag_idx = 0
    word_idx = 0
    char_idx = 0

    word2idx = dict()
    tag2idx = dict()
    idx2tag = dict()
    char2idx = dict()

    with open(train_file, "r") as file:
        for line in file:
            pairs = line.rstrip("\n").rstrip("\r").split(" ")

            for pair in pairs:
                spliting_pair = pair.split("/")
                word = "/".join(spliting_pair[:len(spliting_pair) - 1]).lower()
                tag = spliting_pair[len(spliting_pair) - 1]

                if word not in word2idx.keys():
                    word2idx[word] = word_idx
                    word_idx += 1

                if tag not in tag2idx.keys():
                    tag2idx[tag] = tag_idx
                    tag_idx += 1

                for char in word:
                    if char not in char2idx.keys():
                        char2idx[char] = char_idx
                        char_idx += 1

    for key in tag2idx:
        idx2tag[tag2idx[key]] = key

    # Unknown word index
    word2idx["<UNKNOWN>"] = word_idx
    word_idx += 1

    # Unknown char index
    char2idx['\r'] = char_idx
    char_idx += 1

    my_net = POS_tag_net(len(word2idx.keys()), len(char2idx.keys()), len(tag2idx.keys()))

    optimezer = optim.Adam(my_net.parameters(), lr=0.001)

    EPOCHS = 5

    p_word_to_unknown = 0.05
    p_char_to_unknown = 0.01

    dataset = CustomDataset(train_file)
    dataloader = DataLoader(dataset, batch_size=32)

    for epoch in range(EPOCHS):
        batch_n = 0
        for X, y in dataloader:
            # X - sentence, y - sentence tags

            X = [e.split(" ") for e in X]
            batch_y = [e.split(" ") for e in y]

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

            batch_norm_y = []

            for y in batch_y:
                y_tags = [tag2idx[e] for e in y]

                while len(y_tags) != max_sentence_len:
                    y_tags.append(-1)

                batch_norm_y.append(y_tags)

            batch_norm_y = torch.tensor(batch_norm_y)

            for sentence in X:

                words_ids = []
                words_chars_ids = []

                for word in sentence:

                    if p_word_to_unknown >= uniform(0, 1):
                        words_ids.append(word2idx["<UNKNOWN>"])
                    else:
                        words_ids.append(word2idx[word])
                    char_ids = []

                    for c in word:
                        if p_char_to_unknown >= uniform(0, 1):
                            char_ids.append(char2idx['\r'])
                        else:
                            char_ids.append(char2idx[c])

                    words_chars_ids.append(char_ids)

                batch_char_X.append(words_chars_ids)
                batch_words_X.append(words_ids)

            my_net.zero_grad()
            output = my_net(batch_words_X, batch_char_X, max_word_len, max_sentence_len)

            loss = my_net.loss(output, batch_norm_y)

            loss.backward()

            optimezer.step()

            batch_n += 1
            print(f"Epoch: {epoch+1}/{EPOCHS} Batch: {batch_n}/{len(dataloader)} loss: {loss}")

        current_time = time.time()
        print(f"======== Epoch_end: {epoch+1}/{EPOCHS} loss: {loss} time: {int((current_time - st) // 60)} min, {current_time - st - 60 * ((current_time - st) // 60)} s  ========")


    model_meta_parameters = {
        "vocab_size": my_net.vocab_size,
        "char_size": my_net.char_size,
        "target_size": my_net.target_size,
        "word_emb_dim": my_net.word_emb_dim,
        "char_emb_dim": my_net.char_emb_dim,
        "lstm_hidden_dim": my_net.lstm_hidden_dim,
        "kernel_size": my_net.kernel_size,
        "out_channels": my_net.out_channels
    }
    torch.save((model_meta_parameters, word2idx, tag2idx, idx2tag, char2idx, my_net.state_dict()), model_file)
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
