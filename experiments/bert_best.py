import itertools
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from bert_util import *
from bert_preprocessing import preprocessing_for_bert

if __name__ == "__main__":

    # setup torch/cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(1))
        device = torch.device(int(sys.argv[1]))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    train_data = []
    test_data = []
    y_train = []
    y_val = []
    train_inputs = []
    val_inputs = []
    train_masks = []
    val_masks = []

    # get tokenized data
    # first check whether data already exists
    if os.path.exists("/data/s1620444/tm/data/bert_tokenized/bert_train_data.pkl") and \
       os.path.exists("/data/s1620444/tm/data/bert_tokenized/bert_test_data.pkl"):

        with open("/data/s1620444/tm/data/bert_tokenized/bert_train_data.pkl", "rb") as f:
            train_data = pickle.load(f)
        with open("/data/s1620444/tm/data/bert_tokenized/bert_test_data.pkl", "rb") as f:
            test_data = pickle.load(f)


    else:
        print("Must run bert_preprocessing first")
        sys.exit()

    print("Loaded tokenized data")

    # get train & test
    train_inputs, train_masks = preprocessing_for_bert(train_data)
    test_inputs, test_masks = preprocessing_for_bert(test_data)

    map_sent2int = {"negative": 0, "neutral": 1, "positive": 2}
    y_train = np.array([map_sent2int[label] for label in train_data.label.values])
    y_test = np.array([map_sent2int[label] for label in test_data.label.values])


    # run experiments, write results
    train_dataloader, test_dataloader = create_data_loaders(
        train_data, train_inputs, test_inputs, train_masks, test_masks, y_train, y_test)
    loss_fn = nn.CrossEntropyLoss()
    set_seed(42)

    lr = 3.0400000000000004e-05
    epochs = 3
    hidden_layer = False

    for i in trange(3):
        bert_classifier, optimizer, scheduler = initialize_model(device, train_dataloader,
                                                                 lr=lr, epochs=epochs, hidden_layer=hidden_layer)

        train(bert_classifier, train_dataloader, device, loss_fn,
                             optimizer, scheduler, epochs=epochs)

        predictions = bert_predict(bert_classifier, test_dataloader)
        np.save(f"../results/best_predictions_{i}.npy",predictions)

