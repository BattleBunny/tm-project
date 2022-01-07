import itertools
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from bert_util import *

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
    if os.path.exists("data/bert_tokenized/bert_train_data.pkl") and \
       os.path.exists("data/bert_tokenized/bert_test_data.pkl") and \
       os.path.exists("data/bert_tokenized/bert_y_train.pkl") and \
       os.path.exists("data/bert_tokenized/bert_y_val.pkl") and \
       os.path.exists("data/bert_tokenized/bert_train_inputs.pkl") and \
       os.path.exists("data/bert_tokenized/bert_train_masks.pkl") and \
       os.path.exists("data/bert_tokenized/bert_val_inputs.pkl") and \
       os.path.exists("data/bert_tokenized/bert_val_masks.pkl"):

        with open("data/bert_tokenized/bert_train_data.pkl", "rb") as f:
            train_data = pickle.load(f)
        with open("data/bert_tokenized/bert_test_data.pkl", "rb") as f:
            test_data = pickle.load(f)
        with open("data/bert_tokenized/bert_y_train.pkl", "rb") as f:
            y_train = pickle.load(f)
        with open("data/bert_tokenized/bert_y_val.pkl", "rb") as f:
            y_val = pickle.load(f)
        with open("data/bert_tokenized/bert_train_inputs.pkl", "rb") as f:
            train_inputs = pickle.load(f)
        with open("data/bert_tokenized/bert_val_inputs.pkl", "rb") as f:
            val_inputs = pickle.load(f)
        with open("data/bert_tokenized/bert_train_masks.pkl", "rb") as f:
            train_masks = pickle.load(f)
        with open("data/bert_tokenized/bert_val_masks.pkl", "rb") as f:
            val_masks = pickle.load(f)

    else:
        print("Must run bert_preprocessing first")
        sys.exit()

    print("Loaded tokenized data")

    # run experiments, write results
    train_dataloader, val_dataloader = create_data_loaders(
        train_data, train_inputs, val_inputs, train_masks, val_masks, y_train, y_val)
    loss_fn = nn.CrossEntropyLoss()
    set_seed(42)

    lrs = np.linspace(1e-6, 5e-5, 6)
    epoch_range = [3]
    hidden_layers = [True, False]

    param_combinations = list(itertools.product(
        lrs, epoch_range, hidden_layers))

    for params in tqdm(param_combinations):
        lr = params[0]
        epochs = params[1]
        hidden_layer = params[2]

        for i in trange(3):
            bert_classifier, optimizer, scheduler = initialize_model(device, train_dataloader,
                                                                     lr=lr, epochs=epochs, hidden_layer=hidden_layer)

            t_acc, v_acc = train(bert_classifier, train_dataloader, device, loss_fn,
                                 optimizer, scheduler, val_dataloader=val_dataloader, epochs=epochs)

            with open(f"results/{lr}_{hidden_layer}.csv", "a") as f:
                f.write(f'{t_acc},{v_acc}\n')
