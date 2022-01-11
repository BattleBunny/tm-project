import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Create a function to tokenize a set of texts


def preprocessing_for_bert(data, MAX_LEN=64):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=str(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            padding='max_length',         # Pad sentence to max length
            # return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True      # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


if __name__ == "__main__":
    # Load the BERT tokenizer

    MAX_LEN = 64

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_data = pd.read_csv('./data/downloaded/twitter-train-clean.tsv',
                             sep='\t', names=['label', 'text'], encoding="utf-8")
    test_data = pd.read_csv('./data/downloaded/twitter-test-clean.tsv',
                            sep='\t', names=['label', 'text'], encoding="utf-8")
    train_data.dropna(inplace=True)
    test_data.dropna(inplace=True)

    # remove lines that are too long where something went wrong
    # with reading into pandas
    train_indices = train_data['text'].apply(lambda x: len(
        tokenizer.encode(str(x), add_special_tokens=True)) < MAX_LEN)
    train_data = train_data[train_indices]
    test_indices = test_data['text'].apply(lambda x: len(
        tokenizer.encode(str(x), add_special_tokens=True)) < MAX_LEN)
    test_data = test_data[test_indices]

    with open("data/bert_tokenized/bert_train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open("data/bert_tokenized/bert_test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)

    # encode
    concat = np.concatenate([train_data.text.values, test_data.text.values])
    encoded_tweets = [tokenizer.encode(
        sent, add_special_tokens=True) for sent in concat]

    # split train into train and val
    x_train, x_val, y_train_str, y_val_str = train_test_split(
        train_data.text.values, train_data.label.values, test_size=0.2, random_state=2020)

    # map labels to int
    # map labels to 0,1,2
    map_sent2int = {"negative": 0, "neutral": 1, "positive": 2}
    y_train = np.array([map_sent2int[label] for label in y_train_str])
    y_val = np.array([map_sent2int[label] for label in y_val_str])

    with open("data/bert_tokenized/bert_y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open("data/bert_tokenized/bert_y_val.pkl", "wb") as f:
        pickle.dump(y_val, f)

    # tokenize
    train_inputs, train_masks = preprocessing_for_bert(x_train)
    val_inputs, val_masks = preprocessing_for_bert(x_val)

    with open("data/bert_tokenized/bert_train_inputs.pkl", "wb") as f:
        pickle.dump(train_inputs, f)
    with open("data/bert_tokenized/bert_val_inputs.pkl", "wb") as f:
        pickle.dump(val_inputs, f)
    with open("data/bert_tokenized/bert_train_masks.pkl", "wb") as f:
        pickle.dump(train_masks, f)
    with open("data/bert_tokenized/bert_val_masks.pkl", "wb") as f:
        pickle.dump(val_masks, f)
