import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel


def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")

    enc_tag = preprocessing.LabelEncoder()
    enc_str = np.append(df.tag.unique(), "B-START")
    enc_ids = enc_tag.fit_transform(enc_str)
    def str_to_id(s):
        index = np.where(enc_str == s)
        return enc_ids[index]
        
    df["tag"] = df["tag"].apply(str_to_id)

    sentences = df.groupby("sentence")["word"].apply(list).values
    tag = df.groupby("sentence")["tag"].apply(list).values
    return sentences, tag, enc_tag


if __name__ == "__main__":
    sentences, tag, enc_tag = process_data(config.TRAINING_FILE)
    
    meta_data = {
        "enc_tag": enc_tag
    }

    joblib.dump(meta_data, "meta.bin")

    num_tag = len(list(enc_tag.classes_))
    print(list(enc_tag.classes_))

    (
        train_sentences,
        test_sentences,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, tag, random_state=42, test_size=0.1)

    train_dataset = dataset.EntityDataset(
        texts=train_sentences, tags=train_tag
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.EntityDataset(
        texts=test_sentences, tags=test_tag
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cpu")
    model = EntityModel(num_tag=num_tag)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_accuracy, train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_accuracy, test_loss = engine.eval_fn(valid_data_loader, model, device)
        
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        print(f"Train Accuracy = {train_accuracy} Valid Accuracy = {test_accuracy}")

        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss
