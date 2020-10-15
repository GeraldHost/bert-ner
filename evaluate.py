import numpy as np
import pandas as pd

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel

def process_data(data_path, enc_tags):
    df = pd.read_csv(data_path, encoding="latin-1")

    def str_to_id(s):
        index = np.where(enc_tags == s)[0][0]
        return index
        
    df["tag"] = df["tag"].apply(str_to_id)
    sentences = df.groupby("sentence")["word"].apply(list).values
    tags = df.groupby("sentence")["tag"].apply(list).values
    return sentences, tags

if __name__ == "__main__":
    meta_data = joblib.load("meta.bin")
    enc_tag = meta_data["enc_tag"]

    enc_tags = enc_tag.classes_
    print('enc_tags', enc_tags)
    num_tag = len(enc_tags)

    sentences, tags = process_data(config.EVAL_FILE, enc_tags)

    dataset = dataset.EntityDataset(
        texts=sentences, 
        tags=tags)
    data_loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=config.TRAIN_BATCH_SIZE, 
    num_workers=4)

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    accuracy, loss = engine.eval_fn(data_loader, model, device)
    print(f"Accuracy = {accuracy} Loss = {loss}")