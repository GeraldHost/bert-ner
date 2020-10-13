import torch
from tqdm import tqdm
from seqeval.metrics import accuracy_score

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    final_acc = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        predicted_tags, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()

        acc = accuracy_score(
            data['target_tag'].detach().numpy().reshape(-1),
            predicted_tags.argmax(2).detach().numpy().reshape(-1))
        final_acc += acc
        
    accuracy = final_acc / len(data_loader)
    loss = final_loss / len(data_loader)
    return accuracy, loss


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    final_acc = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        predicted_tags, loss = model(**data)
        final_loss += loss.item()

        acc = accuracy_score(
            data['target_tag'].detach().numpy().reshape(-1),
            predicted_tags.argmax(2).detach().numpy().reshape(-1))
        final_acc += acc

    loss = final_loss / len(data_loader)
    accuracy = final_acc / len(data_loader)
    return accuracy, loss
