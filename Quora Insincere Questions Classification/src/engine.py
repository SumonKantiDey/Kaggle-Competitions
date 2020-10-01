import utils
import torch
import time
import config
import string
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from settings import get_module_logger
logger = get_module_logger(__name__)


def loss_fn(y_pred, y_true):
    return nn.BCEWithLogitsLoss()(y_pred, y_true.view(-1,1)) *len(y_pred)

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    start = time.time()
    avg_train_loss = 0
    avg_train_f1 = 0
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["targets"]
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        # Reset gradients
        #model.zero_grad()
        optimizer.zero_grad()
        # input_ids=None, attention_mask=None, position_ids=None, head_mask=None, inputs_embeds=None,
        outputs = model(
            input_ids=ids,
            attention_mask=mask
        )
        

        train_loss = loss_fn(outputs, targets) / len(outputs)
        train_f1 = utils.f1_score(outputs, targets)
        
        end = time.time()
        f1 = np.round(train_f1.item(), 3)
        if bi % 10 == 0:
            logger.info(f'bi={bi}, Train F1={f1},Train loss={train_loss}, time={end-start}')

        optimizer.step() # Adjust weights based on calculated gradients
        train_loss.backward() # Calculate gradients based on loss
        scheduler.step() # Update scheduler

        losses.update(train_loss.item(), ids.size(0))
        tk0.set_postfix(loss = losses.avg)
    return train_f1, train_loss

def eval_fn(data_loader, model, device):
        model.eval()
        start = time.time()
        losses = utils.AverageMeter()
        val_points = 0
        val_loss = 0
        val_f1 = 0
        temp_val = 0
        with torch.no_grad():
            #tk0 = tqdm(data_loader, total=len(data_loader))
            for bi, d in enumerate(data_loader):
                ids = d["ids"]
                mask = d["mask"]
                targets = d["targets"]
                ids = ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                targets = targets.to(device, dtype=torch.float)

                outputs = model(
                    input_ids=ids,
                    attention_mask=mask,
                )
                
                val_points += len(targets)
                val_loss += loss_fn(outputs, targets) ## add tensor
                val_f1 += utils.f1_score(outputs, targets)*len(outputs)
            end = time.time()
            val_f1 /= val_points
            val_loss /= val_points
            logger.info(f'bi={bi}, Avg_val F1={val_f1.item()}, Avg_val loss={val_loss.item()} ,time={end-start}')
        return val_f1, val_loss