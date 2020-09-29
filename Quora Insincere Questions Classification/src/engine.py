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
    return nn.BCEWithLogitsLoss()(y_pred, y_true.view(-1,1))*len(y_pred)

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    start = time.time()
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["targets"]
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        # Reset gradients
        model.zero_grad()
        # input_ids=None, attention_mask=None, position_ids=None, head_mask=None, inputs_embeds=None,
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
        )
        
        train_loss = loss_fn(outputs, targets)/len(outputs)
        train_f1 = utils.f1_score(outputs, targets)


        train_loss.backward() # Calculate gradients based on loss
        optimizer.step() # Adjust weights based on calculated gradients
        scheduler.step() # Update scheduler
        
        end = time.time()
        f1 = np.round(train_f1.item(), 3)
        if bi % 100 == 0:
                logger.info(f'bi={bi}, F1={f1},time={end-start}')
        losses.update(train_loss.item(), ids.size(0))
        tk0.set_postfix(loss = losses.avg)

    return train_f1, train_loss
                