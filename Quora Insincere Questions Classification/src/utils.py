import torch 
import torch.nn as nn
from settings import get_module_logger
import numpy as np
import time
from colored import fg, bg, attr
logger = get_module_logger(__name__)

def f1_score(y_pred, y_true):
    y_true = y_true.squeeze()
    y_pred = torch.round(nn.Sigmoid()(y_pred)).squeeze()
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    recall = tp / (tp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    return 2*(precision*recall) / (precision + recall + epsilon)


def print_metric(data, batch, epoch, start, end, metric, typ):
    t = typ, metric, "%s", data, "%s"
    if typ == "Train": pre = "BATCH %s" + str(batch) + "%s  "
    if typ == "Val": pre = "\nEPOCH %s" + str(epoch) + "%s  "
    time = np.round(end - start, 1); time = "Time: %s{}%s s".format(time)
    fonts = [(fg(211), attr('reset')), (fg(212), attr('reset')), (fg(213), attr('reset'))]
    logger.info(pre % fonts[0] + "{} {}: {}{}{}".format(*t) % fonts[1] + "  " + time % fonts[2])

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            logger.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            logger.info('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score