import utils
import config
import dataset
import engine
import torch
#torch.cuda.empty_cache()
# print(torch.cuda.memory_summary())
import transformers
import pandas as pd
import torch.nn as nn
import numpy as np
from settings import get_module_logger
from model import CustomRoberta
from sklearn import model_selection
from transformers import AdamW
from dataset import QuoraDataset
from transformers import get_linear_schedule_with_warmup
import gc
logger = get_module_logger(__name__)


def run():
    dfx = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)
    dfx = pd.concat([
    dfx[['question_text', 'target']].query('target==1').sample(n=20000, random_state=42),
    dfx[['question_text', 'target']].query('target==0').sample(n=20000, random_state=42)
    ]).reset_index(drop=True)
    df_train, df_valid = model_selection.train_test_split(
        dfx, 
        test_size=0.1, 
        random_state=42, 
        stratify=dfx.target.values
    )
    logger.info("train len - {} valid len - {}".format(len(df_train), len(df_valid)))

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = QuoraDataset(
        question_text=df_train.question_text.values,
        targets=df_train.target.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = QuoraDataset(
        question_text=df_valid.question_text.values,
        targets=df_valid.target.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )
    
    device = torch.device("cuda")
    model = CustomRoberta()
    model.to(device)
   

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # Define two sets of parameters: those with weight decay, and those without
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

    num_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=2e-5)
    '''
    Create a scheduler to set the learning rate at each training step
    "Create a schedule with a learning rate that decreases linearly after linearly increasing during a warmup period." (https://pytorch.org/docs/stable/optim.html)
    Since num_warmup_steps = 0, the learning rate starts at 3e-5, and then linearly decreases at each training step
    '''
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    #es = utils.EarlyStopping(patience=15, mode="max")
    print("STARTING TRAINING ...\n")
    logger.info("{}".format("STARTING TRAINING"))
    #model=nn.DataParallel(model)
    val_losses, val_f1s = [], []
    train_losses, train_f1s = [], []
    for epoch in range(config.EPOCHS):
        logger.info(f"epochs = {epoch}")
        train_f1, train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        val_f1, val_loss = engine.eval_fn(valid_data_loader, model, device)
        val_f1s.append(val_f1.item()); train_f1s.append(train_f1.item())
        val_losses.append(val_loss.item()); train_losses.append(train_loss.item())
        # es(val_f1s.item(), model, model_path=f"../models/nmodel.bin")
        # if es.early_stop:
        #     print("Early stopping")
        #     break

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    del model, train_data_loader, valid_data_loader, train_dataset, valid_dataset
    print(gc.collect())
    metric_lists = [val_losses, train_losses, val_f1s, train_f1s]
    metric_names = ['val_loss_', 'train_loss_', 'val_f1_', 'train_f1_']
    
    for i, metric_list in enumerate(metric_lists):
        for j, metric_value in enumerate(metric_list):
            path = '../models/'+  metric_names[i] + str(j) + '.pt'
            # print(metric_names[i] + str(j), metric_value)
            torch.save(metric_value,path)
if __name__ == "__main__":
    run()