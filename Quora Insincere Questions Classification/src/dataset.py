import config
import utils
import pandas as pd 
import torch
import numpy as np


class QuoraDataset:
    def __init__(self, question_text, targets):
        self.question_text = question_text
        self.tokenizer = config.TOKENIZER
        self.max_length = config.MAX_LEN
        self.targets = targets

    def __len__(self):
        return len(self.question_text)

    def __getitem__(self, item):
        
        question_text = str(self.question_text[item])
        question_text = " ".join(question_text.split())

        inputs = self.tokenizer.encode_plus(
            question_text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation_strategy="longest_first",
            pad_to_max_length=True,
        )
        
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)
        }


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop = True)
    dset = QuoraDataset(
        question_text=df.question_text.values,
        targets=df.target.values
        )
    print(df.iloc[0])
    print(dset[0])

