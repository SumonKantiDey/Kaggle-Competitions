import transformers
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig
from transformers import TFAutoModel, AutoTokenizer
import numpy as np
class CustomRoberta(nn.Module):
    def __init__(self):
        super(CustomRoberta, self).__init__()
        self.num_labels = 1
        self.roberta = transformers.XLMRobertaModel.from_pretrained("xlm-roberta-base", output_hidden_states=False, num_labels=1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(768*2, self.num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None):

        o1, o2 = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        
        apool = torch.mean(o1, 1)
        mpool, _ = torch.max(o1, 1)
        cat = torch.cat((apool, mpool), 1)
        bo = self.dropout(cat)
        logits = self.classifier(bo)       
        outputs = logits
        return outputs




model = CustomRoberta()
model.load_state_dict(torch.load('insincerity_model.pt'),strict=False)

device = torch.device("cuda")
model = model.to(device)
model = model.eval()


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


def predict_insincerity(question_text, max_length):
    question_text = str(question_text)
    question_text = " ".join(question_text.split())
    inputs = tokenizer.encode_plus(
        question_text,
        None,
        add_special_tokens=True,
        max_length=max_length,
        truncation_strategy="longest_first",
        pad_to_max_length=True,
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    ids = torch.LongTensor(ids)
    mask = torch.LongTensor(mask)
    ids = torch.reshape(ids, (1,64))
    mask = torch.reshape(mask, (1,64))
    #print(ids.shape)
    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    outputs = model(
            input_ids=ids,
            attention_mask=mask,
        )
    ins = {0: 'sincere', 1: 'insincere'}
    output = ins[int(np.round(nn.Sigmoid()(outputs.detach().cpu()).item()))][0:]
    return {
        output
    }

