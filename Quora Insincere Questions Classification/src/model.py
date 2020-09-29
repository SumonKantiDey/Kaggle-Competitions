import config
import transformers
import torch
import torch.nn as nn

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
       
        return logits
