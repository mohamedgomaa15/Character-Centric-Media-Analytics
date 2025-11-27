from transformers import Trainer
from torch import nn
import torch

class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        labels = inputs.get("labels")

        output = model(**inputs)
        logits = output.get("logits").float()

        loss = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights, dtype=torch.float).to(device=self.device))
        loss_out = loss(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, output) if return_outputs else loss_out
    
    def set_class_weights(self,class_weights):
        self.class_weights = class_weights
    
    def set_device(self,device):
        self.device = device


