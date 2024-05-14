# Custom Model with all the sigmoid outputs
from torch import nn
import torch
from transformers import Swinv2Model
from builder import MCSBuilder

model_name = "microsoft/swinv2-tiny-patch4-window8-256"


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        classes = MCSBuilder.classes

        self.num_labels = len(classes)
        self.labels = classes
        self.id2label = {i: c for i, c in enumerate(classes)}
        self.label2id = {c: i for i, c in enumerate(classes)}

        self.base_model = Swinv2Model.from_pretrained(
            model_name, add_pooling_layer=True
        )

        for name, param in self.base_model.named_parameters():
            if not (
                "layers.3" in name
                or "layers.2" in name
                or "layernorm.weight" == name
                or "layernorm.bias" == name
            ):
                param.requires_grad = False

        pooled_dim = (
            self.base_model.num_features
            if self.base_model.pooler is not None
            else self.base_model.config.hidden_size
        )

        self.heads = {}

        # This will create a separate head for each label
        # We don't have to go down to one dimension. We can also use a CLS token
        # Originally I ended with a sigmoid but read that it was more stable to compute loss with logits
        self.heads = nn.ModuleDict(
            {label: nn.Linear(pooled_dim, 1) for label in self.labels}
        )

    def forward(self, pixel_values=None, labels=None):
        """
        Labels should be a Tensor N by # labels where N is the batch size
        It should be in the same order the labels were originally passed to the model

        This will return a dictionary with key loss and the logits which will be an N by # labels tensor
        """
        # First pass the inputs through the original model
        base_out = self.base_model(pixel_values)
        # base.last_hidden_state gave a non-pooled output we can't operate on all the pathches
        # therefore I am using the pooled output
        pooled_output = base_out.pooler_output

        device = pixel_values.device

        # neg_counts / pos_counts
        pos_weight = torch.tensor(
            [
                9.8272,
                9.7402,
                76.2712,
                35.5047,
                66.9026,
                46.3112,
                34.0782,
                10.1667,
                0.5353,
                8.2407,
                108.0283,
                16.7097,
                39.7754,
                14.0982,
            ]
        ).to(device)

        loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        total_loss = 0
        head_outs = []
        for l in self.labels:
            head_outs.append(self.heads[l](pooled_output).view(-1, 1))

        logits = torch.cat(head_outs, dim=1)
        total_loss = loss_func(logits, labels)

        # # Pass the final output into all the heads
        # head_outs = {}
        # for l in self.labels:
        #     head_outs[l] = self.heads[l](pooled_output)

        # # Get the loss for all the outputs
        # # I am using BCE with logits instead of BCE loss and manually applying the sigmoid because it
        # # should be moe stable on the backprop
        # loss_func = nn.BCEWithLogitsLoss()
        # total_loss = 0
        # for l in self.labels:
        #     total_loss += loss_func(
        #         head_outs[l], labels[:, self.label2id[l] : self.label2id[l] + 1]
        #     )

        # # Combine all outputs into a single tensor
        # # The device part of this is not tested
        # logits = torch.zeros(
        #     (pixel_values.shape[0], self.num_labels),
        #     dtype=torch.float,
        #     device=pooled_output.device,
        # )
        # for l in self.labels:
        #     logits[:, self.label2id[l] : self.label2id[l] + 1] = head_outs[l]

        result = {"logits": logits}
        result["loss"] = total_loss

        # These next two lines are a possible way to speed up computation however I am neervous
        # because I do not know how the loss function will act with mutiple labels as different columns
        # I think this will work but I wanted to give the loss some more thought for reduction
        # logits = torch.cat([head_outs[l] for l in self.labels], dim=1)  # Assuming the order of self.labels matches labels
        # total_loss = loss_func(logits, labels)
        # loss = total_loss.sum(dim=1).mean() # Sums across labels, averages across the batch
        return result
