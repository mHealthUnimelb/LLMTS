import torch
import torch.nn.functional as F
import torch.nn as nn
from .similar_utils import *
from copy import deepcopy

from .losses import mape_loss, mase_loss, smape_loss

loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
}


class cmLoss(nn.Module):
    def __init__(self, feature_loss, output_loss, task_loss, task_name, feature_w=0.01, output_w=1.0, task_w=1.0,
                 temperature=0.07, class_weights=None):
        super(cmLoss, self).__init__()
        self.task_w = task_w
        self.output_w = output_w
        self.feature_w = feature_w
        self.temperature = temperature

        self.feature_loss = loss_dict[feature_loss]
        self.output_loss = loss_dict[output_loss]
        if task_name == "classification" and class_weights is not None:
            self.task_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.task_loss = loss_dict[task_loss]

        self.task_name = task_name

    def clip_contrastive(self, feat_time, feat_text):
        """Compute CLIP loss.

        Args
        ----
        z_time : (B, L_t, D)
        z_text : (B, L_x, D)
        """
        if feat_time.dim() == 3:  # (B, L, D) → (B * L, D)
            B, L, D = feat_time.shape
            feat_time = feat_time.reshape(B * L, D)
            feat_text = feat_text.reshape(B * L, D)
        elif feat_time.dim() == 2:  # already (B, D)
            D = feat_time.size(-1)
        else:
            raise ValueError("Expect shape (B, D) or (B, T, D)")

        # l2‑normalise
        feat_time = F.normalize(feat_time, dim=-1)
        feat_text = F.normalize(feat_text, dim=-1)

        # similarity matrix
        logits = torch.mm(feat_text, feat_time.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        # text to time and time to text directions
        loss_text2time = F.cross_entropy(logits, labels)
        loss_time2text = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_text2time + loss_time2text)


    def forward(self, outputs, batch_y, in_sample=None, freq_map=None, batch_y_mark=None):
        outputs_text, outputs_time, intermidiate_feat_time, intermidiate_feat_text = (
            outputs["outputs_text"],
            outputs["outputs_time"],
            outputs["intermidiate_time"],
            outputs["intermidiate_text"]
        )

        # feture loss
        feature_loss = sum(
            (0.8 ** idx)
            * self.clip_contrastive(
                feat_time, feat_text
            )
            for idx, (feat_time, feat_text) in enumerate(
                zip(intermidiate_feat_time[::-1], intermidiate_feat_text[::-1])
            )
        )

        # output consistency loss
        if self.task_name == "long_term_forecast":
            output_loss = self.output_loss(outputs_time, outputs_text)
        elif self.task_name == "short_term_forecast":
            output_loss = self.output_loss(in_sample, freq_map, outputs_time, outputs_text, batch_y_mark)
        elif self.task_name == "classification":
            output_loss = self.output_loss(outputs_time, outputs_text)
        elif self.task_name == "imputation":
            output_loss = self.output_loss(outputs_time, outputs_text)
        elif self.task_name == "anomaly_detection":
            output_loss = self.output_loss(outputs_time, outputs_text)

        batch_y = batch_y.to(output_loss.device)

        # supervised task loss
        if self.task_name == "long_term_forecast":
            task_loss = self.task_loss(outputs_time, batch_y)
        elif self.task_name == "short_term_forecast":
            task_loss = self.task_loss(in_sample, freq_map, outputs_time, batch_y, batch_y_mark)
        elif self.task_name == "classification":
            task_loss = self.task_loss(outputs_time, batch_y)
        elif self.task_name == "imputation":
            task_loss = self.task_loss(outputs_time, batch_y)
        elif self.task_name == "anomaly_detection":
            task_loss = self.task_loss(outputs_time, batch_y)

        total_loss = self.task_w * task_loss + self.output_w * output_loss + self.feature_w * feature_loss
        print(f"feature loss: {feature_loss}, feature weight: {self.feature_w}")
        print(f"output loss: {output_loss}, output weight: {self.output_w}")
        print(f"task loss: {task_loss}, task weight: {self.task_w}")
        return total_loss