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

    def clip_contrastive(self, z_time, z_text):
        """Compute bidirectional CLIP loss for a single layer.

        Args
        ----
        z_time : Tensor  (B, D)
        z_text : Tensor  (B, D)
        """
        # 1. L2‑normalise
        z_time = F.normalize(z_time, dim=-1)
        z_text = F.normalize(z_text, dim=-1)

        # 2. Similarity matrix (text rows × time cols)
        logits = z_text @ z_time.T / self.temperature  # (B, B)

        # 3. Cross‑entropy in both directions
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_txt2time = F.cross_entropy(logits, targets)
        loss_time2txt = F.cross_entropy(logits.T, targets)
        return 0.5 * (loss_txt2time + loss_time2txt)


    def forward(self, outputs, batch_y, in_sample=None, freq_map=None, batch_y_mark=None):
        outputs_text, outputs_time, intermidiate_feat_time, intermidiate_feat_text = (
            outputs["outputs_text"],
            outputs["outputs_time"],
            outputs["intermidiate_time"],
            outputs["intermidiate_text"]
        )

        # feture loss
        feature_losses = []
        for idx, (f_time, f_text) in enumerate(zip(intermidiate_feat_time[::-1], intermidiate_feat_text[::-1])):
            # If features are token‑wise (B, L, D) → pool to 2d vector via mean
            if f_time.dim() == 3:
                f_time = f_time.mean(dim=1)
                f_text = f_text.mean(dim=1)
            layer_loss = (0.8 ** idx) * self.clip_contrastive(f_time, f_text)
            feature_losses.append(layer_loss)
        feature_loss = torch.stack(feature_losses).sum()

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