from enum import Enum
from typing import Optional

import onegan
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from kornia.filters import sobel
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau, ConstantLR
from torchvision.utils import make_grid

from .model import ResPlanarSeg, ResPlanarSegTimm, ResPlannerSegSMP


class ModelTypes(Enum):
    m_timm = 'timm'
    m_original = 'original'
    m_smp = 'smp'


class LayoutSeg(pl.LightningModule):

    def __init__(
            self,
            backbone: str,
            num_classes: int,
            lr: float = 1e-4,
            l1_factor: float = 0.2,
            l2_factor: float = 0.0,
            edge_factor: float = 0.2,
            input_ch: int = 3,
            model_type: ModelTypes = ModelTypes.m_original,
            pretrained_backbone: bool = False,
            smp_arch: Optional[str] = None,
            log_freq_img: int = 100,
    ):
        super().__init__()
        self.lr = lr
        self.log_freq_img = log_freq_img
        if model_type == ModelTypes.m_timm:
            self.model = ResPlanarSegTimm(
                num_classes, backbone, pretrained=pretrained_backbone,
                input_ch=input_ch)
            logger.info('Use model with timm backbone')
        elif model_type == ModelTypes.m_original:
            self.model = ResPlanarSeg(
                num_classes, pretrained=True, backbone=backbone,
                input_ch=input_ch)
            logger.info('Use model with torchvision backbone')
        elif model_type == ModelTypes.m_smp:
            assert smp_arch is not None, 'When using SMP model set param arch'
            self.model = ResPlannerSegSMP(
                num_classes, smp_arch, backbone=backbone,
                pretrained=pretrained_backbone, input_ch=input_ch)
            logger.info(f'Use model from SMP. Arch: {smp_arch} with encoder '
                        f'{backbone}')
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.edge_factor = edge_factor
        self.num_classes = num_classes
        self.save_hyperparameters()

    def forward(self, inputs):
        scores = self.model(inputs)
        _, outputs = torch.max(scores, 1)
        return scores, outputs

    def training_step(self, batch, batch_idx):
        inputs = batch['image']
        targets = batch['label']
        scores, outputs = self(inputs)
        loss_terms = self.criterion(scores, outputs, targets, batch)

        if self.global_step % self.log_freq_img == 0:
            self.logger.experiment.add_image(
                'train_input', make_grid(inputs, nrow=4, normalize=True), self.global_step)
            self.logger.experiment.add_image(
                'train_prediction', make_grid(label_as_rgb_visual(outputs), nrow=4), self.global_step)
            self.logger.experiment.add_image(
                'train_target', make_grid(label_as_rgb_visual(targets), nrow=4), self.global_step)
            self.logger.experiment.add_image(
                'train_edge',
                make_grid(batch['edge'].unsqueeze(1), nrow=4, normalize=True),
                self.global_step)

        loss = loss_terms['loss']
        loss_terms = {f'train_{k}': v.detach().cpu() for k, v in loss_terms.items()}
        self.log_dict(loss_terms, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch['image']
        targets = batch['label']
        scores, outputs = self(inputs)

        metric_terms = self.metric(outputs, targets, self.num_classes)
        self.log_dict(metric_terms, logger=True)

        loss_terms = self.criterion(scores, outputs, targets, batch)
        loss = loss_terms['loss']
        loss_terms = {f'val_{k}': v for k, v in loss_terms.items()}
        self.log_dict(loss_terms, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch['image']
        targets = batch['label']
        _, outputs = self(inputs)

        metric_terms = self.metric(outputs, targets, self.num_classes)
        self.log('score', metric_terms['score'])
        return metric_terms

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': ReduceLROnPlateau(optimizer),
            'lr_scheduler': ConstantLR(optimizer),
            'monitor': 'val_loss'
        }

    def criterion(self, score, prediction, target, data):
        def layout_gradient(output, σ=5.0):
            return 1 - torch.exp(-sobel(output.unsqueeze(1).float()) / σ)

        loss = 0
        terms = {}
        ''' per-pixel classification loss '''
        seg_loss = F.nll_loss(F.log_softmax(score, dim=1), target, ignore_index=255)
        loss += seg_loss
        terms['loss_cla'] = seg_loss

        ''' area smoothness loss '''
        if self.l1_factor or self.l2_factor:
            l_loss = F.mse_loss if self.l2_factor else F.l1_loss
            l1_λ = self.l1_factor or self.l2_factor
            # TODO ignore 255
            onehot_target = torch.zeros_like(score).scatter_(1, target.unsqueeze(1), 1)
            l1_loss = l_loss(score, onehot_target)
            loss += l1_loss * l1_λ
            terms['loss_area'] = l1_loss

        ''' layout edge constraint loss '''
        if self.edge_factor:
            edge_map = layout_gradient(prediction).squeeze(1)
            target_edge = data['edge'].to(device=edge_map.device)
            edge_loss = F.binary_cross_entropy(edge_map, target_edge)
            loss += edge_loss * self.edge_factor
            terms['loss_edge'] = edge_loss

        terms['loss'] = loss
        return terms

    def metric(self, output, target, num_classes):
        seg_metric = onegan.metrics.semantic_segmentation.Metric(
            num_class=num_classes, only_scalar=True)
        score_metric = onegan.metrics.semantic_segmentation.max_bipartite_matching_score
        accuracies = seg_metric(output, target)
        score = score_metric(output, target)
        return {**accuracies, 'score': score}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items


def label_as_rgb_visual(x):
    """ Make segment tensor into colorful image
    Args:
        x (torch.Tensor): shape in (N, H, W) or (N, 1, H, W)
        colors (tuple or list): list of RGB colors, range from 0 to 1.
    Returns:
        canvas (torch.Tensor): colorized tensor in the shape of (N, C, H, W)
    """
    colors = [
        [0.9764706, 0.27058825, 0.3647059], [1., 0.8980392, 0.6666667],
        [0.5647059, 0.80784315, 0.70980394], [0.31764707, 0.31764707, 0.46666667],
        [0.94509804, 0.96862745, 0.8235294]]

    if x.dim() == 4:
        x = x.squeeze(1)
    assert x.dim() == 3

    n, h, w = x.size()
    palette = torch.tensor(colors).to(x.device)
    labels = torch.arange(x.max() + 1).to(x)

    canvas = torch.zeros(n, h, w, 3).to(x.device)
    for color, lbl_id in zip(palette, labels):
        if canvas[x == lbl_id].size(0):
            canvas[x == lbl_id] = color

    return canvas.permute(0, 3, 1, 2)
