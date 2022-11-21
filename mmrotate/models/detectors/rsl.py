# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmrotate.core import rbbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from mmrotate.models.dense_heads.rsl_saf_v6 import *
from mmrotate.models.dense_heads.rsl_msm_v2 import *
from mmrotate.models.dense_heads.rsl_ssf_v2 import *
from .base import RotatedBaseDetector


@ROTATED_DETECTORS.register_module()
class RSLDet(RotatedBaseDetector):
    """Base class for rotated single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 ssf_cfg=None,
                 saf_cfg=None,
                 msm_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RSLDet, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        assert ssf_cfg is not None, "ssf_cfg is None!"
        assert saf_cfg is not None, "saf_cfg is None!"
        assert msm_cfg is not None, "msm_cfg is None!"
        self.ssf_version = ssf_cfg.pop('version')
        self.ssf = eval('SemanticAlignmentFusion_{}'.format(self.ssf_version))(**ssf_cfg) if self.ssf_version != 'v0' else None
        self.saf_version = saf_cfg.pop('version')
        # saf_cfg.update(train_cfg=train_cfg)
        # saf_cfg.update(test_cfg=test_cfg)
        self.saf = eval("SpatialAlignmentFusion_{}".format(self.saf_version))(**saf_cfg) if self.saf_version != 'v0' else None
        self.msm_version = msm_cfg.pop('version')
        self.msm = eval("MultiScaleModule_{}".format(self.msm_version))(**msm_cfg) if self.msm_version != 'v0' else None

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        super(RSLDet, self).forward_train(img, img_metas)
        x = self.extract_feat(img)

        x = self.msm(x) if self.msm else x
        if self.saf:
            if self.saf_version == 'alignconv' or self.saf_version == 'axis_sup':
                x, loss_align = self.saf(x, (gt_bboxes, gt_labels, img_metas))
            else:
                x = self.saf(x)
        x = self.ssf(x) if self.ssf else x
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        if self.saf_version == 'alignconv' or self.saf_version == 'axis_sup':
            losses.update(loss_align)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        x = self.extract_feat(img)

        x = self.msm(x)
        x = self.saf(x) if self.saf else x
        x = self.ssf(x) if self.ssf else x

        outs = self.bbox_head(x)

        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

