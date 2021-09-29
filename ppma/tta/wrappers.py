import time
from typing import Mapping, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader, Dataset

from .base import Compose, Merger


class AverageMeter:
    """Meter for monitoring losses"""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.reset()

    def reset(self):
        """reset all values to zeros"""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """update avg by val and n, where val is the avg of n values"""
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class ClasTTA(nn.Layer):
    def __init__(self, model: nn.Layer, transforms: Compose, merge_mode: str = "mean"):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode

    def forward(self, image: paddle.Tensor):
        merger = Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image)

            deaugmented_output = transformer.deaugment_label(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        return result

    def evaluate(self, dataset: Dataset, batch_size=64):

        data_loader = DataLoader(dataset, batch_size=batch_size)

        val_acc1_meter = AverageMeter()
        self.model.eval()

        with paddle.no_grad():

            for i, (images, target) in enumerate(data_loader):

                start_time = time.perf_counter()
                output = self(images)
                batch_time = time.perf_counter() - start_time

                pred = F.softmax(output)
                acc1 = paddle.metric.accuracy(pred, target.unsqueeze(1))

                batch_size = images.shape[0]
                val_acc1_meter.update(acc1.numpy()[0], batch_size)

                if i % 40 == 0:
                    print(
                        f"[{i:}/{len(data_loader):}]  tta_acc: {val_acc1_meter.avg:.4f}"
                    )
        print(f"Overall tta_acc: {val_acc1_meter.avg:.4f}")
