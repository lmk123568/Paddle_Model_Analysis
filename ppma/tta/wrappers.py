import paddle
import paddle.nn as nn
from typing import Optional, Mapping, Union, Tuple
from .base import Merger, Compose
from paddle.io import Dataset, DataLoader
import time

class ClassTTA(nn.Layer):

    def __init__(self,
                 model : nn.Layer, 
                 transforms : Compose,  
                 merge_mode : str = "mean"):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode


    def forward(self, image : paddle.Tensor):
        merger = Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image)

            deaugmented_output = transformer.deaugment_label(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        return result

    def evaluate(self, dataset : Dataset, batch_size=64):

        data_loader = DataLoader(dataset, batch_size=batch_size)

        m = paddle.metric.Accuracy()
        self.model.eval()

        with paddle.no_grad():

            end = time.time()
            for i, (images, target) in enumerate(data_loader):
                
                output = self(images)

                m.update(m.compute(output, target))
                acc = m.accumulate()

                batch_time = time.time() - end
                
                if i % 40 == 0:
                    print("[{:}/{:}]  tta_acc {:.3f}  time {:.3f}s  batch {:}".format(i, len(data_loader), acc, batch_time, batch_size))

                end = time.time()

        print("{{TTA acc {:.3f}}}".format(m.accumulate()))    










