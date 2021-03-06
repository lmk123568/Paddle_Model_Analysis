import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.io import DataLoader, Dataset
from PIL import Image

from .utils import AverageMeter, get_val_transforms


class ImageNet2012Dataset(Dataset):
    def __init__(self, data, image_size, crop_pct, normalize):
        super().__init__()
        self.data = data
        self.transforms = get_val_transforms(image_size, crop_pct, normalize)

    def __getitem__(self, idx):

        img_path = self.data[idx][0]
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.transforms(img)

        label = self.data[idx][1]
        label = np.array([label], dtype="int64")
        return img, label

    def __len__(self):
        return len(self.data)


def val(
    model, data_path, batch_size=128, img_size=224, crop_pct=0.875, normalize="default"
):

    data_list = []
    with open(data_path + "/" + "val.txt") as f:
        for line in f:
            a, b = line.strip("\n").split(" ")
            data_list.append([data_path + "/" + a, int(b)])

    val_loader = DataLoader(
        ImageNet2012Dataset(data_list, img_size, crop_pct, normalize),
        batch_size=batch_size,
    )

    model.eval()

    val_acc1_meter = AverageMeter()
    val_acc5_meter = AverageMeter()

    with paddle.no_grad():

        start_time = time.perf_counter()

        for i, (images, target) in enumerate(val_loader):

            output = model(images)

            pred = F.softmax(output)
            acc1 = paddle.metric.accuracy(pred, target)
            acc5 = paddle.metric.accuracy(pred, target, k=5)

            batch_size = images.shape[0]
            val_acc1_meter.update(acc1.numpy()[0], batch_size)
            val_acc5_meter.update(acc5.numpy()[0], batch_size)

            batch_time = time.perf_counter() - start_time

            if i % 40 == 0:
                print(
                    f"[{i: >3}/{len(val_loader):}]  top1_acc: {val_acc1_meter.avg:.4f}  top5_acc: {val_acc5_meter.avg:.4f}  time: {batch_time:.3f}s"
                )

            start_time = time.perf_counter()

    print(
        "Overall  top1_acc: {:.4f}  top5_acc: {:.4f}".format(
            val_acc1_meter.avg, val_acc5_meter.avg
        )
    )
