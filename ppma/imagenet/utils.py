import math

from paddle.vision import transforms


def get_val_transforms(img_size=224, crop_pct=0.875, normalize="default"):

    if normalize == "inception":
        val_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif normalize == "default":
        val_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        raise ValueError(f'Invalid keywords "{normalize}"')

    scale_size = int(math.floor(img_size / crop_pct))
    transforms_val = transforms.Compose(
        [
            transforms.Resize(
                scale_size, "bicubic"
            ),  # single int for resize shorter side of image
            transforms.CenterCrop((img_size, img_size)),
            transforms.ToTensor(),
            val_normalize,
        ]
    )
    return transforms_val


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
