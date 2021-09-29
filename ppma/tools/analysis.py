import time

import paddle
from paddle.io import DataLoader, Dataset


def params(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_params = int(total_params)
    print("# Params: {:,}".format(total_params))


class test_set(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, idx):
        img = self.data[idx]
        return img

    def __len__(self):
        return len(self.data)


def throughput(model, image_size=224):
    H = W = image_size
    data = paddle.randn([2000, 1, 3, H, W])
    test_loader = DataLoader(test_set(data), batch_size=1)

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    model.eval()
    with paddle.no_grad():
        # inference
        print("Warmup iter 5 ...")

        for i, data in enumerate(test_loader):

            start_time = time.perf_counter()
            model(data[0])
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                pure_inf_time += elapsed

                if i % 100 == 0:
                    infer_speed = (i + 1 - num_warmup) / pure_inf_time
                    print(
                        "[{: >4}/{:}]  infer_speed: {:.1f} img/s ".format(
                            i, len(test_loader), infer_speed
                        )
                    )

    print(f"Overall fps: {infer_speed:.1f} img/s")
