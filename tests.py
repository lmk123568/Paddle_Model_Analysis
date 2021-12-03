import ppma
import paddle

def test_1():
    model = paddle.vision.models.resnet50(pretrained=True)
    data_path = "data/ILSVRC2012"	                        

    ppma.imagenet.val(model, data_path, batch_size=128 ,img_size=224, crop_pct=0.875, normalize='default')

def test_2():
    img_path = 'source/test.jpg'   
    model = paddle.vision.models.resnet50(pretrained=True)

    ppma.imagenet.test_img(model, img_path, img_size=224, crop_pct=0.875, normalize='default')

def test_3():

    res50 = paddle.vision.models.resnet50()

    # FLOPs、Params -- depend model and resolution
    ppma.modelstat.flops(model=res50, img_size=224, detail=True)

    # Thoughput -- depend model and resolution
    ppma.modelstat.throughput(model=res50, img_size=224)

if __name__ == '__main__':
    test_1()
    test_2()
    test_3()

