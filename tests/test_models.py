import pytest
import paddle

class TestModel():

    def setup_class(self):
        """
        环境初始化
        """
        paddle.set_device('cpu')
        self.dummy_img = paddle.randn([4,3,224,224])
        self.dummy_tensor = paddle.to_tensor(self.dummy_img)
        self.model = paddle.vision.models.resnet50(pretrained=True)

    #@pytest.mark.skip(reason='skip for debug')
    def test_out_shape(self):
        out = self.model(self.dummy_tensor)
        assert out.shape == [4, 1000]

    def teardown_class(self):
        pass
