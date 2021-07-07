import cv2
import numpy as np
import paddle
from .utils import img_to_tensor

class BaseCAM:
    def __init__(self, 
                 model, 
                 target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = []

    def hook(self, layer, input, output):
        self.feature_maps.append(output)
    
    def get_cam_weights(self,
                        input_tensor,
                        label,
                        activations,
                        grads):
        raise Exception("Not Implemented")
        
        
    def get_cam_image(self,
                      input_tensor,
                      label,
                      activations,
                      grads,):
        weights = self.get_cam_weights(input_tensor, label, activations, grads)
        weighted_activations = weights[:, :, None, None] * activations

        cam = weighted_activations.sum(axis=1)

        return cam


    def forward(self, img_path, label):
        
        img_tensor = img_to_tensor(img_path)
        
        self.model.eval()
        self.model.clear_gradients()
        
        self.target_layer.register_forward_post_hook(self.hook)
        
        out = self.model(img_tensor)
        out = paddle.nn.functional.softmax(out, axis=1)
        
        if label is None:
            label = out.argmax(axis=1).numpy()
            
        label_onehot = paddle.nn.functional.one_hot(paddle.to_tensor(label), num_classes=out.shape[1])
        target = paddle.sum(out * label_onehot, axis=1)

        gradients = paddle.grad(outputs=[target], inputs=[self.feature_maps[0]])[0]

        activations = self.feature_maps[0].numpy()
        grads = gradients.numpy()
        

        cam = self.get_cam_image(img_tensor, label, activations, grads)

        cam = np.maximum(cam, 0)

        result = []
        for img in cam:
            img = cv2.resize(img, (224, 224))
            img = img - np.min(img)
            img = img / np.max(img)
            result.append(img)
        result = np.float32(result)
        return result[0]
    
    
    def __call__(self,
                 input_tensor,
                 label = None):

        return self.forward(input_tensor, label)    

class GradCAM(BaseCAM):
    def __init__(self, model, target_layer):
        super(GradCAM, self).__init__(model, target_layer)

    def get_cam_weights(self,
                        input_tensor,
                        label,
                        activations,
                        grads):
        return np.mean(grads, axis=(2, 3))
    
    
class GradCAMPlusPlus(BaseCAM):
    def __init__(self, model, target_layer):
        super(GradCAMPlusPlus, self).__init__(model, target_layer)

    def get_cam_weights(self, input_tensor, 
                              label, 
                              activations, 
                              grads):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2*grads

        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2*grads_power_2 + 
            sum_activations[:, :, None, None]*grads_power_3 + eps)

        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0)*aij
        weights = np.sum(weights, axis=(2, 3))
        return weights


class XGradCAM(BaseCAM):
    def __init__(self, model, target_layer):
        super(XGradCAM, self).__init__(model, target_layer)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        weights = grads * activations / (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2, 3))
        return weights