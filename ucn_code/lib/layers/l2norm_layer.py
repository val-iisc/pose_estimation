import numpy as np
import caffe


class L2NormLayer(caffe.Layer):
    """
    Normalize all features
    """
    def setup(self, bottom, top):
        # Load layer param
        # layer_params = yaml.load(self.param_str)
        pass

    def reshape(self, bottom, top):
        bottom_shape = [x for x in bottom[0].data.shape]
        top[0].reshape(*bottom_shape)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        # Activation from the bottom layers
        bottom_data = bottom[0].data

        # Compute L2 norm for each element
        bottom_sqr = bottom_data ** 2
        bottom_norm_sqr = np.sum(bottom_sqr, axis=1, keepdims=True)
        bottom_norm = np.sqrt(bottom_norm_sqr)

        #print bottom_norm
        top[0].data[...] = bottom_data / bottom_norm

        # Cache the l2 norm and bottom_norm_sqr, bottom_sqr for backward pass
        self.bottom_sqr = bottom_sqr
        self.bottom_norm_sqr = bottom_norm_sqr
        self.bottom_norm = bottom_norm

    def backward(self, top, propagate_down, bottom):
        r"""
        The derivative is the following
        @f[
        \frac{\partial y_j}{\partial x_i} =
        \begin{cases}
            - \frac{\sum_{j \neq i} x_j x_i}{(\sum_j x_j^2)^{3/2}} \text{j \neq i} \\
            - \frac{\sum_{j \neq i} x_j^2}{(\sum_j x_j^2)^{3/2}} \text{i = j}
        \end{cases}
        @f]

        The final gradient of the loss w.r.t. the input is
        \f[\frac{\partial L}{\partial x_i} =
        \frac{1}{\|x\|^3}
         \left[ x_i
          \left(
           \frac{\partial L}{\partial y_i} x_i - \sum_j \frac{\partial L}{\partial y_j} x_j
          \right) + \frac{\partial L}{\partial y_i} (\|x\|^2 - x_i^2)
         \right]
        \f]
        """
        bottom_data = bottom[0].data
        top_diff = top[0].diff

        top_diff_x_bottom_data = top_diff * bottom_data
        bottom[0].diff[...] = (bottom_data * (top_diff_x_bottom_data - np.sum(top_diff_x_bottom_data, axis=1, keepdims=True)) + top_diff * (self.bottom_norm_sqr - self.bottom_sqr)) / (self.bottom_norm ** 3)

