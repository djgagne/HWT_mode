import tensorflow as tf
from tensorflow.keras.regularizers import Regularizer
import tensorflow_probability as tfp


@tf.keras.utils.register_keras_serializable(package='Custom', name='Covariance')
class CovarianceRegularizer(Regularizer):
    """
    The CovarianceRegularizer is designed to minimize the covariance between different features (columns) in a
    batch of data being passed through a neural network. It is meant to be used as an `activity_regularizer` in a
    Tensorflow Dense layer with the goal of making each output of the layer as independent as possible.

    Examples:
        ```
        hidden = 10
        i = Input(shape=(2,))
        d1 = Dense(50, activation="relu", name="d1")(i)
        d2 = Dense(hidden, activity_regularizer=CovarianceRegularizer(scale=100), name="d2")(d1)
        a = Activation("relu")(d2)
        o = Dense(1)(a)
        m = Model(i, o)
        m.compile(optimizer="adam", loss="mse")
        ```
    Attributes:
        scale: A tunable scaling factor for the regularization. A lower factor means the regularizer has less impact on
            the loss.

    """
    def __init__(self, scale=1.0):
        self.scale = scale
        return

    def __call__(self, x):
        x_cov = tfp.stats.covariance(x, name="x_cov")
        x_cov_diag = tf.linalg.diag_part(x_cov, name="x_cov_diag")
        cov_norm = tf.norm(x_cov)
        diag_sum_sq = tf.sqrt(tf.reduce_sum(x_cov_diag ** 2))
        output = self.scale * 0.5 * (cov_norm - diag_sum_sq)
        return output

    def get_config(self):
        return {"scale": float(self.scale)}
