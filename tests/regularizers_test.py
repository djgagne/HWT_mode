import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model
from hwtmode.regularizers import CovarianceRegularizer
import numpy as np


def test_covariance_regularizer():
    tf.random.set_seed(1243)
    x = tf.random.normal(stddev=2.0, shape=(128, 3))
    x_ones = tf.ones(shape=(128, 3))
    cr = CovarianceRegularizer(scale=1)
    output = cr(x)
    output_ones = cr(x_ones)
    print("Output ones", output_ones)
    assert len(output.shape) == 0
    assert output > 0
    return


def test_convariance_network():
    hidden = 10
    i = Input(shape=(2,))
    d1 = Dense(50, activation="relu", name="d1")(i)
    d2 = Dense(hidden, activity_regularizer=CovarianceRegularizer(scale=100000.0), name="d2")(d1)
    a = Activation("relu")(d2)
    o = Dense(1)(a)
    m = Model(i, o)
    m.compile(optimizer="adam", loss="mse")
    x = np.random.normal(size=(1024, 2))
    y = x[:, 0] ** 2 + 2 * x[:, 1]
    m_sub = Model(m.input, m.layers[-3].output)
    pred_start = m_sub.predict(x)
    l_out_first = np.cov(pred_start.T)
    r_matrix_start = np.corrcoef(pred_start.T)
    m.fit(x, y, batch_size=64, epochs=100)
    l_out = np.cov(m_sub.predict(x).T)
    assert l_out.shape[1] == hidden
    start_reg = np.linalg.norm(l_out_first) - np.sqrt(np.sum(np.diag(l_out_first) ** 2))
    end_reg = np.linalg.norm(l_out) - np.sqrt(np.sum(np.diag(l_out) ** 2))
    print("Start reg", start_reg)
    print("End reg", end_reg)
    assert end_reg < start_reg

    r_matrix = np.corrcoef(m_sub.predict(x).T)
    for (r, c), m in np.ndenumerate(r_matrix):
        if c == r_matrix.shape[1] - 1:
            print(f"{m:+0.3f} ")
        else:
            print(f"{m:+0.3f} ", end="")
    print()
    for (r, c), m in np.ndenumerate(r_matrix_start):
        if c == r_matrix.shape[1] - 1:
            print(f"{m:+0.3f} ")
        else:
            print(f"{m:+0.3f} ", end="")


if __name__ == '__main__':
    test_covariance_regularizer()
    test_convariance_network()