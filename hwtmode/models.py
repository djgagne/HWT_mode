from tensorflow.keras.layers import Dense, Conv2D, Activation, Input, Flatten, AveragePooling2D, MaxPool2D, LeakyReLU
from tensorflow.keras.layers import SpatialDropout2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error, binary_crossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from .regularizers import CovarianceRegularizer
import tensorflow as tf
import numpy as np
from tqdm import trange, tqdm
import pandas as pd
import xarray as xr
from os.path import join
import yaml

losses = {"mse": mean_squared_error,
          "mae": mean_absolute_error,
          "binary_crossentropy": binary_crossentropy}


class BaseConvNet(object):
    """
    Convolutional Neural Network consisting of sequential convolution and pooling layers. The class is built on
    Tensorflow 2/Keras under the hood but uses the scikit-learn design paradigm to enable greater flexibility in
    evaluating hyperparameters.

    Attributes:
        min_filters (int): Minimum number of convolutional filters
        filter_growth_rate (float): Factor to scale filter count after each layer.
        filter_width (int): Width of square convolutional filter.
        min_data_width (int): Output of conv->pooling layer combo is flattened after the data width reaches this
            threshold.
        pooling_width (int): Factor by which pooling should reduce the spatial dimensions of the input.
        hidden_activation (str): Activation function used after each convolution and dense hidden layer. leaky produces leaky relu
        output_type: (str): Either linear (regression), sigmoid (binary classification), or softmax (multiclass)
        pooling (str): Either max or mean
        use_dropout (bool): If True or 1, include SpatialDropout and regular Dropout layers
        dropoout_alpha (float): Dropout relative frequency between 0 and 1.
        dense_neurons (int): Number of neurons in dense hidden layer. Used as information bottleneck for interpretation.
        data_format (str): channels_last (default) or channels_first.
        optimizer (str): Supports adam or sgd
        loss (str): Supports mse, mae, or binary_crossentropy
        leaky_alpha (float): If leaky activation is used, this controls the scaling factor for the leaky ReLU
        metrics (list): List of additional metrics to calculate during training.
        learning_rate (float): Learning rate for optimizer
        batch_size (int): Number of examples per batch
        verbose (int): Level of verbosity in fit loop. 1 results in a progress bar and 2 prints loss for each batch.
        l2_alpha (float): if l2_alpha > 0 then l2 regularization with strength l2_alpha is used.
        early_stopping (int): If > 0, then early stopping of training is triggered when validation loss does not change
            for early_stopping epochs.
        covariance_scale (float): if > 0, then a covariance regularizer is applied to the activation of the
            last hidden layer to promote more independent activations.

    """

    def __init__(self, min_filters=16, filter_growth_rate=2, filter_width=5, min_data_width=4, pooling_width=2,
                 hidden_activation="relu", output_type="linear",
                 pooling="mean", use_dropout=False, dropout_alpha=0.0, dense_neurons=64,
                 data_format="channels_last", optimizer="adam", loss="mse", leaky_alpha=0.1, metrics=None,
                 learning_rate=0.0001, batch_size=1024, epochs=10, verbose=0, l2_alpha=0, early_stopping=0,
                 covariance_scale=0, **kwargs):
        self.min_filters = min_filters
        self.filter_width = filter_width
        self.filter_growth_rate = filter_growth_rate
        self.pooling_width = pooling_width
        self.min_data_width = min_data_width
        self.hidden_activation = hidden_activation
        self.output_type = output_type
        self.use_dropout = use_dropout
        self.pooling = pooling
        self.dropout_alpha = dropout_alpha
        self.data_format = data_format
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.dense_neurons = dense_neurons
        self.metrics = metrics
        self.leaky_alpha = leaky_alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2_alpha = l2_alpha
        self.covariance_scale = covariance_scale
        if l2_alpha > 0:
            self.use_l2 = 1
        else:
            self.use_l2 = 0
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.model_ = None

    def build_network(self, conv_input_shape, output_size):
        """
        Create a keras model with the hyperparameters specified in the constructor.

        Args:
            conv_input_shape (tuple of shape [variable, y, x]): The shape of the input data
            output_size: Number of neurons in output layer.
        """
        if self.use_l2:
            reg = l2(self.l2_alpha)
        else:
            reg = None
        if self.covariance_scale > 0:
            cov_reg = CovarianceRegularizer(self.covariance_scale)
        else:
            cov_reg = None
        conv_input_layer = Input(shape=conv_input_shape, name="conv_input")
        num_conv_layers = int(np.round((np.log(conv_input_shape[1]) - np.log(self.min_data_width))
                                       / np.log(self.pooling_width)))
        num_filters = self.min_filters
        scn_model = conv_input_layer
        for c in range(num_conv_layers):
            scn_model = Conv2D(num_filters, (self.filter_width, self.filter_width),
                               data_format=self.data_format, kernel_regularizer=reg,
                               padding="same", name="conv_{0:02d}".format(c))(scn_model)
            if self.hidden_activation == "leaky":
                scn_model = LeakyReLU(self.leaky_alpha, name="hidden_activation_{0:02d}".format(c))(scn_model)
            else:
                scn_model = Activation(self.hidden_activation, name="hidden_activation_{0:02d}".format(c))(scn_model)
            if self.use_dropout:
                scn_model = SpatialDropout2D(rate=self.dropout_alpha)(scn_model)
            num_filters = int(num_filters * self.filter_growth_rate)
            if self.pooling.lower() == "max":
                scn_model = MaxPool2D(pool_size=(self.pooling_width, self.pooling_width),
                                      data_format=self.data_format, name="pooling_{0:02d}".format(c))(scn_model)
            else:
                scn_model = AveragePooling2D(pool_size=(self.pooling_width, self.pooling_width),
                                             data_format=self.data_format, name="pooling_{0:02d}".format(c))(scn_model)
        scn_model = Flatten(name="flatten")(scn_model)
        if self.use_dropout:
            scn_model = Dropout(rate=self.dropout_alpha)(scn_model)
        scn_model = Dense(self.dense_neurons, name="dense_hidden",
                          kernel_regularizer=reg, activity_regularizer=cov_reg)(scn_model)
        if self.hidden_activation == "leaky":
            scn_model = LeakyReLU(self.leaky_alpha, name="hidden_dense_activation")(scn_model)
        else:
            scn_model = Activation(self.hidden_activation, name="hidden_dense_activation")(scn_model)
        scn_model = Dense(output_size, name="dense_output")(scn_model)
        scn_model = Activation(self.output_type, name="activation_output")(scn_model)
        self.model_ = Model(conv_input_layer, scn_model)

    def compile_model(self):
        """
        Compile the model in tensorflow with the right optimizer and loss function.
        """
        if self.optimizer.lower() == "adam":
            opt = Adam(lr=self.learning_rate)
        else:
            opt = SGD(lr=self.learning_rate, momentum=0.99)
        self.model_.compile(opt, losses[self.loss], metrics=self.metrics)

    @staticmethod
    def get_data_shapes(x, y):
        """
        Extract the input and output data shapes in order to construct the neural network.
        """
        if len(x.shape) != 4:
            raise ValueError("Input data does not have dimensions (examples, y, x, predictor)")
        if len(y.shape) == 1:
            output_size = 1
        else:
            output_size = y.shape[1]
        return x.shape[1:], output_size

    def fit(self, x, y, val_x=None, val_y=None, build=True, callbacks=None, **kwargs):
        """
        Train the neural network.
        """
        if build:
            x_conv_shape, y_size = self.get_data_shapes(x, y)
            self.build_network(x_conv_shape, y_size)
            self.compile_model()
        if val_x is None:
            val_data = None
        else:
            val_data = (val_x, val_y)
        if callbacks is None:
            callbacks = []
        if self.early_stopping > 0:
            callbacks.append(EarlyStopping(patience=self.early_stopping))
        self.model_.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose,
                        validation_data=val_data, callbacks=callbacks, **kwargs)

    def predict(self, x):
        preds = self.model_.predict(x, batch_size=self.batch_size)
        if len(preds.shape) == 2:
            if preds.shape[1] == 1:
                preds = preds.ravel()
        return preds

    def output_hidden_layer(self, x, layer_index=-3):
        """
        Chop the end off the neural network and capture the output from the specified layer index

        Args:
            x: input data
            layer_index (int): list index of the layer being output.

        Returns:
            output: array containing output of that layer for each example.
        """
        sub_model = Model(self.model_.input, self.model_.layers[layer_index].output)
        output = sub_model.predict(x, batch_size=self.batch_size)
        return output

    def saliency(self, x, layer_index=-3, ref_activation=10):
        """
        Output the gradient of input field with respect to each neuron in the specified layer.

        Args:
            x:
            layer_index:
            ref_activation: Reference activation value for loss function.

        Returns:

        """
        saliency_values = np.zeros((self.model_.layers[layer_index].output.shape[-1],
                                    x.shape[0], x.shape[1],
                                    x.shape[2], x.shape[3]),
                                   dtype=np.float32)
        for s in trange(self.model_.layers[layer_index].output.shape[-1], desc="neurons"):
            sub_model = Model(self.model_.input, self.model_.layers[layer_index].output[:, s])
            batch_indices = np.append(np.arange(0, x.shape[0], self.batch_size), x.shape[0])
            for b, batch_index in enumerate(tqdm(batch_indices[:-1], desc="batch examples", leave=False)):
                x_case = tf.Variable(x.values[batch_index:batch_indices[b + 1]])
                with tf.GradientTape() as tape:
                    tape.watch(x_case)
                    act_out = sub_model(x_case)
                    loss = (ref_activation - act_out) ** 2
                saliency_values[s, batch_index:batch_indices[b + 1]] = tape.gradient(loss, x_case)
        saliency_da = xr.DataArray(saliency_values, dims=("neuron", "p", "row", "col", "var_name"),
                                   coords=x.coords, name="saliency")
        return saliency_da

    def model_config(self):
        all_model_attrs = pd.Series(list(self.__dict__.keys()))
        config_attrs = all_model_attrs[all_model_attrs.str[-1] != "_"]
        model_config_dict = {}
        for attr in config_attrs:
            model_config_dict[attr] = self.__dict__[attr]
        return model_config_dict

    def save_model(self, out_path, model_name):
        model_config_dict = self.model_config()
        model_config_file = join(out_path, "config_" + model_name + ".yml")
        with open(model_config_file, "w") as mcf:
            yaml.dump(model_config_dict, mcf, Dumper=yaml.Dumper)
        if self.model_ is not None:
            model_filename = join(out_path, model_name + ".h5")
            self.model_.save(model_filename, save_format="h5")
        return


def load_conv_net(model_path, model_name):
    model_config_file = join(model_path, "config_" + model_name + ".yml")
    with open(model_config_file, "r") as mcf:
        model_config_dict = yaml.load(mcf, Loader=yaml.Loader)
    conv_net = BaseConvNet(**model_config_dict)
    model_filename = join(model_path, model_name + ".h5")
    conv_net.model_ = load_model(model_filename)
    return conv_net
