import numpy as np
import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    """A multilayer perceptron"""
    def __init__(self, layers_dim):
        self.num_layers = len(layers_dim)
        self.hidden_layer = [0 for _ in range(self.num_layers)]
        super(Encoder, self).__init__()
        for idx, dim in enumerate(layers_dim):
            self.hidden_layer[idx] = tf.keras.layers.Dense(units=dim, activation=tf.nn.relu)
        # self.output_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)

    def call(self, input_features):
        activation = self.hidden_layer[0](input_features)
        for idx in range(1, self.num_layers):
            activation = self.hidden_layer[idx](activation)
        return activation

class Decoder(tf.keras.layers.Layer):
    """A multilayer perceptron"""
    def __init__(self, layers_dim):
        self.num_layers = len(layers_dim)
        self.hidden_layer = [0 for _ in range(self.num_layers)]
        super(Decoder, self).__init__()
        for idx, dim in enumerate(layers_dim):
            self.hidden_layer[idx] = tf.keras.layers.Dense(units=dim, activation=tf.nn.relu)
        # self.output_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)

    def call(self, input_features):
        activation = self.hidden_layer[0](input_features)
        for idx in range(1, self.num_layers):
            activation = self.hidden_layer[idx](activation)
        return activation

class Autoencoder(tf.keras.Model):
    def __init__(self, encoder_dims, decoder_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(layers_dim=encoder_dims)
        self.decoder = Decoder(layers_dim=decoder_dims)

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed

def loss(model, original):
  reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
  return reconstruction_error

def train(loss, model, opt, original):
  with tf.GradientTape() as tape:
    gradients = tape.gradient(loss(model, original), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)

autoencoder = Autoencoder([64], [784])
opt = tf.optimizers.SGD(learning_rate=1e-4, momentum=0.9)

(training_features, _), (test_features, _) = tf.keras.datasets.mnist.load_data()
training_features = training_features / np.max(training_features)
training_features = training_features.reshape(training_features.shape[0],
                                              training_features.shape[1] * training_features.shape[2]).astype(np.float32)
training_dataset = tf.data.Dataset.from_tensor_slices(training_features).batch(256)

writer = tf.summary.create_file_writer('tmp')

with writer.as_default():
  with tf.summary.record_if(True):
      for epoch in range(100):
        print(epoch)
        for step, batch_features in enumerate(training_dataset):
            train(loss, autoencoder, opt, batch_features)
            loss_values = loss(autoencoder, batch_features)
            original = tf.reshape(batch_features, (batch_features.shape[0], 28, 28, 1))
            reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)), (batch_features.shape[0], 28, 28, 1))
            tf.summary.scalar('loss', loss_values, step=step)
            tf.summary.image('original', original, max_outputs=10, step=step)
            tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)
