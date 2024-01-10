import time
from time import strftime, localtime

from tqdm import tqdm

import tensorflow as tf
import tensorflow_addons as tfa

from gpt import GPT, GPTConfig

class Trainer:
    def __init__(self, model: GPT, config: GPTConfig, epochs: int, batch_size: int, ds: tf.data.Dataset, test_ds: tf.data.Dataset, total_steps: int) -> None:
        self.config = config
        self.dataset = ds
        self.test_dataset = test_ds
        # self.idx = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.total_steps = total_steps

        self.model = model(config)
        step = tf.Variable(0, trainable=False)
        schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            config.learning_rate, 500, config.weight_decay, staircase=False)
        # lr and wd can be a function or a tensor
        lr = config.learning_rate * schedule(step)
        wd = lambda: config.weight_decay * schedule(step)
        self.optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd, epsilon=config.epsilon, clipnorm=config.clipnorm)
        self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)  # NONE
        self.train_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    def train(self):
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

        @tf.function
        def train_step(inputs):
            x_char, y_char = inputs
            y = self.model.char2id(y_char)

            with tf.GradientTape() as tape:
                logits, loss = self.model(x_char, y_char, training=True)
                n_labels = tf.shape(logits)[-1]
                mask = tf.reshape(tf.math.logical_not(y < 0), (-1,))
                logits = tf.reshape(logits, (-1, n_labels))
                logits_masked = tf.boolean_mask(logits, mask)
                label_ids = tf.reshape(y, (-1,))
                label_ids_masked = tf.boolean_mask(label_ids, mask)
                cross_entropy = self.cce(label_ids_masked, logits_masked)
                #loss = tf.reduce_sum(cross_entropy) * (1. / self.batch_size)
            
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
            return loss
        # print(self.total_steps)
        # todo fix cardinality
        for epoch in range(self.epochs):
            print(f"\nStart of epoch {epoch}")
            start = time.time()

            bar = tqdm(self.dataset.enumerate())
            for step, inputs in bar:  #, total=self.total_steps):
                loss = train_step(inputs)
                # labels = inputs[-1]
                # self.idx += tf.reduce_sum(tf.cast(labels >= 0, tf.int32)).numpy()
                train_loss(loss)
                progress_str = f"step {step}: step loss {float(loss):.5f} - "\
                f"loss {train_loss.result()} - "\
                f"lr {self.optimizer.learning_rate.numpy()}"
                bar.set_description(progress_str)
            train_loss.reset_states()
            end = time.time()
            print(f"Epoch done in {strftime('%Y-%m-%d %H:%M:%S', localtime(end - start))}")
