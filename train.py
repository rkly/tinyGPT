import os
import re
import time
import itertools

from tqdm import tqdm

import numpy as np
import tensorflow as tf

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
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epsilon=config.epsilon,
            global_clipnorm=config.clipnorm
        )
        # self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)  # NONE
        self.train_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    def train(self):
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

        @tf.function
        def train_step(inputs):
            x, y = inputs

            with tf.GradientTape() as tape:
                logits, loss = self.model(x, y, training=True)
            
            gradients = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
            self.train_metric.update_state(y, logits)
            return loss

        # @tf.function
        # def train_step(inputs):
        #     x, y = inputs

        #     with tf.GradientTape() as tape:
        #         logits = self.model(x, True)  #, training=tf.convert_to_tensor(True, dtype=tf.bool))
        #         n_labels = tf.shape(logits)[-1]
        #         mask = tf.reshape(tf.math.logical_not(y < 0), (-1,))
        #         logits = tf.reshape(logits, (-1, n_labels))
        #         logits_masked = tf.boolean_mask(logits, mask)
        #         label_ids = tf.reshape(y, (-1,))
        #         label_ids_masked = tf.boolean_mask(label_ids, mask)
        #         cross_entropy = self.cce(label_ids_masked, logits_masked)
        #         loss = tf.reduce_sum(cross_entropy) * (1. / self.batch_size)
            
        #     grads = tape.gradient(loss, self.model.trainable_variables)
        #     self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
        #     return loss
        print(self.total_steps)
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
                progress_str = f"step {step}: loss {float(loss):.5f} - "\
                f"acc {float(train_loss.result()*100):.2f}% - "\
                f"lr {float(self.optimizer.learning_rate.numpy()):.6f}"
                bar.set_description(progress_str)
            train_loss.reset_states()
            end = time.time()
            print(f"Epoch done in {end - start}")
