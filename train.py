from tqdm import tqdm, trange

import tensorflow as tf

from gpt import GPT, GPTConfig

class Trainer:
    def __init__(self, model: GPT, config: GPTConfig, epochs: int, batch_size: int, ds: tf.data.Dataset, total_steps: int, ds_val: tf.data.Dataset, val_total_steps: int, ds_test: tf.data.Dataset, test_total_steps: int) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.val_total_steps = val_total_steps
        self.test_total_steps = test_total_steps

        self.strategy = tf.distribute.OneDeviceStrategy("cpu:0")
        with self.strategy.scope():
            self.model = model(config)
            lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
                config.lr, total_steps, alpha=0.1, warmup_target=config.lr * 10, warmup_steps=total_steps * 0.1)
            decay_fn = tf.keras.optimizers.schedules.PolynomialDecay(6e-4, epochs * total_steps, end_learning_rate=6e-5)
            self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=decay_fn, weight_decay=config.weight_decay,
                beta_1=config.beta1, beta_2=config.beta2, epsilon=config.epsilon, clipnorm=config.clipnorm)
            self.optimizer.exclude_from_weight_decay(var_names=['layer_normalization', 'bias'])
            self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            self.ds_dist = self.strategy.experimental_distribute_dataset(ds)
            self.ds_val_dist = self.strategy.experimental_distribute_dataset(ds_val)
            self.ds_test_dist = self.strategy.experimental_distribute_dataset(ds_test)

    def train(self):
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)

        @tf.function
        def train_step(inputs_dist):
            def step_fn(inputs):
                x_char, y_char = inputs
                y = self.model.char2id(y_char)
                # y = y_char
                with tf.GradientTape() as tape:
                    logits = self.model(x_char, training=True)
                    n_labels = tf.shape(logits)[-1]
                    mask = tf.reshape(tf.math.logical_not(y < 0), (-1,))
                    logits = tf.reshape(logits, (-1, n_labels))
                    logits_masked = tf.boolean_mask(logits, mask)
                    label_ids = tf.reshape(y, (-1,))
                    label_ids_masked = tf.boolean_mask(label_ids, mask)
                    cross_entropy = self.cce(label_ids_masked, logits_masked)
                    loss = tf.reduce_sum(cross_entropy) * (1. / self.batch_size)

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
                accuracy.update_state(label_ids_masked, logits_masked)
                return cross_entropy

            per_device_losses = self.strategy.run(step_fn, args=(inputs_dist,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_device_losses, axis=0)
            mean_loss = sum_loss / self.batch_size
            return mean_loss

        @tf.function
        def val_step(inputs_dist):
            def step_fn(inputs):
                x_char, y_char = inputs
                y = self.model.char2id(y_char)

                logits = self.model(x_char, training=False)
                n_labels = tf.shape(logits)[-1]
                mask = tf.reshape(tf.math.logical_not(y < 0), (-1,))
                logits = tf.reshape(logits, (-1, n_labels))
                logits_masked = tf.boolean_mask(logits, mask)
                label_ids = tf.reshape(y, (-1,))
                label_ids_masked = tf.boolean_mask(label_ids, mask)
                cross_entropy = self.cce(label_ids_masked, logits_masked)
                accuracy.update_state(label_ids_masked, logits_masked)
                return cross_entropy

            per_device_losses = self.strategy.run(step_fn, args=(inputs_dist,))
            sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_device_losses, axis=0)
            mean_loss = sum_loss / self.batch_size
            return mean_loss

        with self.strategy.scope():
            for _ in trange(self.epochs, desc="Epochs"):
                bar = tqdm(self.ds_dist, total=self.total_steps)
                for inputs in bar:
                    loss = train_step(inputs)
                    train_loss(loss)
                    progress_str = f"train loss {train_loss.result()} - acc {accuracy.result() * 100} "\
                        f"lr {self.optimizer.learning_rate.numpy()}"
                    bar.set_description(progress_str)
                train_loss.reset_states()
                accuracy.reset_states()

                if self.ds_val_dist:
                    val_bar = tqdm(self.ds_val_dist, total=self.val_total_steps)
                    for inputs in val_bar:
                        loss = val_step(inputs)
                        val_loss(loss)
                        val_progress_str = f"val loss {val_loss.result()} - acc {accuracy.result() * 100}"
                        val_bar.set_description(val_progress_str)
                    val_loss.reset_states()
                    accuracy.reset_states()

            if self.ds_test_dist:
                test_bar = tqdm(self.ds_test_dist, total=self.test_total_steps)
                for inputs in test_bar:
                    loss = val_step(inputs)
                    test_loss(loss)
                    test_progress_str = f"test loss {test_loss.result()} - acc {accuracy.result() * 100}"
                    test_bar.set_description(test_progress_str)
