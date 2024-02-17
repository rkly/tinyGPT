from absl import app
from absl import flags

import tensorflow as tf
import tensorflow_datasets as tfds

from train import Trainer
from gpt import GPT, GPTConfig

FLAGS = flags.FLAGS
# Train config
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('epochs', 15, 'Number of training epochs')
# GPT config
flags.DEFINE_integer('block_size', 256, 'Maximum context length for predictions')
flags.DEFINE_integer('n_embed', 384, 'Embedding')
flags.DEFINE_integer('n_heads', 8, 'Number of heads')
flags.DEFINE_integer('n_layer', 8, 'Number of blocks')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate')
flags.DEFINE_float('residual_dropout', 0.1, 'Dropout rate')
flags.DEFINE_float('embed_dropout', 0.1, 'Dropout rate for embeddings')
flags.DEFINE_float('lr', 6e-4, 'Learning rate')
flags.DEFINE_float('weight_decay', 0.1, 'Only applied on matmul weights')
flags.DEFINE_float('beta1', 0.9, 'The exponential decay rate for the 1st moment estimates')
flags.DEFINE_float('beta2', 0.95, 'The exponential decay rate for the 2nd moment estimates')
flags.DEFINE_float('epsilon', 1e-5, 'Optimizer epsilon')
flags.DEFINE_float('clipnorm', 1.0, 'Optimizer clipnorm')


def main(argv):
    del argv

    ds = tfds.load(name='tiny_shakespeare')
    ds_train, ds_val, ds_test = ds['train'], ds['validation'], ds['test']

    @tf.function
    def splitter(sequence):
        return tf.strings.unicode_split(sequence['text'], 'UTF-8')

    @tf.function
    def split_target(sequence):
        return sequence[:-1], sequence[1:]

    def get_steps(ds):
        all_chars = next(iter(ds.map(splitter))).numpy()
        vocab = sorted(set(all_chars))
        steps_per_epoch = len(all_chars) // (FLAGS.block_size + 1) // FLAGS.batch_size
        return vocab, steps_per_epoch

    vocab, steps_per_epoch = get_steps(ds_train)
    _, val_steps_per_epoch = get_steps(ds_val)
    _, test_steps_per_epoch = get_steps(ds_test)

    def process_ds(ds):
        ds = ds.map(splitter, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.unbatch().batch(FLAGS.block_size + 1, drop_remainder=True)
        ds = ds.map(split_target, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(steps_per_epoch).batch(FLAGS.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    ds_train, ds_val, ds_test = map(process_ds, (ds_train, ds_val, ds_test))

    gpt_config = GPTConfig.make(vocab, **FLAGS.flag_values_dict())
    trainer = Trainer(GPT, gpt_config, FLAGS.epochs, FLAGS.batch_size, ds_train, steps_per_epoch, ds_val, val_steps_per_epoch, ds_test, test_steps_per_epoch)
    trainer.train()
    trainer.model.save('char_keras_model')


if __name__ == '__main__':
    app.run(main)
