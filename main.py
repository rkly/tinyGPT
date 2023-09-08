from absl import app
from absl import flags

import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

flags.DEFINE_integer('seq_length', 128, 'Maximum context length for predictions')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_integer('epochs', 3, 'Number of training epochs')

AUTOTUNE = tf.data.AUTOTUNE


def main(argv):
    del argv
    from train import Trainer  # to parse flags
    from gpt import GPT, GPTConfig  

    ds = tfds.load(name='tiny_shakespeare')
    ds_train, ds_val, ds_test = ds['train'], ds['validation'], ds['test']
    splitter = lambda x: tf.strings.unicode_split(x['text'], 'UTF-8')
    ds_train, ds_val, ds_test = (
        ds_train.map(splitter, num_parallel_calls=AUTOTUNE),
        ds_val.map(splitter, num_parallel_calls=AUTOTUNE),
        ds_test.map(splitter, num_parallel_calls=AUTOTUNE)
    )

    vocab = sorted(set(next(iter(ds_train)).numpy()))
    char2id = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    id2char = tf.keras.layers.StringLookup(vocabulary=char2id.get_vocabulary(), invert=True, mask_token=None)

    ds_train, _, _ = ds_train.map(char2id), ds_val.map(char2id), ds_test.map(char2id)
    ds_train = ds_train.unbatch()
    ds_train = ds_train.batch(FLAGS.seq_length + 1, drop_remainder=True)
    # total_steps = ds_train.cardinality()
    # print(total_steps)
    # exit()

    @tf.function
    def split_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    ds_train = ds_train.map(split_target, num_parallel_calls=AUTOTUNE)
    total_steps = 40000  #ds_train.cardinality()
    # print(total_steps)
    # exit()
    ds_train = ds_train.shuffle(10000).batch(FLAGS.batch_size)
    ds_train = ds_train.prefetch(AUTOTUNE)
    print(ds_train)

    setattr(GPTConfig, 'vocab_size', char2id.vocabulary_size())
    trainer = Trainer(GPT, GPTConfig, FLAGS.epochs, FLAGS.batch_size, ds_train, None, total_steps)
    trainer.train()

    trainer.model.save('model')

    context = "O Love, O World"
    inputs = tf.strings.unicode_split(context, 'UTF-8')
    x = tf.convert_to_tensor(char2id(inputs), dtype=tf.int64)[None, ...]
    y = trainer.model.sample(x, 512, temperature=1.0, sample=True, top_k=10)[0]
    completion = "".join([c.decode('utf-8') for c in id2char(y).numpy()])
    print(completion)


if __name__ == '__main__':
    app.run(main)
