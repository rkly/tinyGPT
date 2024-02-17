import tensorflow as tf


def sample(model, input_chars, steps, temperature=1.0, sample=False, top_k=None):
        ctx_sz = model.get_config().get('block_size')

        for _ in range(steps):
            input_chars_crop = tf.keras.utils.pad_sequences(input_chars, maxlen=ctx_sz, dtype='str', padding='pre', truncating='pre', value=None)
            logits = model(input_chars_crop, training=False)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # optionally crop probabilities to only the top k options
                v, i = tf.math.top_k(logits, top_k, sorted=True)
                cond = tf.math.less(logits, tf.expand_dims(v[:, -1], axis=-1))
                logits = tf.where(cond, tf.constant(-float('inf'), dtype=logits.dtype), logits)
            probabilities = tf.nn.softmax(logits, axis=-1)
            if sample:
                chunk_id = tf.random.categorical(tf.math.log(probabilities), num_samples=1)
            else:
                _, chunk_id = tf.math.top_k(probabilities, k=1)
            chunk_chars = model.id2char(chunk_id)
            input_chars = tf.concat((input_chars, chunk_chars), axis=1)
        return input_chars

model = tf.keras.models.load_model('char_keras_model')
context = "O Love, O God, o this beautiful world"
inputs = tf.strings.unicode_split(context, 'UTF-8')
x = tf.convert_to_tensor(inputs, dtype=tf.string)[None, ...]
y = sample(model, x, 256, temperature=0.9, sample=True, top_k=10)[0]
completion = tf.strings.reduce_join(y).numpy().decode('utf-8')
print(completion)
