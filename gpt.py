from typing import Optional
from dataclasses import dataclass, fields

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal


class TinyMultiHeadAttention(tf.keras.layers.Layer):
    # the class is particularly written by chatgpt :)
    def __init__(self, d_model: int, num_heads: int, drop_rate: Optional[float]=0.1, residual_drop_rate: Optional[float]=0.1):
        super(TinyMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.drop = tf.keras.layers.Dropout(drop_rate)
        self.residual_drop = tf.keras.layers.Dropout(residual_drop_rate)

        self.query = tf.keras.layers.Dense(d_model, kernel_initializer=RandomNormal(mean=0., stddev=0.02), name="query")
        self.key = tf.keras.layers.Dense(d_model, kernel_initializer=RandomNormal(mean=0., stddev=0.02), name="key")
        self.value = tf.keras.layers.Dense(d_model, kernel_initializer=RandomNormal(mean=0., stddev=0.02), name="value")
        self.dense = tf.keras.layers.Dense(d_model, name='projection')

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor]=None, training: Optional[bool]=False):
        batch_size = tf.shape(x)[0]

        # Linearly transform queries, keys, and values for all heads
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Split heads: (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, head_dim)
        q = self.split_heads(query, batch_size)
        k = self.split_heads(key, batch_size)
        v = self.split_heads(value, batch_size)

        # Compute scaled dot-product attention scores
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        scaled_attention_logits = matmul_qk / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))  #tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.drop(attention_weights, training=training)
        # Compute the weighted sum of values using attention weights
        scaled_attention = tf.matmul(attention_weights, v)
        # Concatenate heads: (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, d_model)
        attention_output = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        # Linear transformation to get the final output
        output = self.dense(attention_output)
        return self.residual_drop(output, training=training)

class Block(layers.Layer):
    def __init__(self, conf: dict) -> None:
        super().__init__()

        self.tmha = TinyMultiHeadAttention(conf.n_embed, conf.n_heads, drop_rate=conf.dropout, residual_drop_rate=conf.residual_dropout)
        self.ffn = tf.keras.Sequential([
            layers.Dense(conf.n_embed * 4, activation='gelu', kernel_initializer=RandomNormal(mean=0., stddev=0.02)),
            layers.Dense(conf.n_embed),
            layers.Dropout(conf.residual_dropout)
            ])
        self.norm1 = layers.LayerNormalization(epsilon=conf.epsilon)
        self.norm2 = layers.LayerNormalization(epsilon=conf.epsilon)

    def call(self, x: tf.Tensor, mask: tf.Tensor, training: Optional[bool]=False):
        x = x + self.tmha(self.norm1(x), mask=mask, training=training)
        x = x + self.ffn(self.norm2(x), training=training)
        return x

@dataclass
class GPTConfig:
    vocab: list
    block_size: int
    n_layer: int
    n_heads: int
    n_embed: int
    dropout: float
    residual_dropout: float
    embed_dropout: float
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    clipnorm: float

    @classmethod
    def make(cls, vocab: list, **kwargs):
        class_fields = {f.name for f in fields(cls)}
        return cls(vocab=vocab, **{k: v for k, v in kwargs.items() if k in class_fields})

class GPT(tf.keras.Model):
    def __init__(self, config: GPTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert all([x is not None for x in fields(config)])

        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.char2id = tf.keras.layers.StringLookup(vocabulary=sorted(config.vocab), mask_token=None, name="char2id")
        self.id2char = tf.keras.layers.StringLookup(vocabulary=self.char2id.get_vocabulary(), invert=True, mask_token=None, name="id2char")
        vocab_size = self.char2id.vocabulary_size()

        self.token_emb = layers.Embedding(vocab_size, config.n_embed,
            embeddings_initializer=RandomNormal(mean=0., stddev=0.02), name='token_embedding')
        self.position_emb = self.add_weight("position_embeddings",
            shape=(config.block_size, config.n_embed),
            initializer=tf.keras.initializers.Zeros(),
            dtype=tf.float32)
        self.embed_drop = layers.Dropout(config.embed_dropout)

        self.blocks = [Block(config) for _ in range(self.n_layer)]
        self.norm = layers.LayerNormalization(epsilon=config.epsilon)
        self.head = layers.Dense(vocab_size, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02))

    def get_config(self):
        config = super().get_config()
        config.update(block_size=self.block_size)
        return config

    def call(self, input_chars: tf.Tensor, training=False):
        input_ids = self.char2id(input_chars)
        T = tf.shape(input_ids)[1]
        token_embeddings = self.token_emb(input_ids)
        position_embeddings = self.position_emb[:T, :]

        e = self.embed_drop(token_embeddings + position_embeddings, training)
        mask = 1 - tf.linalg.band_part(tf.ones((T, T)), -1, 0)  #  tf.linalg.band_part(input, -1, 0) ==> Lower triangular part.

        for i in range(self.n_layer):
            e = self.blocks[i](e, mask=mask, training=training)
        h = self.norm(e, training=training)
        logits = self.head(h)
        return logits
