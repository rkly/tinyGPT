from absl import flags

from typing import Optional

from dataclasses import dataclass, fields

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal

FLAGS = flags.FLAGS

flags.DEFINE_integer('n_embed', 768, 'Embedding')
flags.DEFINE_integer('n_head', 6, 'Number of heads')
flags.DEFINE_integer('n_layer', 6, 'Number of blocks')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate')
flags.DEFINE_float('weight_decay', 0.1, 'Only applied on matmul weights')
flags.DEFINE_float('epsilon', 1e-5, 'Optimizer epsilon')
flags.DEFINE_float('clipnorm', 1.0, 'Optimizer clipnorm')

AUTOTUNE = tf.data.AUTOTUNE


class TinyMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.0, residual_dropout_rate=0.0):
        super(TinyMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_rate = dropout_rate
        self.residual_dropout_rate = residual_dropout_rate

        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)

        self.drop = tf.keras.layers.Dropout(self.residual_dropout_rate)

        self.final_dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask=None, training=False):
        batch_size = tf.shape(query)[0]

        # Linearly transform queries, keys, and values for all heads
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Scaled dot-product attention for all heads
        attention_output, attention_weights = self.scaled_dot_product_attention(
            query, key, value, mask, training
        )

        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))

        # Apply dropout to attention output
        attention_output = self.drop(attention_output, training=training)

        # Linear transformation to get the final output
        output = self.final_dense(attention_output)

        return output, attention_weights

    def scaled_dot_product_attention(self, query, key, value, mask, training):
        # Compute scaled dot-product attention scores
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))

        # Apply mask (if provided) during training only
        if mask is not None and training:
            scaled_attention_logits += (mask * -1e9)

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply dropout to attention weights
        attention_weights = self.drop(attention_weights, training=training)

        # Compute the weighted sum of values using attention weights
        output = tf.matmul(attention_weights, value)

        return output, attention_weights



# class MultiHeadAttention(layers.Layer):
#     def __init__(self, n_embed: int, num_heads: int, att_dropout: Optional[float]=0.1, residual_dropout: Optional[float]=0.1) -> None:
#         super(MultiHeadAttention, self).__init__()
#         self.key = layers.Dense(n_embed, kernel_initializer=RandomNormal(mean=0., stddev=0.02), name='key')
#         self.query = layers.Dense(n_embed, kernel_initializer=RandomNormal(mean=0., stddev=0.02), name='query')
#         self.value = layers.Dense(n_embed, kernel_initializer=RandomNormal(mean=0., stddev=0.02), name='value')
#         self.n_embed = n_embed
#         self.num_heads = num_heads
#         self.att_dropout = att_dropout
#         self.residual_dropout_layer = layers.Dropout(rate=residual_dropout, name='resid_dropout')
#         self.depth = n_embed // self.num_heads
#         self.attention_layer = layers.Attention(use_scale=True, dropout=att_dropout)

#         # reg todo
#         # projection
#         self.proj = layers.Dense(n_embed, name='proj')

#     @tf.function
#     def split_heads(self, x: tf.Tensor) -> tf.Tensor:
#         """
#         Transpose to the shape of (batch_size, num_heads, seq_length, depth)
#         """
#         x = tf.reshape(x, (FLAGS.batch_size, -1, self.num_heads, self.depth), name='split_heads_reshape')
#         return tf.transpose(x, perm=[0, 2, 1, 3], name='split_heads_transpose')

#     def call(self, x: tf.Tensor, mask: tf.Tensor, training=False) -> tf.Tensor:
#         key = self.key(x)  # (batch_size, seq_length, n_embed)
#         query = self.query(x)
#         value = self.value(x)

#         key = self.split_heads(key)  # (batch_size, num_heads, seq_length, depth)
#         query = self.split_heads(query)
#         value = self.split_heads(value)

#         scaled_attention_logits = self.attention_layer([query, value, key], mask, training)
#         scaled_attention = tf.transpose(scaled_attention_logits, perm=[0, 2, 1, 3])
#         concat_attention = tf.reshape(scaled_attention, (FLAGS.batch_size, -1, self.n_embed))
#         output = self.proj(concat_attention)
#         output = self.residual_dropout_layer(output, training=training)
#         return output

class Block(layers.Layer):
    def __init__(self, conf: dict) -> None:
        super().__init__()

        self.mha = TinyMultiHeadAttention(conf.n_embed, conf.n_head, dropout_rate=conf.att_dropout, residual_dropout_rate=conf.residual_dropout)
        # self.mha = layers.MultiHeadAttention(conf.n_head, conf.n_embed, dropout=conf.att_dropout)
        self.ffn = tf.keras.Sequential([
            layers.Dense(conf.n_embed * 4, activation='gelu', kernel_initializer=RandomNormal(mean=0., stddev=0.02)),
            layers.Dense(conf.n_embed),
            layers.Dropout(conf.residual_dropout)
            ])
        self.layernorm1 = layers.LayerNormalization(epsilon=conf.epsilon)
        self.layernorm2 = layers.LayerNormalization(epsilon=conf.epsilon)

        self.drop = layers.Dropout(conf.att_dropout)
        self.drop_ffn = layers.Dropout(conf.att_dropout)

    # def call(self, x: tf.Tensor, mask, training) -> tf.Tensor:
    #     norm = self.layernorm1(x)
    #     x += self.mha(norm, mask, training=training)
    #     x += self.ffn(self.layernorm2(x), training=training)
    #     return x
    def call(self, inputs, mask, training=False):
        # Multi-head self-attention
        attn_output, _ = self.mha(inputs, inputs, inputs, mask=mask, training=training)
        attn_output = self.drop(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feedforward network
        ffn_output = self.ffn(out1)
        ffn_output = self.drop_ffn(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

@dataclass
class GPTConfig:
    block_size: int = FLAGS.seq_length
    n_layer: int = FLAGS.n_layer
    n_head: int = FLAGS.n_head
    n_embed: int = FLAGS.n_embed
    att_dropout: float = FLAGS.dropout
    residual_dropout: float = FLAGS.dropout
    embed_dropout: float = FLAGS.dropout
    learning_rate: float = FLAGS.learning_rate
    weight_decay: float = FLAGS.weight_decay
    epsilon: float = FLAGS.epsilon
    clipnorm: float = FLAGS.clipnorm

class GPT(tf.keras.Model):
    def __init__(self, GPTConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = GPTConfig()
        assert all([x is not None for x in fields(self.conf)])

        self.token_emb = layers.Embedding(self.conf.vocab_size, self.conf.n_embed,
                                        embeddings_initializer=RandomNormal(mean=0., stddev=0.02),
                                        name='token_embedding')
        self.drop = layers.Dropout(self.conf.embed_dropout)

        self.blocks = tf.keras.Sequential([
            Block(self.conf) for _ in range(self.conf.n_layer)
        ])
        self.layernorm = layers.LayerNormalization(epsilon=self.conf.epsilon)
        self.head = layers.Dense(self.conf.vocab_size, use_bias=False, kernel_initializer=RandomNormal(mean=0., stddev=0.02))

        self.block_size = self.conf.block_size
        self.n_embed = self.conf.n_embed
        self.n_layer = self.conf.n_layer

        # def build(self, input_shape):
        self.position_emb = self.add_weight("position_embeddings",
            shape=(1, self.conf.block_size, self.conf.n_embed),
            initializer=tf.keras.initializers.Zeros(),
            dtype=tf.float32,
            trainable=True)
            #super().build(input_shape)

    def call(self, input_ids, targets=None, training=False):
        _, T = input_ids.shape
        token_embeddings = self.token_emb(input_ids)
        position_embeddings = self.position_emb[:, :T, :]
        e = self.drop(token_embeddings + position_embeddings, training)
        mask = 1 - tf.linalg.band_part(tf.ones((T, T)), -1, 0)  #  tf.linalg.band_part(input, -1, 0) ==> Lower triangular part.
        # for i in range(self.n_layer):
        #     x = self.blocks[i](x, mask, training)
        # x = self.layernorm(x)
        
        h = self.blocks(e, mask=mask, training=training)
        h = self.layernorm(h, training=training)
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            loss = loss_fn(targets, logits)
        outputs = (logits, loss)
        return outputs
    
    def sample(
        self,
        input_ids,
        steps,
        temperature=1.0,
        sample=False,
        top_k=None
    ):
        # get model's context size
        ctx_sz = self.conf.block_size
        
        for _ in range(steps):
            B, S = input_ids.shape
            input_ids_cond = input_ids
            if S > ctx_sz: # crop context if needed
                input_ids_cond = input_ids[:,-ctx_sz:]
            logits, _ = self(input_ids_cond, training=False)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # optionally crop probabilities to only the top k options
                v, _ = tf.math.top_k(logits, top_k, sorted=True)
                logits = tf.identity(logits).numpy()
                logits[logits < v.numpy()[:, [-1]]] = -float('Inf')
            probabilities = tf.nn.softmax(logits, axis=-1)
            if sample:
                chunk_id = tf.random.categorical(tf.math.log(probabilities), num_samples=1)
            else:
                _, chunk_id = tf.math.top_k(probabilities, k=1)
            input_ids = tf.concat([
                input_ids, tf.reshape(tf.cast(chunk_id, dtype=input_ids.dtype), shape=(B, 1))], axis=-1
            )
        return input_ids
