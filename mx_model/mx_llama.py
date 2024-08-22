import jax
import mx_transformer
import embedding
import mx_logits_weights
import mx
import jax.numpy as jnp
from typing_extensions import NamedTuple

class Llama(NamedTuple):
    tran: mx_transformer.Transformer
    embed_weights: embedding.Embedding
    log_weights: mx_logits_weights.LogitsWeights

def init_llama(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks, vocab_size):
    #split keys
    key0, key1, key2 = jax.random.split(prng_key, num=3)
    
    #init abstraction params
    tran = mx_transformer.init_transformer(key0, batch_size, sequence_length, d_model, d_ff, num_blocks)
    embed_weights = embedding.create_table(key1, d_model, vocab_size)
    log_weights = mx_logits_weights.init_logits_weights(key2, d_model, vocab_size)
    
    return Llama(tran, embed_weights, log_weights)

#modify forward llama to use MX FP8
def forward_llama(model: Llama, seq, num_heads, drop, prng_key):
    #embed seq
    embedded = embedding.embedding_lookup(model.embed_weights, seq) #using fp32 since no multiplication involved

    #forward pass
    logits = mx_transformer.transformer_forward(model.tran, embedded, num_heads, drop, prng_key)

    #logits weights
    output = mx_logits_weights.logits_weights_lookup(model.log_weights, logits)
    return output
