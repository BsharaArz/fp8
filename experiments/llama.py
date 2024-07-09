import jax
from jax import numpy as jnp
from typing_extensions import NamedTuple

import transformer
import embedding
import logits_weights

class Llama(NamedTuple):
    tran: transformer.Transformer
    embed_weights: embedding.Embedding
    log_weights: logits_weights.LogitsWeights

def init_llama(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks, vocab_size):
    #split keys
    key0, key1, key2 = jax.random.split(prng_key, num=3)
    
    #init abstraction params
    tran = transformer.init_transformer(key0, batch_size, sequence_length, d_model, d_ff, num_blocks)
    embed_weights = embedding.create_table(key1, d_model, vocab_size)
    log_weights = logits_weights.init_logits_weights(key2, d_model, vocab_size)
    
    return Llama(tran, embed_weights, log_weights)

def forward_llama(model: Llama, seq, num_heads, drop, prng_key):
    #embed seq
    embedded = embedding.embedding_lookup(model.embed_weights, seq)

    #forward pass
    logits = transformer.transformer_forward(model.tran, embedded, num_heads, drop, prng_key)

    #logits weights
    output = logits_weights.logits_weights_lookup(model.log_weights, logits)
    return output

def test():
    #initialize params
    prng_key = jax.random.PRNGKey(0)
    d_model = 512
    d_ff = 1024
    batch_size = 32
    sequence_length = 16
    num_heads = 8
    drop = 0.5
    num_blocks = 12
    vocab_size = 30000

    #initialize llama
    llama = init_llama(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks, vocab_size)

    #create seq - only integers
    seq = jax.random.randint(prng_key, (batch_size, sequence_length), 0, vocab_size)

    #forward llama
    output = forward_llama(llama, seq, num_heads, drop, prng_key)

def main():
    test()

if __name__ == "__main__":
    main()