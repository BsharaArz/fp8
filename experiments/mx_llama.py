import jax
import transformer
import mx_transformer
import llama

def init_llama(prng_key, batch_size, sequence_length, d_model, d_ff, num_blocks, vocab_size):
    #split keys
    key0, key1, key2 = jax.random.split(prng_key, num=3)
    
    #init abstraction params
    tran = mx_transformer.init_transformer(key0, batch_size, sequence_length, d_model, d_ff, num_blocks)
    #embed_weights = embedding.create_table(key1, d_model, vocab_size)
    #log_weights = logits_weights.init_logits_weights(key2, d_model, vocab_size)
    
    return llama.Llama(tran, None, None)

#JUST FORWARD
def forward_llama(model: llama.Llama, seq, num_heads, drop, prng_key):
    #forward pass
    logits = mx_transformer.transformer_forward(model.tran, seq, num_heads, drop, prng_key)
    return logits