import jax
import transformer
import mx_transformer
import llama
import embedding
import mx_logits_weights

#modify forward llama to use MX FP8
def forward_llama(model: llama.Llama, seq, num_heads, drop, prng_key):
    #embed seq
    embedded = embedding.embedding_lookup(model.embed_weights, seq) #using fp32 since no multiplication involved

    #forward pass
    logits = mx_transformer.transformer_forward(model.tran, embedded, num_heads, drop, prng_key)

    #logits weights
    output = mx_logits_weights.logits_weights_lookup(model.log_weights, logits)
    return output