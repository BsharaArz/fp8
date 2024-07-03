import jax
import transformer
import transformer_block
import mx_transformer
import llama
import embedding
import attention
import mx_logits_weights
import logits_weights
import mx
import mlp
import jax.numpy as jnp
#modify forward llama to use MX FP8
def forward_llama(model: llama.Llama, seq, num_heads, drop, prng_key):
    #embed seq
    embedded = embedding.embedding_lookup(model.embed_weights, seq) #using fp32 since no multiplication involved

    #forward pass
    logits = mx_transformer.transformer_forward(model.tran, embedded, num_heads, drop, prng_key)

    #logits weights
    output = mx_logits_weights.logits_weights_lookup(model.log_weights, logits)
    return output

def quantize_llama(model: llama.Llama):
    quantized_blocks = []
    for block in model.tran.blocks:
        #attention
        quantized_attn_layer = attention.Attention(mx.quantize(block.attn_layer.q, jnp.float8_e4m3fn), mx.quantize(block.attn_layer.k, jnp.float8_e4m3fn), mx.quantize(block.attn_layer.v, jnp.float8_e4m3fn))
        quantized_mlp_layers = []
        for w, b in block.mlp_layer.layers:
            quantized_mlp_layers.append([mx.quantize(w, jnp.float8_e4m3fn), b])
        quantized_mlp_layer = mlp.MLP(quantized_mlp_layers)
        quantized_blocks.append(transformer_block.TransformerBlock(quantized_attn_layer, quantized_mlp_layer))
    return llama.Llama(transformer.Transformer(quantized_blocks), model.embed_weights, logits_weights.LogitsWeights(mx.quantize(model.log_weights.table, jnp.float8_e4m3fn)))


def quantize_grad(model: llama.Llama):
    quantized_blocks = []
    for block in model.tran.blocks:
        #attention
        quantized_attn_layer = attention.Attention(mx.quantize(block.attn_layer.q.seq, jnp.float8_e5m2), mx.quantize(block.attn_layer.k, jnp.float8_e5m2), mx.quantize(block.attn_layer.v, jnp.float8_e5m2))
        quantized_mlp_layers = []
        for w, b in block.mlp_layer.layers:
            quantized_mlp_layers.append([mx.quantize(w, jnp.float8_e5m2), mx.quantize(b, jnp.float8_e5m2)])
        quantized_mlp_layer = mlp.MLP(quantized_mlp_layers)
        quantized_blocks.append(transformer_block.TransformerBlock(quantized_attn_layer, quantized_mlp_layer))
    return llama.Llama(transformer.Transformer(quantized_blocks), embedding.Embedding(mx.quantize(model.embed_weights.table, jnp.float8_e5m2)), logits_weights.LogitsWeights(mx.quantize(model.log_weights.table, jnp.float8_e5m2)))