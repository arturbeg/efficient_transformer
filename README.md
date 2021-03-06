# Efficient Transformers for Language Modelling

Scaling Transformer architectures has been critical for pushing the frontiers of Language Modelling (LM), a problem central to Natural Language Processing (NLP) and Language Understanding. Although there is a direct positive relationship between the Transformer capacity and its LM performance, there are practical limitations which make training massive models impossible. These limitations come in the form of computation and memory costs which cannot be solely addressed by training on parallel devices. In this thesis, we investigate two approaches which can make Transformers more computationally and memory efficient. First, we introduce the Mixture-of-Experts (MoE) Transformer which can scale its capacity at a sub-linear computational cost. Second, we present a novel content-based sparse attention mechanism called Hierarchical Self Attention (HSA). We demonstrate that the MoE Transformer is capable of achieving lower test perplexity values than a vanilla Transformer model with higher computational demands. Language Modelling experiments, involving a Transformer which uses HSA in place of conventional attention, revealed that HSA can speed up attention computation by up to 330% at a negligible cost in model performance.

## Hierarchical Self Attention

![shot 1](hsa.png?raw=true)

## Hierarchical Self Attention Results

![shot 1](hsa_results.png?raw=true)
