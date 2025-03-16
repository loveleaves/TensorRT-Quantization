# TensorRT-LLM

## References
- [example: deploy gpt](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/gpt#gpt)
- [docs](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)

## Paged Attention
Paged Attention无论是在TensorRT-LLM还是vLLM中，都是一个核心的加速功能。通过Paged Attention高效地管理attention中缓存的张量，实现了比HuggingFace Transformers高14-24倍的吞吐量，详见[vLLM官方文档Paged Attention解读](https://zhuanlan.zhihu.com/p/712664813)。

### LLM的内存管理限制性能
在自回归解码过程中，LLM的所有输入token都会生成attention机制的keys和values的张量，并且这些张量被保留在GPU内存中，用来生成下一个token。大语言模型需要大量内存来存储每个token的keys和values，并且随着输入序列变长时，所需的内存也会扩展到非常大的规模。

这些缓存的key和value的张量通常称为KV缓存。KV缓存有2个特点：
- 内存占用大：在LLaMA-13B中，单个序列的KV缓存占用高达1.7GB的内存
- 动态化：其大小取决于序列的长度，而序列长度高度易变，且不可预测
在常规的attention机制中，keys和values的值必须连续存储。因此，即使我们能在给序列分配的内存的中间部分做了内存释放，也无法使用这部分空间供给其他序列，这会导致内存碎片和浪费。因此，有效管理KV缓存是一个重大挑战。

 

### Paged Attention机制
为了解决上述的内存管理问题，Paged Attention的解决办法非常直接：它允许在非连续的内存空间中存储连续的keys和values。

具体来说，Paged Attention将每个序列的KV缓存分为若干个block，每个block包含固定数量token的key和value张量。在注意力计算过程中，Paged Attention内核能够高效地识别和提取这些block。由于这些block在内存中不需要连续，因此也就可以像操作系统的虚拟内存一样，以更灵活的方式管理key和value张量——将block看作page，token看作bytes，序列看作process。

序列的连续逻辑块通过块表映射到非连续的物理块。随着生成新的token，物理块会按需进行分配。这种方式可以防止内存碎片化，提高内存利用率。在生成输出序列时，page即可根据需要动态分配和释放。因此,如果我们中间释放了一些page，那这些空间就可以被重新用于存储其他序列的KV。

通过这种机制，可以使得极大减少模型在执行复杂采样算法（例如parallel sampling和beam search）的内存开销，提升推理吞吐性能。


## KV Cache
KV Cache是LLM推理优化里的一个常用技术，可以在不影响计算精度的情况下，通过空间换时间的办法，提高推理性能。KV Cache发生在多个token生成的步骤中，并且只发生在Decoder-only模型中（例如GPT。BERT这样的encoder模型不是生成式模型，而是判别性模型）。

我们知道在生成式模型中，是基于现有的序列生成下一个token。我们给一个输入文本，模型会输出一个回答（长度为 N），其实该过程中执行了 N 次推理过程。模型一次推理只输出一个token，输出的 token 会与输入序列拼接在一起，然后作为下一次推理的输入，这样不断反复直到遇到终止符。

而由于每个token都需要计算其Key和Value（Attention机制），所会存在一个问题：每次生成新的token时，都需要计算之前token的KV，存在大量冗余计算。而在整个计算过程中，每个token的K、V计算方式是一样，所以可以把每个token（也就是生成过程中每个token推理时）的K、V放入内存进行缓存。这样在进行下一次token预测时，就可以不需要对之前的token KV再进行一次计算，从而实现推理加速。这便是KV Cache的基本原理，如果各位希望了解更多的细节，可以继续阅读[这篇文章](https://zhuanlan.zhihu.com/p/700197845)。