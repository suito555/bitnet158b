This is an implementation of the BitLinear part of BitNet1.58b, as described in the paper.<br>
https://arxiv.org/abs/2310.11453<br>
https://arxiv.org/abs/2402.17764<br>
https://github.com/microsoft/unilm/tree/master/bitnet<br>
<br>
The following code was used as a reference for the implementation of RMSNorm:<br>
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py<br>
<br>
<br>
### How to use
1. Replace all nn.linear in attention and swiGLU with BitLinear.
2. Remove RMSNorm before attention and swiGLU because BitLinear has built-in RMSNorm.
<br>
