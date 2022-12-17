# crypto-transformers

The attention-based methods and transformers made a significant breakthrough in the deep learning area and greatly impacted NLP task solutions [3]. Although recent works show that they could potentially improve results in different tasks domains, the application of transformer for financial data in particular transactions data is underexplored.

While applying attention mechanisms, one can face the apparent restriction on input sequence length due to the method's quadratic complexity. Recent papers proposed different ways to overcome this problem, but we want to concentrate on two promising approaches: Informer and Performer [1, 2].                                                                                                                                                                                                                                                                                      
[1] Martins, Pedro Henrique, Zita Marinho, and André FT Martins. "oo-former: Infinite Memory Transformer." - "Informer" model
[2] Choromanski, Krzysztof, et al. "Rethinking attention with performers." arXiv preprint arXiv:2009.14794 (2020). - "Performer" model
[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. - "Full attention" model

This repo is an attempt to experiment with these models on financial data. The main objectives are listed below:

1) implementation for baseline model
2) implementation for performer model
3) implementation for Informer model
4) benchmarking all models in terms of speed and memory consumption


### performer usage

```bash
git clone https://github.com/lucidrains/performer-pytorch
```
Run the performer notebook in performer_experiments
model checkpoint: https://drive.google.com/file/d/1CeF6Pdq9HYCdNebXOcFeD8B-vA6iT1ns/view?usp=share_link

### informer usage

you'll need to clone the performer repository 

```bash
git clone https://github.com/zhouhaoyi/Informer2020
```
Run the informer notebook in informer_experiments
model checkpoint: https://drive.google.com/file/d/1ek4F8ldTTn3AN4kK_zRdO_g9Z2lfF_Y9/view?usp=share_link
