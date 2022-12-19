# Efficient Transformers

## About The Project
### TLDR
A course project aimed to review existing improvement over the classical Transformer architecture. We review and test the classical Transformer (baseline), Informer and Performer using a time series of high frequency crypto data. The choice of data is more or less arbitrary but mostly it was due to simplicity of the data.

The goal of the project is to study and compare the GPU memory footprint and speed of inference.

### Background
The attention-based methods and transformers made a significant breakthrough in the deep learning area and greatly impacted NLP task solutions [3]. Although recent works show that they could potentially improve results in different tasks domains, the application of transformer for financial data in particular transactions data is underexplored.

While applying attention mechanisms, one can face the apparent restriction on input sequence length due to the method's quadratic complexity. Recent papers proposed different ways to overcome this problem, but we want to concentrate on two promising approaches: Informer and Performer [1, 2].                                                                                                                                                                                                                                                                                      
[1] Martins, Pedro Henrique, Zita Marinho, and André FT Martins. "oo-former: Infinite Memory Transformer." - "Informer" model
[2] Choromanski, Krzysztof, et al. "Rethinking attention with performers." arXiv preprint arXiv:2009.14794 (2020). - "Performer" model
[3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. - "Full attention" model

This repo is an attempt to experiment with these models on financial data. The main objectives are listed below:

1) implementation for baseline model
2) implementation for Informer model
3) ~~implementation for Performer model~~ The model is not included in the final report due to some software related challenges
4) benchmarking all models in terms of speed and memory consumption

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Nik212/efficient-transformers-4seqdata.git
   ```
2. Navigate to a folder ```baseline_experiments``` or ```informer_experiments``` depending on the model for which you would like to have an inference
3. Download the checkpoints for different encoder seqeuence lengths from google drive https://drive.google.com/drive/folders/1_13U4DmnYnTqc8FfvA7AL5pO5V5HDcTn. Place the folder in the corresponding folder. (!) Make sure that the path ```/*_experiments/checkpoints_*/``` exists (!). 
4. Inference speed and memory footprint. Run ```main_*.ipynb``` in the folder and follow the initial instructions given in the notebooks. The inference part is done automatically: all the metrics should be printed out.

