# scSemiGAN
Introduction
-----
As a crucial step in scRNA-seq data analytical tasks, accurate cell-type annotation paves the way for the subsequent discovery of gene function and disease mechanism. In this article, we propose scSemiGAN, a semi-supervised cell-type annotation and dimensionality reduction framework based on generative adversarial network, modeling scRNA-seq data from the aspect of data generation. Our proposed scSemiGAN is capable of performing deep latent representation learning and cell-type label prediction simultaneously. Guided by a few known cell-type labels, dimensionality reduction and cell-type annotation are jointly optimized during the training process.Through extensive comparison with four state-of-the-art annotation methods on diverse simulated and real scRNA-seq datasets , scSemiGAN achieves competitive or superior performance in multiple downstream tasks including cell-type annotation, latent representation visualization, confounding factor removal and enrichment analysis.

Requirement
-----
Python >= 3.6

Tensorflow (GPU version) >= 1.13.1

scanpy >= 1.14.post1

umap-learn == 0.3.9


Example
-----
Here, we use a simulated dataset with 2000 genes and 4 cell types generated by Splatter to give an example. You can download this data from folder "scSemiGAN/data/simulate". You just need to download all code files and focus on the "eval_sim.py" file. We take 10% of cells as the reference data and the remaining ones as query data. You can run the following code in your command lines:

python eval_sim.py

After finishing the entire training, you can get the annotation accuracy and F1-score on the query data respectively. Besides, we use t-SNE and UMAP to visualize the learned latent space representation in a two dimensional plane. The t-SNE and UMAP plots are in folder "figures/tsne.pdf" and "figures/umap.pdf" respectively. 

Method
-----
![model](https://github.com/rafa-nadal/scSemiGAN/blob/main/result/method_color1.png)

