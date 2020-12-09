# MetaPerturb: Transferable Regularizer for Heterogeneous Tasks and Architectures

This is the **Pytorch implementation** for the paper *MetaPerturb: Transferable Regularizer for Heterogeneous Tasks and Architectures* (accepted at **NeurIPS 2020 , spotlight presentation**)



## Links

 [Paper](https://papers.nips.cc/paper/2020/file/84ddfb34126fc3a48ee38d7044e87276-Paper.pdf) 
 <!-- [Slide(todo)]() [Poster(todo)]() -->



## Abstract

![](figures/concept.png)

Regularization and transfer learning are two popular techniques to enhance model generalization on unseen data, which is a fundamental problem of machine learning. Regularization techniques are versatile, as they are task- and architecture-agnostic, but they do not exploit a large amount of data available. Transfer learning methods learn to transfer knowledge from one domain to another, but may not generalize across tasks and architectures, and may introduce new training cost for adapting to the target task. To bridge the gap between the two, we propose a transferable perturbation, *MetaPerturb*, which is meta-learned to improve generalization performance on unseen data. MetaPerturb is implemented as a set-based lightweight network that is agnostic to the size and the order of the input, which is shared across the layers. Then, we propose a meta-learning framework, to jointly train the perturbation function over heterogeneous tasks in parallel. As MetaPerturb is a set-function trained over diverse distributions across layers and tasks, it can generalize to heterogeneous tasks and architectures. We validate the efficacy and generality of MetaPerturb trained on a specific source domain and architecture, by applying it to the training of diverse neural architectures on heterogeneous target datasets against various regularizers and fine-tuning. The results show that the networks trained with MetaPerturb significantly outperform the baselines on most of the tasks and architectures, with a negligible increase in the parameter size and no hyperparameters to tune.



__Contribution of this work__

- We propose a lightweight and versatile perturbation function that can transfer the knowledge of a source task to **heterogeneous target tasks and architectures**.
- We propose **a novel meta-learning framework in the form of joint training**, which allows to efficiently perform meta-learning on large-scale datasets in the standard learning framework.
- We validate our perturbation function on a large number of datasets and architectures, on which it successfully **outperforms existing regularizers and finetuning**.



<!-- __Architecture__
![ceo](figures/ceo.png)

The perturbation function should be applicable to 

1. Neural networks with **undefined number of convolutional layers**.

   We solve this problem by allowing the function to be **shared across the convolutional layers**.

2. Convolutional layers with **undefined number of channels**.

   We solve this problem by **sharing the function across channels** and **using permutation-equivariant set encodings**. (Left on above figure)



Further, to adptively scale noise of each channel to different values for different dataset, we propose **batch-dependent scaling function**. (Right on above figure)



![model](figures/model.png)Finally, we combine two componets as above figure.  -->

## Prerequisites
We recommend to use attached Dockerfile.

## Running code

To run meta-training,
```
run.sh src
```

To run meta-testing,
```
run.sh tgt <gpu-to-use>
```

To change training configuration, change arguments at the top line in run.sh (ex. SRC_MODEL, TGT_DATA, ...)

## Citation
If you found the provided code useful, please cite our work.
```
@inproceedings{
    ryu2020metaperturb,
    title={MetaPerturb: Transferable Regularizer for Heterogeneous Tasks and Architectures},
    author={Jeong Un Ryu, JaeWoong Shin, Hae Beom Lee, Sung Ju Hwang},
    booktitle={NeurIPS},
    year={2020}
}
```