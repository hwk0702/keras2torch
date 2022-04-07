# Model Soup

Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time ( [link](https://arxiv.org/abs/2203.05482) )


**Code reference**

- [timm](https://github.com/rwightman/pytorch-image-models)
- https://github.com/kuangliu/pytorch-cifar


# Resuls

**ViT-T/16**

pretrained on ImageNet-21K and fine-tuning on CIFAR10


```bash
# greedy soup
first acc: 96.53%
acc of 0 ingradients: 96.79%
acc of 1 ingradients: 96.66%
acc of 2 ingradients: 96.70%
acc of 3 ingradients: 96.74%
acc of 4 ingradients: 96.90%
```

acc | loss | ckp_path
---|---|---
96.53 | 0.1119 | ./pytorch-cifar/checkpoint/vit_tiny_patch16_224_in21k_sgd_lr-0.003_seed-2.pth
96.5  | 0.1153 | ./pytorch-cifar/checkpoint/vit_tiny_patch16_224_in21k_sgd_lr-0.003_seed-3.pth
96.66 | 0.1097 | ./pytorch-cifar/checkpoint/vit_tiny_patch16_224_in21k_sgd_lr-0.003_seed-5.pth
96.75 | 0.1131 | ./pytorch-cifar/checkpoint/vit_tiny_patch16_224_in21k_sgd_lr-0.003_seed-4.pth
96.83 | 0.1095 | ./pytorch-cifar/checkpoint/vit_tiny_patch16_224_in21k_sgd_lr-0.003_seed-1.pth
96.66 | 0.1095 | ./pytorch-cifar/checkpoint/vit_tiny_patch16_224_in21k_sgd_lr-0.003_seed-0.pth
**96.9**  | 0.1061 | soup (Um~ Good)


**ResNet50**

Pretrained on ImageNet-1K and fine-tuning on CIFAR10

I think that models with batch normalization is not good as ingradients

```bash
# greedy soup
first acc: 91.27%
acc of 0 ingradients: 14.55%
acc of 1 ingradients: 14.07%
acc of 2 ingradients: 9.08%
acc of 3 ingradients: 10.10%
acc of 4 ingradients: 10.01%
```

acc | loss | ckp_path
---|---|---
91.27 | 0.3062 | ./pytorch-cifar/checkpoint/resnet50d_sgd_lr-0.1_seed-4.pth
91.35 | 0.3016 | ./pytorch-cifar/checkpoint/resnet50d_sgd_lr-0.1_seed-1.pth
91.34 | 0.302, | /pytorch-cifar/checkpoint/resnet50d_sgd_lr-0.1_seed-3.pth
91.29 | 0.3015 | ./pytorch-cifar/checkpoint/resnet50d_sgd_lr-0.1_seed-2.pth
91.35 | 0.3004 | ./pytorch-cifar/checkpoint/resnet50d_sgd_lr-0.1_seed-0.pth
91.36 | 0.3071 | ./pytorch-cifar/checkpoint/resnet50d_sgd_lr-0.1_seed-5.pth
**91.27** | 0.3062 | soup (...)

