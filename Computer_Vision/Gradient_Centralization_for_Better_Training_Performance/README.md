# Gradient Centralization (GC)

- Author: Jaehyuk Heo
- paper: [Gradient Centralization: A New Optimization Technique for Deep Neural Networks](https://arxiv.org/abs/2004.01461)
- official github: https://github.com/Yonghongwei/Gradient-Centralization

## Proposed Methods 

<p align='center'>
    <img width='500' src='https://github.com/Yonghongwei/Gradient-Centralization/raw/master/fig/gradient.png'>
    <img width='500' src='https://github.com/Yonghongwei/Gradient-Centralization/raw/master/fig/projected_Grad.png'>
</p>

## Experiments 

- Dataset: CIFAR100
- Model: ResNet-50

**Results**

0. Total Comparison
1. SGD vs SGD + GC
2. SGDW vs SGDW + GC
3. Adam vs Adam + GC
4. AdamW vs AdamW + GC

## 0. Total Comparison

<p align='center'>
    <img width='1500' src='https://user-images.githubusercontent.com/37654013/152784574-86d066ce-1868-4999-b328-d811e0442b60.png'>
</p>

## 1. SGD vs SGD + GC

<p align='center'>
    <img width='1500' src='https://user-images.githubusercontent.com/37654013/152785078-9a2279f0-3777-422d-910d-e29e53ef34ad.png'>
</p>

## 2. SGDW vs SGDW + GC

<p align='center'>
    <img width='1500' src='https://user-images.githubusercontent.com/37654013/152785004-185a006a-af5c-470b-ae0e-077da8164a06.png'>
</p>


## 3. Adam vs Adam + GC

<p align='center'>
    <img width='1500' src='https://user-images.githubusercontent.com/37654013/152784878-52db793f-642b-4618-8917-6b358757b7db.png'>
</p>

## 4. AdamW vs AdamW + GC

<p align='center'>
    <img width='1500' src='https://user-images.githubusercontent.com/37654013/152784766-db69952c-bac4-4617-b2f4-80851442cdb7.png'>
</p>