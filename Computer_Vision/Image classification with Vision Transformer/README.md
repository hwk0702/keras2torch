# ViT (Vision Transformer)

**Author**: [Tootouch](https://github.com/TooTouch)

**Image classification with Vision Transformer**

- paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)


**Keras Doc: [Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/)**

# TIL

**Keras Documentation에서 이상한 점**

- Augmentation이 Test set에도 적용
- CLS token 없음
- Transformer의 FeedForward와 MLP header를 동인한 파라미터로 사용

**이번 예제에서 배운 점**

- einops나 einsum이 굉장히 많이 활용되고 있음
- Google에서 tensorflow 말고 flax 사용 많이 하는 듯
- Pytorch의 chunk function
- Activation function으로 ReLU에에 GELU로 대체되고 있다
- Optimizer로 ViT 계열에서는 AdamW가 대부분 사용됨

# Requirement

```
torch=1.7.0
pytorch_pretrained_vit=0.0.7
```

# Dataset

- CIFAR 100 

# Model Architecture

![](https://camo.githubusercontent.com/8798b9bd20667947ec2ef5f2a1ae51b927377413/68747470733a2f2f757365722d696d616765732e67697468756275736572636f6e74656e742e636f6d2f33373635343031332f3132343236343033382d66383135646430302d646236652d313165622d383034322d6131323065383230613832332e706e67)

# Experiments

I built the model based ViT-B. The difference between setting 1 and setting 2 was image size and patch size. Fine-tuning model learned better when the images in downsteam task have higher resolution than pre-trained images. I used gradients accumulation becaused of large batch size in limited gpu memory.

**Setting 1: model from scratch**

```python
class Config:
    datadir            = '../data'
    image_size         = 72
    patch_size         = 6
    seed               = 223
    epochs             = 100
    batch_size         = 16
    accumulation_steps = 16
    depth              = 12
    dim                = 768 
    mlp_dim            = 3072
    heads              = 12
    dropout            = 0.1
    emb_dropout        = 0.1
    lr                 = 0.001
    num_classes        = 100
    num_workers        = 8
    gpu_num            = 0
```

**Setting 2: fine-tuning model from pretrained model with ImageNet 1k**

I used pre-trained model with ImageNet 1k, which based ViT-B/16.

```python
class Config:
    datadir            = '../data'
    image_size         = 384
    patch_size         = 16
    seed               = 223
    epochs             = 30
    batch_size         = 8
    accumulation_steps = 64
    lr                 = 0.001
    num_classes        = 100
    num_workers        = 8
    gpu_num            = 1
```

## Augmentation

```python
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((config.image_size,)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(degrees=0.02),
    torchvision.transforms.RandomResizedCrop(size=(config.image_size,), scale=(0.8,1), ratio=(1,1.)),
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((config.image_size,)),
])
```


# Results

Model | Pretrained | Resolution in pretraing | Resolution in fine-tuning | Batch size | Epochs or Steps | Top-1 Acc % (train) | Top-1 Acc % (test) | Top-5 Acc % (test)
---|---|---|---|---|---|---|---|---
ViT-B/16<br>(in paper) | ImageNet 1k | 224x224 | 384x384 | 512 | 10,000 steps | - |  87.13 | - 
ViT-B/6 | x | - | 72x72 | 256 | 100 epochs | 82.83 | 23.93 | 49.68
ViT-B/16 | ImageNet 1k | 384x384 | 384x384 | 512 | 30 epochs | 86.79 | 86.54 | 98.17

## Setting 1

**History**

Overffting was occured. In my opinion, more powerful regularization and augmentation was needed, such as AutoAug, CutMix, Mixup and so on.

![image](https://user-images.githubusercontent.com/37654013/124404516-3cbf9500-dd76-11eb-99e7-09dad0b9c22a.png)

**Visualize positional embedding**z

![image](https://user-images.githubusercontent.com/37654013/124404648-c8392600-dd76-11eb-9550-65b5809312bc.png)


## Setting 2

**History**

![image](https://user-images.githubusercontent.com/37654013/124405621-16035d80-dd7a-11eb-9e19-8b6a413bedc6.png)

**Visualization positional embedding**

![image](https://user-images.githubusercontent.com/37654013/124404695-ffa7d280-dd76-11eb-9eee-687123e9b761.png)
