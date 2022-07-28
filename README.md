# keras2torch Study

## 1. 스터디 소개

Keras documentaion에 올라온 코드를 Pytorch로 코드 이전하는 스터디 입니다.

[keras documentaion examples](https://keras.io/examples/)

Original github address: https://github.com/keras-team/keras-io

시작 일자: 2021.03.04(목)

---

### Members


수료생(Previous members)

|                 Hyeongwon               |                 Subin                |                   Yonggi                   |                   Yookyung                   |
| :------------------------------------------: | :-----------------------------------------: | :----------------------------------------: | :----------------------------------------: |
| <img src="https://avatars.githubusercontent.com/u/31944451?s=400&u=5ee1388c2507ddddb5298eb608393032b4aad489&v=4" width=150px> | <img src="https://github.com/yukyunglee/Transformer_Survey_Study/blob/3254384e154ff2a3232a9fe723da36b1ceb92705/img/sb.png?raw=true" width=150px> | <img src="https://user-images.githubusercontent.com/68496320/168459826-6d01c007-d545-45d5-9601-cbc493971540.png" width=150px> | <img src="https://user-images.githubusercontent.com/68496320/174434997-38cd9083-b359-4b25-9696-9369596335a3.png" width=150px> |
|                   **[Github](https://github.com/hwk0702)**                   |                   **[Github](https://github.com/suubkiim)**                   |                   **[Github](https://github.com/animilux)**                   |                   **[Github](https://github.com/yookyungkho)**                   


참여자(Current members)

|              Jeongsub               |                   Jaehyuk                   |                   Sunwoo                   |                   Suzie                   |
| :------------------------------------------: | :-----------------------------------------: | :---------------------------------------------: | :---------------------------------------------: |
| <img src="https://avatars.githubusercontent.com/u/63832233?v=4" width=150px> | <img src="https://github.com/yukyunglee/Transformer_Survey_Study/blob/3254384e154ff2a3232a9fe723da36b1ceb92705/img/jh.png?raw=true" width=150px> | <img src="https://user-images.githubusercontent.com/68496320/168459726-ac6ce3dc-eeb8-485c-a870-10af89efe102.png" width=150px> | <img src="https://user-images.githubusercontent.com/68496320/174434838-8c470e18-b51f-4551-ae68-4e8032664555.png" width=150px> |
|               **[Github](https://github.com/jskim0406)**               |                   **[Github](https://github.com/TooTouch)**                   |                   **[Github](https://github.com/SunwooKimstar)**                   |                   **[Github](https://github.com/ohsuz)**                   |


---

### 목록
#### Computer Vision

- [Image segmentation with a U-Net-like architecture](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Image_Segmentation_Unet_Xception/%5BKJS%5DImage%20segmentation%20with%20a%20U-Net-like%20architecture(torch).ipynb) _[Jeongseob Kim]_
- 3D image classification from CT scans
- Semi-supervision and domain adaptation with AdaMatch
- Classification using Attention-based Deep Multiple Instance Learning (MIL).
- [Convolutional autoencoder for image denoising](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Convolutional%20autoencoder%20for%20image%20denoising) _[Jeongseob Kim]_
- Barlow Twins for Contrastive SSL
- Image Classification using BigTransfer (BiT)
- [OCR model for reading Captchas](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/OCR_model_for_reading_Captchas/captcha_ocr_KSB.ipynb) _[Subin Kim]_
- Compact Convolutional Transformers
- Consistency training with supervision
- Next-Frame Video Prediction with Convolutional
- Image classification with ConvMixer
- [CutMix data augmentation for image classification](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Augmentation/CutMix%20data%20augmentation%20for%20image%20classification.ipynb) _[Jaehyuk Heo]_
- Multiclass semantic segmentation using DeepLabV3+
- [Monocular depth estimation](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Monocular_depth_estimation/Monocular_depth_estimation.ipynb) _[Hyeongwon Kang]_
- Image classification with EANet (External Attention Transformer)
- FixRes: Fixing train-test resolution discrepancy
- [Grad-CAM class activation visualization](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Grad-CAM_class_activation_visualization/Grad-CAM%20class%20activation%20visualization%20HJH.ipynb) _[Jaehyuk Heo]_
- [Gradient Centralization for Better Training Performance] (https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Gradient_Centralization_for_Better_Training_Performance) _[Jaehyuk Heo]_
- Handwriting recognition
- Image Captioning _[Yonggi Jeong]_
- Image classification via fine-tuning with EfficientNet
- [Image classification with Vision Transformer](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Image_classification_with_Vision_Transformer/Image%20classification%20with%20Vision%20Transformer.ipynb) _[Jaehyuk Heo]_
- [Model interpretability with Integrated Gradients](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Model_Interpretability_with_Integrated_Gradients/Model%20interpretability%20with%20Integrated%20Gradients.ipynb) _[Jaehyuk Heo]_
- [Involutional neural networks](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Involutional%20neural%20networks/Involutional%20neural%20networks.ipynb) _[Subin Kim]_
- Keypoint Detection with Transfer Learning
- [Knowledge Distillation](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Knowledge_Distillation/Knowledge%20Distillation%20HJH.ipynb) _[Jaehyuk Heo]_
- Learning to Resize in Computer Vision
- Masked image modeling with Autoencoders
- [Metric learning for image similarity search](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Metric_Learning_for_Image_Similarity_Search/Metric%20learning%20for%20image%20similarity%20search%20HJH.ipynb) _[Jaehyuk Heo]_
- Low-light image enhancement using MIRNet
- MixUp augmentation for image classification
- Image classification with modern MLP models
- MobileViT: A mobile-friendly Transformer-based model for image classification
- Near-duplicate image search
- 3D volumetric rendering with NeRF
- Self-supervised contrastive learning with NNCLR
- Augmenting convnets with aggregated attention
- Image classification with Perceiver
- [Point cloud classification with PointNet](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Point_cloud_classification/Point_cloud_classification.ipynb) _[Hyeongwon Kang]_
- [Point cloud segmentation with PointNet](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Point_cloud_segmentation_with_PointNet/Point_cloud_segmentation_with_PointNet.ipynb) _[Hyeongwon Kang]_
- [RandAugment for Image Classification for Improved Robustness](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Augmentation/RandAugment/RandAugment.ipynb) _[Yonggi Jeong]_
- Few-Shot learning with Reptile
- [Object Detection with RetinaNet](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Object_Detection_with_RetinaNet) _[Jaehyuk Heo]_
- [Semantic Image Clustering](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Semantic_Image_Clustering/image_clustering.ipynb) _[Yonggi Jeong]_
- [Semi-supervised image classification using contrastive pretraining with SimCLR](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Semi-supervised%20image%20classification%20using%20contrastive%20pretraining%20with%20SimCLR) _[Subin Kim]_
- Image similarity estimation using a Siamese Network with a contrastive loss
- [Image similarity estimation using a Siamese Network with a triplet loss](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Triplet_loss/triplet_loss.ipynb) _[Yonggi Jeong]_
- [Self-supervised contrastive learning with SimSiam](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Self-supervised_contrastive_learning_with_SimSiam) _[Jaehyuk Heo]_
- Image Super-Resolution using an Efficient Sub-Pixel CNN
- [Supervised Contrastive Learning](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Supervised_contrastive_learning/supervised_contrastive_learning.ipynb) _[Subin Kim]_
- Image classification with Swin Transformers
- Learning to tokenize in Vision Transformers
- [Video Classification with Transformers](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Video_Classification_with_Transformers/Video_Classification_with_Transformers.ipynb) + [Video Vision Transformer](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Video_Classification_with_Transformers/ViViT.ipynb) _[Hyeongwon Kang]_
- [Visualizing what convnets learn](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Visualizing_what_convnets_learn) _[Jaehyuk Heo]_
- Train a Vision Transformer on small datasets
- Zero-DCE for low-light image enhancement
- [Model Soup](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Model_Soup) _[Jaehyuk Heo]_
- [Finetuning ViT with LoRA](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Finetuning_ViT_with_LoRA) _[Jaehyuk Heo]_

### Natural Language Processing

- Review Classification using Active Learning
- [Sequence to sequence learning for performing number addition](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Seq2seq_Number_Addition/seq2seq_number_addition_KYK.ipynb) _[Yookyung Kho]_
- [Bidirectional LSTM on IMDB](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Bidirectional_LSTM_on_IMDB/Text_classification_with_bi-LSTM_KJS.ipynb) _[Jeongseob Kim]_
- [Character-level recurrent sequence-to-sequence model](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Character-level_recurrent_sequence-to-sequence_model/Character_level_Machine_translator_with_seq2seq_KJS_3.ipynb) _[Jeongseob Kim]_
- [End-to-end Masked Language Modeling with BERT](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/End-to-end_Masked_Language_Modeling_with_BERT/mlm_and_finetune_with_bert_KSB.ipynb) _[Subin Kim]_
- Large-scale multi-label text classification
- [Multimodal entailment](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Multimodal_Entailment/multimodal_entailment_KYK.ipynb) _[Yookyung Kho]_
- [Named Entity Recognition using Transformers](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Named_Entity_Recognition%20using_Transformers/NER_using_Transformers_KSB.ipynb) _[Subin Kim]_
- [English-to-Spanish translation with a sequence-to-sequence Transformer](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Machine_Translation_via_seq2seq_Transformer/spn2eng_Translation_via_seq2seq_Transformer_KYK.ipynb) _[Yookyung Kho]_
- [Natural language image search with a Dual Encoder](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Natural_language_image%20search_with_a_Dual_Encoder/nl_image_search_KSB.ipynb) _[Subin Kim]_
- Using pre-trained word embeddings
- [Question Answering with Hugging Face Transformers](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Question_Answering_Huggingface/QA_huggingface_KYK.ipynb) _[Yookyung Kho]_
- [Semantic Similarity with BERT](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Semantic_Similarity_with_BERT/Semantic_Similarity_with_BERT_HJH.ipynb) _[Jaehyuk Heo]_
- Text classification with Switch Transformer _[Subin Kim]_
- [Text classification with Transformer](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Text_Classification_with_Transformers/text_classification_with_transformers_KYK.ipynb) _[Yookyung Kho]_
- [Text Extraction with BERT](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Text_Extraction_with_BERT/Text_Extraction_with_BERT_HJH.ipynb) _[Jaehyuk Heo]_
- Text Generation using FNet

#### Extra
- [TorchText introduction](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Extra/TorchText_introduction_KJS.ipynb) _[Jeongseob Kim]_
- [Table Pre-training with TapasForMaskedLM](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Table_pretraining_with_TAPAS/Table_Pretraining_with_TapasForMaskedLM_KYK.ipynb) _[Yookyung Kho]_


### Structured Data

- [Classification with Gated Residual and Variable Selection Networks](https://github.com/hwk0702/keras2torch/tree/main/Structured_Data/Classification_with_Gated_Residual_and_Variable_Selection_Networks) _[Hyeongwon Kang]_
- [Collaborative Filtering for Movie Recommendations](https://github.com/hwk0702/keras2torch/blob/main/Structured_Data/Collaborative_Filtering_for_Movie_Recommendations/Collaborative_Filtering_for_Movie_Recommendations.ipynb) _[Hyeongwon Kang]_
- Classification with Neural Decision Forests
- Imbalanced classification: credit card fraud detection
- [A Transformer-based recommendation system](https://github.com/hwk0702/keras2torch/blob/main/Structured_Data/Collaborative_Filtering_for_Movie_Recommendations/Collaborative_Filtering_for_Movie_Recommendations.ipynb) _[Hyeongwon Kang]_
- [Structured data learning with TabTransformer](https://github.com/hwk0702/keras2torch/tree/main/Structured_Data/Structured_data_learning_with_TabTransformer) _[Hyeongwon Kang]_
- Structured data learning with Wide, Deep, and Cross networks

### Timeseries

- [Timeseries anomaly detection using an Autoencoder](https://github.com/hwk0702/keras2torch/blob/main/Timeseries/Timeseries_anomaly_detection_using_an_Autoencoder/Timeseries_anomaly_detection_using_an_Autoencoder.ipynb) _[Hyeongwon Kang]_
- [Timeseries classification with a Transformer model](https://github.com/hwk0702/keras2torch/blob/main/Timeseries/Timeseries_classification_with_a_Transformer_model/Timeseries_classification_with_a_Transformer_model.ipynb) _[Hyeongwon Kang]_
- Traffic forecasting using graph neural networks and LSTM
- [Timeseries forecasting for weather prediction](https://github.com/hwk0702/keras2torch/tree/main/Timeseries/Timeseries_forecasting_for_weather_prediction) _[Hyeongwon Kang]_

### Audio Data

- Automatic Speech Recognition using CTC
- MelGAN-based spectrogram inversion using feature matching
- [Speaker Recognition](https://github.com/hwk0702/keras2torch/blob/main/Audio_Data/Speaker%20Recognition.ipynb) _[Subin Kim]_
- Automatic Speech Recognition with Transformer

### Generative Deep Learning

- [Variational AutoEncoder](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Variational_AutoEncoder) _[Jaehyuk Heo]_
- [DCGAN to generate face images](https://github.com/hwk0702/keras2torch/blob/main/Generative_Deep_Learning/DCGAN_to_generate_face_images/DCGAN_to_generate_face_images.ipynb) _[Hyeongwon Kang]_
- WGAN-GP overriding Model.train_step
- [Neural style transfer](https://github.com/hwk0702/keras2torch/blob/main/Generative_Deep_Learning/Neural_style_transfer/Neural_style_transfer.ipynb) _[Subin Kim]_
- [Deep Dream](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Deep_Dream) _[Jaehyuk Heo]_
- Neural Style Transfer with AdaIN
- [Conditional GAN](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Conditional_GAN) _[Yonggi Jeong]_
- [CycleGAN](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/CycleGAN) _[Yonggi Jeong]_
- Data-efficient GANs with Adaptive Discriminator Augmentation
- GauGAN for conditional image generation
- Character-level text generation with LSTM
- [PixelCNN](https://github.com/hwk0702/keras2torch/blob/main/Generative_Deep_Learning/PixclCNN/pixelcnn.ipynb) _[Jeongseob Kim]_
- [Density estimation using Real NVP](https://github.com/hwk0702/keras2torch/blob/main/Generative_Deep_Learning/Normalizing-Flow/RNVP/real-nvp-pytorch.ipynb) _[Jeongseob Kim]_
- Face image generation with StyleGAN
- [Text generation with a miniature GPT](https://github.com/hwk0702/keras2torch/blob/main/Generative_Deep_Learning/Text_generation_with_a_miniauture_GPT/Text_generation_with_a_miniauture_GPT_KSB.ipynb) _[Subin Kim]_
- Vector-Quantized Variational Autoencoders
- WGAN-GP with R-GCN for the generation of small molecular graphs

#### Extra

- [Distributions_TFP_Pyro](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Normalizing-Flow/Framework_practice/Distributions_TFP_Pyro) _[Jeongseob Kim]_
- [Non-linear Independent Component Estimation (NICE)](https://github.com/hwk0702/keras2torch/blob/main/Generative_Deep_Learning/Normalizing-Flow/NICE/NICE_codes.ipynb) _[Jeongseob Kim]_
- [Diffusion generative model(Tutorials)](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Score_Diffusion/Tutorial) _[Jeongseob Kim]_
- [Diffusion generative model(Examples - Swiss-roll, MNIST, F-MNIST, CELEBA)](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Score_Diffusion/Diffusion) _[Jeongseob Kim]_
- [Score based generative model(Tutorials)](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Score_Diffusion/Tutorial) _[Jeongseob Kim]_


### Reinforcement Learning

- [Actor Critic Method](https://github.com/hwk0702/keras2torch/blob/main/Reinforcement_Learning/Actor_Critic_Method/Actor_Critic_Method_KHW.ipynb) _[Hyeongwon Kang]_
- [Deep Deterministic Policy Gradient (DDPG)](https://github.com/hwk0702/keras2torch/blob/main/Reinforcement_Learning/DDPG/DDPG.ipynb) _[Hyeongwon Kang]_
- [Deep Q-Learning for Atari Breakout](https://github.com/hwk0702/keras2torch/blob/main/Reinforcement_Learning/Deep_Q_Learning_for_Atari_Breakout/Deep_Q_Learning_for_Atari_Breakout_KHW.ipynb) _[Hyeongwon Kang]_
- [Proximal Policy Optimization](https://github.com/hwk0702/keras2torch/blob/main/Reinforcement_Learning/Proximal_Policy_Optimization/Proximal_Policy_Optimization.ipynb) _[Hyeongwon Kang]_

### Graph Data

- Graph attention networks for node classification
- Node Classification with Graph Neural Networks
- Message-passing neural network for molecular property prediction
- Graph representation learning with node2vec

### Adversarial Attacks

- [Fast Gradient Sign Method](https://github.com/hwk0702/keras2torch/tree/main/Adversarial_Attack/Fast_Gradient_Sign_Method) _[Jaehyuk Heo]_
- [Projected Gradient Descent](https://github.com/hwk0702/keras2torch/tree/main/Adversarial_Attack/Projected_Gradient_Descent) _[Jaehyuk Heo]_

### Anomaly Detection

- [PatchCore](https://github.com/hwk0702/keras2torch/tree/main/Anomaly_Detection/PatchCore) _[Jaehyuk Heo]_

### Pytorch Accelerator

- [Automatic Mixed Precision](https://github.com/hwk0702/keras2torch/tree/main/Pytorch-Accelerator/AMP) _[Jaehyuk Heo]_ 
- [Gradient Accumulation](https://github.com/hwk0702/keras2torch/tree/main/Pytorch-Accelerator/gradient_accumulation) _[Jaehyuk Heo]_
- [Distributed Data Parallel](https://github.com/hwk0702/keras2torch/tree/main/Pytorch-Accelerator/DDP) _[Jaehyuk Heo]_
