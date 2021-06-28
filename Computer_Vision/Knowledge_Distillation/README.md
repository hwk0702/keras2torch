# Knowledge Distillation 

- **Editor**: Tootouch

<div align='center'>
    <img width="1195" alt="image" src="https://user-images.githubusercontent.com/37654013/118425747-c4890b80-b704-11eb-83cc-9e6cd56fad83.png"><br>
    The specific architecture of the benchmark knowledge distillation (Hinton et al., 2015).<br>
    Source: Knowledge Distillation: A Survey
</div>

**Reference Paper**

- Hinton, G., Vinyals, O., & Dean, J. (2015). [Distilling the knowledge in a neural network](https://arxiv.org/pdf/1503.02531.pdf). arXiv preprint arXiv:1503.02531.
- Tung, F., & Mori, G. (2019). [Similarity-preserving knowledge distillation](https://arxiv.org/pdf/1907.09682.pdf). In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1365-1374).
- Gou, J., Yu, B., Maybank, S. J., & Tao, D. (2021). [Knowledge distillation: A survey](https://arxiv.org/pdf/2006.05525.pdf). International Journal of Computer Vision, 1-31.


# Run

```
sh exp.sh
```


# Training Detail

**Models**
- Teacher: ResNet 18
- Student: Two CNN layer

**Hyperparamters**
- Epochs : 100
- Batch Size : 128
- Learning Rate : 0.1
  - Scheduler : CosineAnnealingLR

**Alpha and Temperature**

- Alpha : 0.1, 0.25, 0.5, 0.75, 0.9
- Temperature : 1, 2.5, 5, 7.5, 10, 20, 30, 40, 50

**Loss Function**

<div align='center'>
    <img alt="formula" src="https://render.githubusercontent.com/render/math?math=L%20=%20(1-\alpha)%20L_{CE}(y,%20\sigma(Z_S))%20%2B%20\alpha%20L_{CE}%20(\sigma(\frac{Z_S}{T}),%20\sigma(\frac{Z_T}{T}))" />
</div>




# Result

**Training History** : https://tensorboard.dev/experiment/rMGgbd12Q8yiPVQOtPg2FQ/#scalars

## Teacher and Student Model Performance

Model | Accuracy |
---|---|
Teacher (ResNet 18) | **0.9154**
Student (CNN 2 Layers)| 0.6128

## Student Performance Per Alpha and Tempertature

- CNN 2 Layers

|               |temperature1.0	 |temperature2.5	 |temperature5.0	 |temperature7.5	 |temperature10	 |temperature20	 |temperature30	 |temperature40	 |temperature50 |
|---|---|---|---|---|---|---|---|---|---|
|   **alpha0.1**	|0.6113	|0.6117	|0.6100	|0.6121	|0.6095	|0.6127	|0.6095	|0.6096	|0.6115|
|   **alpha0.25**	|0.6108	|0.6086	|0.6103	|0.6187	|0.6177	|0.6226	|0.6201	|0.6213	|0.6206|
|   **alpha0.5**	|**0.6118**	|0.6198	|0.6193	|0.6205	|**0.6255**|**0.6238**	|**0.6233**	|**0.6245**	|**0.6245**|
|   **alpha0.75**	|0.6106	|**0.6227**	|0.6227	|0.6215	|0.6227	|0.6214	|0.6228	|0.6205	|0.6208|
|   **alpha0.9**	|0.6122	|0.6194	|**0.6257**	|**0.6218**	|0.6178	|0.6150	|0.6146	|0.6136	|0.6136|

- ResNet 18
  
| | ResNet18 |
|--|--|
**alpha0.5 temperature10** |	0.9021
**alpha0.9 temperature5.0**  | 0.8812
		
