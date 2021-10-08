# AMP(Automatic Mixed-Precision)

AMP는 크게 두 가지 방법 존재
- Pytorch [ [doc](https://pytorch.org/docs/stable/amp.html) ]
- NVIDIA [ [doc](https://nvidia.github.io/apex/) ]


# Pytorch

`torch.cuda.amp.autocast_mode.autocast` 이라는 메세지가 나오면 OK

```python 
getattr(torch.cuda.amp, 'autocast')
```

**code example**

```python
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)

for input, target in data:
    optimizer.zero_grad()

    # Enables autocasting for the forward pass (model + loss)
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward()
    loss.backward()
    optimizer.step()
```

## Gradient Scaling

If the forward pass for a particular op has float16 inputs, the backward pass for that op will produce float16 gradients. Gradient values with small magnitudes may not be representable in float16. These values will flush to zero (“underflow”), **so the update for the corresponding parameters will be lost**.

To prevent underflow, “gradient scaling” multiplies the network’s loss(es) by a scale factor and invokes a backward pass on the scaled loss(es). Gradients flowing backward through the network are then scaled by the same factor. In other words, gradient values have a larger magnitude, so they don’t flush to zero.

Each parameter’s gradient (.grad attribute) should be unscaled before the optimizer updates the parameters, so the scale factor does not interfere with the learning rate.

**code example**

```python
# Creates model and optimizer in default precision
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), ...)
scaler = torch.cuda.amp.GradScaler()

for input, target in data:
    optimizer.zero_grad()

    # Enables autocasting for the forward pass (model + loss)
    with autocast():
        output = model(input)
        loss = loss_fn(output, target)

    # Exits the context manager before backward()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

# NVIDIA

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F91yfe%2FbtqFBfSoi4Y%2Fkq1KxHP1VN7NB0pICFK8U0%2Fimg.png)


```python
# Declare model and optimizer as usual, with default (FP32) precision
model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Allow Amp to perform casts as required by the opt_level
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
...
# loss.backward() becomes:
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

# Experiments

- Model: ResNet50
- Dataset: CIFAR10
- Batch Size : 128

# Results

- [Comparison AMP (wandb)](https://wandb.ai/tootouch/Comparison%20AMP?workspace=)


![W B Chart 10_8_2021, 3_53_45 PM](https://user-images.githubusercontent.com/37654013/136511689-cb349f95-1690-4615-b9a8-e5c5213f3669.png)

![W B Chart 10_8_2021, 3_53_04 PM](https://user-images.githubusercontent.com/37654013/136511679-f6b4083b-3b88-47ce-a646-481c0c71094e.png)

