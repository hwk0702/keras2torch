# - learning rate 실험
# - no_grad()와 requires_grad = False의 차이를 실험
# 
# 1. None (lr = 0.001)
#     1.1 only no grad
#     1.2 only requires_grad = False
#     1.3 all
# 2. lr = 0.0001
#     2.1 only no grad
#     2.2 only requires_grad = False
#     2.3 all
# 3. lr = 0.00001
#     3.1 only no grad
#     3.2 only requires_grad = False
#     3.3 all
# 4. lr = 0.000001
#     4.1 only no grad
#     4.2 only requires_grad = False
#     4.3 all


nohup python -u main.py --lr 0.001 --requires_grad 
nohup python -u main.py --lr 0.001 --no_grad --requires_grad 
nohup python -u main.py --lr 0.001 
nohup python -u main.py --lr 0.001 --no_grad 
nohup python -u main.py --lr 0.0001 --requires_grad 
nohup python -u main.py --lr 0.0001 --no_grad --requires_grad 
nohup python -u main.py --lr 0.0001 
nohup python -u main.py --lr 0.0001 --no_grad 
nohup python -u main.py --lr 0.00001 --requires_grad 
nohup python -u main.py --lr 0.00001 --no_grad --requires_grad 
nohup python -u main.py --lr 0.00001 
nohup python -u main.py --lr 0.00001 --no_grad 
nohup python -u main.py --lr 0.000001 --requires_grad 
nohup python -u main.py --lr 0.000001 --no_grad --requires_grad 
nohup python -u main.py --lr 0.000001 
nohup python -u main.py --lr 0.000001 --no_grad 

