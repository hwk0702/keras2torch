# standard
python main.py --exp-name 'standard_resnet50_cifar'

# torch-amp
python main.py --exp-name 'torch-amp_resnet50_cifar' --torch-amp

# nvidia-apex-amp
python main.py --exp-name 'nvidia-apex-amp_resnet50_cifar' --apex