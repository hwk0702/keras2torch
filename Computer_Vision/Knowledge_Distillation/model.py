import torch 

class Teacher(torch.nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2) # padding 'same' do not exit...
        
        self.dense = torch.nn.Linear(4*4*512, 10)
        
    def forward(self, x):
        output = self.conv1(x)
        output = self.leakyrelu(output)
        output = self.max_pool(output)
        output = self.conv2(output)
        output = self.dense(output.view(output.size(0), -1))
        
        return output
    
class Student(torch.nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2) # padding 'same' do not exit...
        
        self.dense = torch.nn.Linear(4*4*32, 10)
        
    def forward(self, x):
        output = self.conv1(x)
        output = self.leakyrelu(output)
        output = self.max_pool(output)
        output = self.conv2(output)
        output = self.dense(output.view(output.size(0), -1))
        
        return output