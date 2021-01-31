from memory_profiler import profile

fp=open('memory_profiler.log','w+')
@profile(stream=fp)
def memtest():

    import datetime

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint as checkpoint

    # conv3x3 wrapper function
    def conv3x3(in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    class Neural_Network_Layers(nn.Module):

        def __init__(self):

            super(Neural_Network_Layers, self).__init__()
            #0_0
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU()
            self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            #1_0
            self.conv2 = conv3x3(in_planes=64, out_planes=64, stride=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU()
            self.conv3 = conv3x3(in_planes=64, out_planes=64)
            self.bn3 = nn.BatchNorm2d(64)
            self.relu2 = nn.ReLU()
            #1_1
            self.conv4 = conv3x3(in_planes=64, out_planes=64, stride=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.relu3 = nn.ReLU()
            self.conv5 = conv3x3(in_planes=64, out_planes=64)
            self.bn5 = nn.BatchNorm2d(64)

            #2_0
            self.conv6 = conv3x3(in_planes=64, out_planes=128, stride=2)
            self.bn6 = nn.BatchNorm2d(128)
            self.relu4 = nn.ReLU()
            self.conv7 = conv3x3(in_planes=128, out_planes=128)
            self.bn7 = nn.BatchNorm2d(128)
            self.downsample20 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128),
            )
            self.relu4 = nn.ReLU()
            #2_1
            self.conv8 = conv3x3(in_planes=128, out_planes=128)
            self.bn8 = nn.BatchNorm2d(128)
            self.relu5 = nn.ReLU()
            self.conv9 = conv3x3(in_planes=128, out_planes=128)
            self.bn9 = nn.BatchNorm2d(128)
            #3_0
            self.conv10 = conv3x3(in_planes=128, out_planes=256, stride=2)
            self.bn10 = nn.BatchNorm2d(256)
            self.relu6 = nn.ReLU()
            self.conv11 = conv3x3(in_planes=256, out_planes=256)
            self.bn11 = nn.BatchNorm2d(256)
            self.downsample30 = nn.Sequential(
                nn.Conv2d(128, 256,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256),
            )
            #3_1
            self.conv12 = conv3x3(in_planes=256, out_planes=256)
            self.bn12 = nn.BatchNorm2d(256)
            self.relu7 = nn.ReLU()
            self.conv13 = conv3x3(in_planes=256, out_planes=256)
            self.bn13 = nn.BatchNorm2d(256)
            #4_0
            self.conv14 = conv3x3(in_planes=256, out_planes=512, stride=2)
            self.bn14 = nn.BatchNorm2d(512)
            self.relu8 = nn.ReLU()
            self.conv15 = conv3x3(in_planes=512, out_planes=512)
            self.bn15 = nn.BatchNorm2d(512)
            self.downsample40 = nn.Sequential(
                nn.Conv2d(256, 512,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
            )
            #4_1
            self.conv16 = conv3x3(in_planes=512, out_planes=512)
            self.bn16 = nn.BatchNorm2d(512)
            self.relu9 = nn.ReLU()
            self.conv17 = conv3x3(in_planes=512, out_planes=512)
            self.bn17 = nn.BatchNorm2d(512)
            #0_1
            self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)


        def forward(self, a):
            # FORWARD PASS START - DO NOT REMOVE THIS COMMENT (This is used to parse memory details of forward pass from the memory profiler output)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.mp1(out)
            out = self.bn3(self.conv3(self.relu2(self.bn2(self.conv2(out))))) + out # If we need to add residual after some layers all the layers inbetween are considered as a single layer, hence the expression is in a single line
            out = self.relu2(out)
            out = self.bn5(self.conv5(self.relu3(self.bn4(self.conv4(out))))) + out # If we need to add residual after some layers all the layers inbetween are considered as a single layer, hence the expression is in a single line
            out = self.relu3(out)
            out = self.bn7(self.conv7(self.relu4(self.bn6(self.conv6(out))))) + self.downsample20(out) # If we need to add residual after some layers all the layers inbetween are considered as a single layer, hence the expression is in a single line
            out = self.relu4(out)
            out = self.bn9(self.conv9(self.relu5(self.bn8(self.conv8(out))))) + out # If we need to add residual after some layers all the layers inbetween are considered as a single layer, hence the expression is in a single line
            out = self.relu5(out)
            out = self.bn11(self.conv11(self.relu6(self.bn10(self.conv10(out))))) + self.downsample30(out)  # If we need to add residual after some layers all the layers inbetween are considered as a single layer, hence the expression is in a single line
            out = self.relu6(out)
            out = self.bn13(self.conv13(self.relu7(self.bn12(self.conv12(out))))) + out # If we need to add residual after some layers all the layers inbetween are considered as a single layer, hence the expression is in a single line
            out = self.relu7(out)
            out = self.bn15(self.conv15(self.relu8(self.bn14(self.conv14(out))))) + self.downsample40(out)
            out = self.relu8(out)
            out = self.bn17(self.conv17(self.relu9(self.bn16(self.conv16(out))))) + out
            out = self.relu9(out)
            out = self.avgpool(out)
            # FORWARD PASS END - DO NOT REMOVE THIS COMMENT (This is used to parse memory details of forward pass from the memory profiler output)
            return out

    class Fully_Connected_Layers(nn.Module):

        def __init__(self):

            super(Fully_Connected_Layers, self).__init__()
            self.fc1 = nn.Linear(in_features=512 , out_features=1000)

        def forward(self, x):
            out = self.fc1(x)
            return out

    class My_Nerual_Network(nn.Module):
        def __init__(self):
            super(My_Nerual_Network, self).__init__()

            self.neural_network = Neural_Network_Layers()
            self.final_module = Fully_Connected_Layers()

        def forward(self, x):
            out = self.neural_network(x)
            out = out.reshape(out.size(0), -1)
            out = self.final_module(out)
            return out

    model = My_Nerual_Network()

    learning_rate = 0.001
    # num_epochs & minibatch_epochs set to 1 for analyzing memory requirements per iteration.
    num_epochs = 1
    minibatch_epochs = 1
    # Set minibatch_size
    N = 64
    # Set Input and Output Specifications here
    x = torch.ones(N, 3, 224, 224, requires_grad=True)
    target = torch.ones(N).type("torch.LongTensor")
    
    input_image = x
    labels = target
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    times = []

    for epoch in range(num_epochs):

        if epoch == 0 :
            time = datetime.datetime.now()
            times.append(str(time))
        for i in range(minibatch_epochs):
            out = model(input_image)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch == num_epochs-1 :
            time = datetime.datetime.now()
            times.append(str(time))

memtest()
