from memory_profiler import profile

fp=open('memory_profiler.log','w+')
@profile(stream=fp)
def memtest():

    import datetime

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint as checkpoint

    class Neural_Network_Layers(nn.Module):

        def __init__(self):

            super(Neural_Network_Layers, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
            self.b1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU(inplace=True)

            self.conv20 = nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False)
            self.b20 = nn.BatchNorm2d(32)
            self.relu20 = nn.ReLU()
            self.conv21 = nn.Conv2d(32, 64, 1, 1, 0, bias=False)
            self.b21 = nn.BatchNorm2d(64)
            self.relu21 = nn.ReLU()

            self.conv30 = nn.Conv2d(64, 64, 3, 2, 1, groups=64, bias=False)
            self.b30 = nn.BatchNorm2d(64)
            self.relu30 = nn.ReLU()
            self.conv31 = nn.Conv2d(64, 128, 1, 1, 0, bias=False)
            self.b31 = nn.BatchNorm2d(128)
            self.relu31 = nn.ReLU()

            self.conv40 = nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False)
            self.b40 = nn.BatchNorm2d(128)
            self.relu40 = nn.ReLU()
            self.conv41 = nn.Conv2d(128, 128, 1, 1, 0, bias=False)
            self.b41 = nn.BatchNorm2d(128)
            self.relu41 = nn.ReLU()

            self.conv50 = nn.Conv2d(128, 128, 3, 2, 1, groups=128, bias=False)
            self.b50 = nn.BatchNorm2d(128)
            self.relu50 = nn.ReLU()
            self.conv51 = nn.Conv2d(128, 256, 1, 1, 0, bias=False)
            self.b51 = nn.BatchNorm2d(256)
            self.relu51 = nn.ReLU()

            self.conv60 = nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False)
            self.b60 = nn.BatchNorm2d(256)
            self.relu60 = nn.ReLU()
            self.conv61 = nn.Conv2d(256, 256, 1, 1, 0, bias=False)
            self.b61 = nn.BatchNorm2d(256)
            self.relu61 = nn.ReLU()

            self.conv70 = nn.Conv2d(256, 256, 3, 2, 1, groups=256, bias=False)
            self.b70 = nn.BatchNorm2d(256)
            self.relu70 = nn.ReLU()
            self.conv71 = nn.Conv2d(256, 512, 1, 1, 0, bias=False)
            self.b71 = nn.BatchNorm2d(512)
            self.relu71 = nn.ReLU()

            self.conv80 = nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False)
            self.b80 = nn.BatchNorm2d(512)
            self.relu80 = nn.ReLU()
            self.conv81 = nn.Conv2d(512, 512, 1, 1, 0, bias=False)
            self.b81 = nn.BatchNorm2d(512)
            self.relu81 = nn.ReLU()

            self.conv90 = nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False)
            self.b90 = nn.BatchNorm2d(512)
            self.relu90 = nn.ReLU()
            self.conv91 = nn.Conv2d(512, 512, 1, 1, 0, bias=False)
            self.b91 = nn.BatchNorm2d(512)
            self.relu91 = nn.ReLU()

            self.conv100 = nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False)
            self.b100 = nn.BatchNorm2d(512)
            self.relu100 = nn.ReLU()
            self.conv101 = nn.Conv2d(512, 512, 1, 1, 0, bias=False)
            self.b101 = nn.BatchNorm2d(512)
            self.relu101 = nn.ReLU()

            self.conv110 = nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False)
            self.b110 = nn.BatchNorm2d(512)
            self.relu110 = nn.ReLU()
            self.conv111 = nn.Conv2d(512, 512, 1, 1, 0, bias=False)
            self.b111 = nn.BatchNorm2d(512)
            self.relu111 = nn.ReLU()

            self.conv120 = nn.Conv2d(512, 512, 3, 1, 1, groups=512, bias=False)
            self.b120 = nn.BatchNorm2d(512)
            self.relu120 = nn.ReLU()
            self.conv121 = nn.Conv2d(512, 512, 1, 1, 0, bias=False)
            self.b121 = nn.BatchNorm2d(512)
            self.relu121 = nn.ReLU()

            self.conv130 = nn.Conv2d(512, 512, 3, 2, 1, groups=512, bias=False)
            self.b130 = nn.BatchNorm2d(512)
            self.relu130 = nn.ReLU()
            self.conv131 = nn.Conv2d(512, 1024, 1, 1, 0, bias=False)
            self.b131 = nn.BatchNorm2d(1024)
            self.relu131 = nn.ReLU()

            self.conv140 = nn.Conv2d(1024, 1024, 3, 1, 1, groups=1024, bias=False)
            self.b140 = nn.BatchNorm2d(1024)
            self.relu140 = nn.ReLU()
            self.conv141 = nn.Conv2d(1024, 1024, 1, 1, 0, bias=False)
            self.b141 = nn.BatchNorm2d(1024)
            self.relu141 = nn.ReLU()

            self.avgpl = nn.AvgPool2d(7)

        def forward(self, a):
            # FORWARD PASS START - DO NOT REMOVE THIS COMMENT (This is used to parse memory details of forward pass from the memory profiler output)
            out = self.conv1(a)
            out = self.b1(out)
            out = self.relu1(out)
            out = self.conv20(out)
            out = self.b20(out)
            out = self.relu20(out)
            out = self.conv21(out)
            out = self.b21(out)
            out = self.relu21(out)
            out = self.conv30(out)
            out = self.b30(out)
            out = self.relu30(out)
            out = self.conv31(out)
            out = self.b31(out)
            out = self.relu31(out)
            out = self.conv40(out)
            out = self.b40(out)
            out = self.relu40(out)
            out = self.conv41(out)
            out = self.b41(out)
            out = self.relu41(out)
            out = self.conv50(out)
            out = self.b50(out)
            out = self.relu50(out)
            out = self.conv51(out)
            out = self.b51(out)
            out = self.relu51(out)
            out = self.conv60(out)
            out = self.b60(out)
            out = self.relu60(out)
            out = self.conv61(out)
            out = self.b61(out)
            out = self.relu61(out)
            out = self.conv70(out)
            out = self.b70(out)
            out = self.relu70(out)
            out = self.conv71(out)
            out = self.b71(out)
            out = self.relu71(out)
            out = self.conv80(out)
            out = self.b80(out)
            out = self.relu80(out)
            out = self.conv81(out)
            out = self.b81(out)
            out = self.relu81(out)
            out = self.conv90(out)
            out = self.b90(out)
            out = self.relu90(out)
            out = self.conv91(out)
            out = self.b91(out)
            out = self.relu91(out)
            out = self.conv100(out)
            out = self.b100(out)
            out = self.relu100(out)
            out = self.conv101(out)
            out = self.b101(out)
            out = self.relu101(out)
            out = self.conv110(out)
            out = self.b110(out)
            out = self.relu110(out)
            out = self.conv111(out)
            out = self.b111(out)
            out = self.relu111(out)
            out = self.conv120(out)
            out = self.b120(out)
            out = self.relu120(out)
            out = self.conv121(out)
            out = self.b121(out)
            out = self.relu121(out)
            out = self.conv130(out)
            out = self.b130(out)
            out = self.relu130(out)
            out = self.conv131(out)
            out = self.b131(out)
            out = self.relu131(out)
            out = self.conv140(out)
            out = self.b140(out)
            out = self.relu140(out)
            out = self.conv141(out)
            out = self.b141(out)
            out = self.relu141(out)
            out  = self.avgpl(out)
            # FORWARD PASS END - DO NOT REMOVE THIS COMMENT (This is used to parse memory details of forward pass from the memory profiler output)
            return out

    class Fully_Connected_Layers(nn.Module):

        def __init__(self):

            super(Fully_Connected_Layers, self).__init__()
            self.fc1 = nn.Linear(1024, 1000)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            out = self.softmax(self.fc1(x))
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
    N = 32
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
