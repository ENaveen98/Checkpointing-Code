### n_checkpoints=10 configuration
### NOTE: n_checkpoints is the maximum no. of checkpoints that can be used

def memtest():

    import torch.nn as nn
    from torch.nn import functional as F
    from torch.autograd import Variable
    import torch
    import torch.utils.checkpoint as checkpoint

    class Segment_1(nn.Module):

        def __init__(self):
            super(Segment_1, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
            self.b1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv20 = nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False)

        def forward(self, x):
            out = self.conv1(x)
            out = self.b1(out)
            out = self.relu1(out)
            out = self.conv20(out)
            return out

    class Segment_2(nn.Module):

        def __init__(self):
            super(Segment_2, self).__init__()
            self.b20 = nn.BatchNorm2d(32)
            self.relu20 = nn.ReLU()
            self.conv21 = nn.Conv2d(32, 64, 1, 1, 0, bias=False)

        def forward(self, x):
            out = self.b20(x)
            out = self.relu20(out)
            out = self.conv21(out)
            return out

    class Segment_3(nn.Module):

        def __init__(self):
            super(Segment_3, self).__init__()
            self.b21 = nn.BatchNorm2d(64)
            self.relu21 = nn.ReLU()
            self.conv30 = nn.Conv2d(64, 64, 3, 2, 1, groups=64, bias=False)
            self.b30 = nn.BatchNorm2d(64)

        def forward(self, x):
            out = self.b21(x)
            out = self.relu21(out)
            out = self.conv30(out)
            out = self.b30(out)
            return out

    class Segment_4(nn.Module):

        def __init__(self):
            super(Segment_4, self).__init__()
            self.relu30 = nn.ReLU()
            self.conv31 = nn.Conv2d(64, 128, 1, 1, 0, bias=False)
            self.b31 = nn.BatchNorm2d(128)
            self.relu31 = nn.ReLU()

        def forward(self, x):
            out = self.relu30(x)
            out = self.conv31(out)
            out = self.b31(out)
            out = self.relu31(out)
            return out

    class Segment_5(nn.Module):

        def __init__(self):
            super(Segment_5, self).__init__()
            self.conv40 = nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False)
            self.b40 = nn.BatchNorm2d(128)
            self.relu40 = nn.ReLU()
            self.conv41 = nn.Conv2d(128, 128, 1, 1, 0, bias=False)

        def forward(self, x):
            out = self.conv40(x)
            out = self.b40(out)
            out = self.relu40(out)
            out = self.conv41(out)

            return out

    class Segment_6(nn.Module):

        def __init__(self):
            super(Segment_6, self).__init__()
            self.b41 = nn.BatchNorm2d(128)
            self.relu41 = nn.ReLU()

            self.conv50 = nn.Conv2d(128, 128, 3, 2, 1, groups=128, bias=False)
            self.b50 = nn.BatchNorm2d(128)
            self.relu50 = nn.ReLU()
            self.conv51 = nn.Conv2d(128, 256, 1, 1, 0, bias=False)
            self.b51 = nn.BatchNorm2d(256)

        def forward(self, x):
            out = self.b41(x)
            out = self.relu41(out)

            out = self.conv50(out)
            out = self.b50(out)
            out = self.relu50(out)
            out = self.conv51(out)
            out = self.b51(out)

            return out

    class Segment_7(nn.Module):

        def __init__(self):
            super(Segment_7, self).__init__()
            self.relu51 = nn.ReLU()
            self.conv60 = nn.Conv2d(256, 256, 3, 1, 1, groups=256, bias=False)
            self.b60 = nn.BatchNorm2d(256)
            self.relu60 = nn.ReLU()
            self.conv61 = nn.Conv2d(256, 256, 1, 1, 0, bias=False)
            self.b61 = nn.BatchNorm2d(256)
            self.relu61 = nn.ReLU()

            self.conv70 = nn.Conv2d(256, 256, 3, 2, 1, groups=256, bias=False)

        def forward(self, x):
            out = self.relu51(x)
            out = self.conv60(out)
            out = self.b60(out)
            out = self.relu60(out)
            out = self.conv61(out)
            out = self.b61(out)
            out = self.relu61(out)

            out = self.conv70(out)

            return out

    class Segment_8(nn.Module):

        def __init__(self):
            super(Segment_8, self).__init__()
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

        def forward(self, x):
            out = self.b70(x)
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

            return out

    class Segment_9(nn.Module):

        def __init__(self):
            super(Segment_9, self).__init__()
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

        def forward(self, x):
            
            out = self.conv91(x)
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

            return out

    class Segment_10(nn.Module):

        def __init__(self):
            super(Segment_10, self).__init__()
            
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

        def forward(self, x):
            
            out = self.b111(x)
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
            return out

    class Fully_Connected_Layers(nn.Module):

        def __init__(self):

            super(Fully_Connected_Layers, self).__init__()
            self.fc1 = nn.Linear(1024, 1000)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            out = self.softmax(self.fc1(x))
            return out

    class MN(nn.Module):
        def __init__(self):
            super(MN, self).__init__()

            # The module that we want to checkpoint
            self.module1 = Segment_1()
            self.module2 = Segment_2()
            self.module3 = Segment_3()
            self.module4 = Segment_4()
            self.module5 = Segment_5()
            self.module6 = Segment_6()
            self.module7 = Segment_7()
            self.module8 = Segment_8()
            self.module9 = Segment_9()
            self.module10 = Segment_10()
            self.final_module = Fully_Connected_Layers()

        def custom(self, module):
            def custom_forward(*inputs):
                inputs = module(inputs[0])
                return inputs
            return custom_forward

        def forward(self, x):
            print(x.shape)
            out = checkpoint.checkpoint(self.custom(self.module1), x)
            print(out.shape)
            out = checkpoint.checkpoint(self.custom(self.module2), out)
            print(out.shape)
            out = checkpoint.checkpoint(self.custom(self.module3), out)
            print(out.shape)
            out = checkpoint.checkpoint(self.custom(self.module4), out)
            print(out.shape)
            out = checkpoint.checkpoint(self.custom(self.module5), out)
            print(out.shape)
            out = checkpoint.checkpoint(self.custom(self.module6), out)
            print(out.shape)
            out = checkpoint.checkpoint(self.custom(self.module7), out)
            print(out.shape)
            out = checkpoint.checkpoint(self.custom(self.module8), out)
            print(out.shape)
            out = checkpoint.checkpoint(self.custom(self.module9), out)
            print(out.shape)
            out = checkpoint.checkpoint(self.custom(self.module10), out)
            print(out.shape)
            out = out.reshape(out.size(0), -1)
            print(out.shape)
            out = self.final_module(out)
            print(out.shape)
            return out

    model = MN()

    learning_rate = 0.001
    num_epochs = 1
    minibatch_epochs = 1
    N = 32 #minibatch_size
    x = torch.ones(N, 3, 224, 224, requires_grad=True)
    target = torch.ones(N).type("torch.LongTensor")
    input_image = x
    labels = target

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    import cProfile, pstats, io
    import datetime

    def profile(fnc):

        """A decorator that uses cProfile to profile a function"""

        def inner(*args, **kwargs):

            pr = cProfile.Profile()
            pr.enable()
            retval = fnc(*args, **kwargs)
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
            return retval

        return inner

    times = []

    for epoch in range(num_epochs):

        if epoch == 0 :
            time = datetime.datetime.now()
            times.append(str(time))
        for i in range(minibatch_epochs):
            #input("Press Enter to continue...to forward")
            out = model(input_image)
            loss = criterion(out, labels)
            #input("Press Enter to continue...to backward")
            optimizer.zero_grad()
            loss.backward()
            #input("Press Enter to continue...to update")
            optimizer.step()
            if (i+1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, minibatch_epochs, loss.item()))
        if epoch == num_epochs-1 :
            time = datetime.datetime.now()
            times.append(str(time))

    print(times)

memtest()
