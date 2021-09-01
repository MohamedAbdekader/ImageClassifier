Open the folder where you saved the data files and the python files 

******
For task 1, I used trainNN.py and testNN.py

To train the network, run: python3 trainNN.py num

Where num is the range you want to train for, so type 10 if you want to train 10 per class.

To test the network, run: python3 testNN.py

I used test_batch_osu to test for it; therefore, it's hard coded in the program.

******
For the python program that allows to test your network on a dataset saved in the same format as the CIFAR batches, I used CIFARTest.py. I used it to test the CIFAR test_batch and ImageNet_test_batch

I assumed you will use a saved network, so no need to train for it.

To test the network, run: python3 CIFARTest.py network data

Where network is the saved network such as "imageNet.pth" and data is the file with the same design as data_batch_X from CIFAR data

******
For the python program from task 4, I assumed you will use a saved network; therefore, I just used task4Test.py

To test the network, run: python3 task4Test.py network image

Where network is the saved network such as "myNet.pth" and image is the image you want to input such as "car.jpg". I have also provided the picture that I used. You should use a 32x32 image.

******
For task 2, I tested my trained network, imageNet.pth, using ImageNet_test_batch and test_batch. I received 18.89% accuracy for test_batch_osu and 17.625% for ImageNet_test_batch, so the trained network got better results from the CIFAR test than from the Image net batch. 

******
*For task 3, I create 3 mini batches instead of 4 and changed momentum to 1. I saved it in firstArch.pth.
I received 12.5% for ImageNet_test_batch and 10.0% for CIFAR test_batch.

*Then, I changed: 
self.fc1 = nn.Linear(16*5*5,120)
self.fc2 = nn.Linear(120,84)
self.fc3 = nn.Linear(84,10)

To:

self.fc1 = nn.Linear(16*5*5,150)
self.fc2 = nn.Linear(150,82)
self.fc3 = nn.Linear(82,10)

I still received 12.5% for ImageNet_test_batch and 10.0% for CIFAR test_batch. I saved the network in secondArch.pth

* I changed optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5) to  
 optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

And received 18.125%  for ImageNet_test_batch and 21.59% for CIFAR test_batch. I saved the network in thirdArch.pth

The thirdArch.pth received the best accuracy for imageNet_test_batch.

*******
I tried task 4 with car.jpg and it predicted that it's a truck using imageNet.pth network, so truck and car are similar maybe?

*******
Total time spent: ~100 hours -> implementing and running the program
