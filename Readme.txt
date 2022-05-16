Das ist das Finale Modell.
Main.py geht alle wichtigen Schritte durch.
Zuerst haben wir mit einem ResNext32x8d alle 10 000 Bilder in Feature-Vektoren transformiert und in ein CSV geschrieben. 
Im Verlauf des Programms, werden die tripletts geladen, und in einem custom_dataset, die zugehörigen Feature-Vektoren gespeichert.
Das Ausgewählte Modell ist ein 6-Layer Netzwerk aus abwechselnden Linear layers und LeakyReLu's. 
Trainiert wird es mit dem tripletloss (die forward-Methode nimmt 3 inputs und macht daraus 3 outputs).
Das Modell wird dann mit einem pytorch dataloader/dataset trainiert und macht dann predictions.

Main.py: main
prediction.py: makes predictions with pre-loaded model
evaluation.py: functions to evaluate and plot model every epoch during training
train.py: train loop Definition
triplet_loss_network.py: archtecture of the simaese network used (class triplet_net)
custom_dataset.py: implements dataset of triplet in custom class, such that pytorch dataloader gets a whole triplet
preprocess_model.py: preprocess model for feature-vectors


DPENDENCIES:
-numpy
-pandas
-torch, torchvision
-PIL, random
-Matplotlib
-time

You absolutely need to run preprocess_model.py first, to create the feature vectors. We couldn't include then because the csv file was 189MB. 
For this task we tried two different approches: A siamese NN and K-means clustering with PCA. The K-means clustering with PCA showed very poor performance, thus we concentrated on the former pporach.
The final model is comprised of various modules, implemented in different python files. The algorithm used is in principle a siamese neural network with triplet loss.
In a first step, all 10'000 (pre-processed, resized, normalized) images are converted into 1000-dimensional feature vectors with a ResNext32x8d and saved. Then these are used to construct a training and validation dataset (pytorch dataset& dataloader class) together with the triplet information. 
We created a 10-layer NN consiting of linear layers with LeakyRelu activations and one batch-normalization layer. This network was trained with Triplet-margin-loss and the before gathered data. We tested noth stochastic gradient descent and adam optimizers, where the adam optimizer showed better performance. Best results were achieved with margins around 0.1. Another behaviour we spotted, was that the network started to overfit rather quickly. For this, the weights are saved in a separate file for each second training epoch, to choose for the prediction step.