import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import time

if __name__ == '__main__':
    # Check if CUDA (GPU support) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not found, using CPU")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), #data augmentation- focus on features, translation invariance- recognisable irrespective of position in frame, regularisation 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4) #multiple worker processes- faster loading of data

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4) #128 data samples that model looks at in one go before making a small adjustment

    # Use a pre-defined ResNet18 model
    #residual block includes a "shortcut connection" that bypasses one or more layers- guided learning; uses difference instead of direct solution
    net = torchvision.models.resnet18(pretrained=False, num_classes=10) #no pre-trained weights- start from scratch, output units in the final layer of the network (CIFAR-10 has 10 classes)
    net.to(device) 

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss() #loss function- combines softmax (converts raw output scores to probabilities) and Negative Log-Likelihood (how good probabilities are) operations
    #Stochastic Gradient Descent- optimisation alg.- update model's parameters during training.
    optimizer = optim.SGD(net.parameters(), lr=0.01, #learning rate- how much parameters updated during training (low= slow, high=overshoot)
                          momentum=0.9) #momentum- retains a fraction of the previous parameter update,which helps in pushing the optimizer in a consistent direction.


    # Initialize GradScaler-  automatically scale gradients to maintain the numerical stability of mixed-precision training.
    scaler = GradScaler()

    # Training loop
    start_time = time.time()
    for epoch in range(30): #30 iterations through dataset
        running_loss = 0.0
        total_samples = 0

        for i, data in enumerate(trainloader, 0): #iterates over the training dataset in mini-batches- data-loading parameters are determined by trainloader.
            inputs, labels = data[0].to(device), data[1].to(device) #Moves the input images and labels to the GPU (if available).

            # Zero the parameter gradients before calculating new gradients
            optimizer.zero_grad()

            #automatically casts certain PyTorch operations to half-precision- speed computation, saving memory
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            # Use GradScaler to scale the loss and backpropagate- prevent underflow during the gradient calculation- used to update the model weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() #adjusts the scaling factor for the next iteration

            #track loss
            running_loss += loss.item()
            total_samples += labels.size(0)

#performance assessment
        average_loss = running_loss / total_samples
        print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")

    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")

    # Evaluate the trained model on the test dataset
    net.eval() #Sets the model to evaluation mode- some layers (i.e. dropout, batch normalization) behave differently during training and evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")