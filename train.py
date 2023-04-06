import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import VIT
from dataset import GSQDataset
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np



"""
HYPERPARAMETERS
"""

LEARNING_RATE = 1e-4
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
BATCH_SIZE = 50
TRAIN_SIZE = 200000
VAL_SIZE = 39999
NUM_EPOCHS = 25
NUM_HEADS = 8           # num_heads must divide embed_size completely
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
EMBED_SIZE = 784
PIN_MEMORY = True
LOAD_MODEL = False

#   ********** Add Paths Below ***********

TRAIN_IMG_DIR = './Dataset/Experiment 1/X_exp1.npy'
TRAIN_LABELS = './Dataset/Experiment 1/y_exp1.npy'
VAL_IMG_DIR = ''
VAL_LABELS = ''



def train(loader, model, optimizer, loss, scaler, epoch):
    loop = tqdm(loader)
    model.train()
    # data, targets = next(iter(loader))
    epoch_loss = 0
    for batch_index, (data, targets) in enumerate(loop):
        loop.set_description(f'Epoch : {epoch}')
        data = data['image']
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        batch_loss = 0
        
        # Forward propagation
        # with torch.cuda.amp.autocast_mode.autocast():       # Using FP16
        predictions = model(data)
        batch_loss = loss(predictions, targets)
            
        # print(f'Batch Loss : {batch_loss}')

        epoch_loss += batch_loss.item()
        # Backward Propagation
        # optimizer.zero_grad()
        # scaler.scale(batch_loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        loop.set_postfix(loss = batch_loss.item())
    avg_loss = epoch_loss / (TRAIN_SIZE//BATCH_SIZE)
    print(f'Loss for epoch per batch average : {avg_loss}')
    return avg_loss

def validation(loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader)
    loss = 0

    for data, labels in loop:
        data = data['image']
        data = data.to(device=DEVICE)
        labels = labels.to(device=DEVICE)

        predictions = model(data)
        loss = loss_fn(labels, predictions)






if __name__ == '__main__':
    train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=45, p=0.75),
            # A.HorizontalFlip(p=0.3),
            # A.VerticalFlip(p=0.4),
            # A.Normalize(
            #     mean=[]
            # )
            ToTensorV2(),
        ]
    )

    validation_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Normalize(
            #     mean=[0.0, 0.0, 0.0],
            #     std=[1.0, 1.0, 1.0],
            #     max_pixel_value=255.0,
            # ),
            ToTensorV2(),
        ]
    )

    model = VIT(
        embed_size=EMBED_SIZE, 
        in_size=IMAGE_HEIGHT, 
        num_classes=3, 
        in_channels=5,
        num_heads=NUM_HEADS
        )
    
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    
    print('Loading train set')
    dataset = GSQDataset(img_path=TRAIN_IMG_DIR, labels_path=TRAIN_LABELS, transforms=train_transforms)
    train_set, valdatn_set = torch.utils.data.random_split(dataset, [TRAIN_SIZE, VAL_SIZE])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    # scaler = torch.cuda.amp.grad_scaler.GradScaler()
    
    test_loss_list = []
    print('Training')
    
    for epoch in range(NUM_EPOCHS):
        # print(f'epoch : {epoch}/{NUM_EPOCHS}')
        epoch_loss = train(
                train_loader, 
                model, 
                optimizer, 
                loss, 
                None,
                epoch+1
                )
        test_loss_list.append(epoch_loss)

    

    print('*******  Training finished   *********')

    x = np.arange(1, NUM_EPOCHS+1)
    y = np.array(test_loss_list)

    plt.plot(x, y)
    plt.show()

    input = input('Do you want to save the model ? Y/N')
    if input=='Y':
        torch.save(model.state_dict(), './saved_model')
        print('Validation')
        val_loader = torch.utils.data.DataLoader(valdatn_set, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY)

    

    

    