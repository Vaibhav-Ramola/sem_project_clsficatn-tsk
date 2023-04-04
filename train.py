import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import VIT
from dataset import GSQDataset
import torch.utils.data




"""
HYPERPARAMETERS
"""

LEARNING_RATE = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
NUM_EPOCHS = 5
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
EMBED_SIZE = 786
PIN_MEMORY = True
LOAD_MODEL = False

#   ********** Add Paths Below ***********

TRAIN_IMG_DIR = './Dataset/Experiment 1/X_exp1.npy'
TRAIN_LABELS = './Dataset/Experiment 1/y_exp1.npy'
VAL_IMG_DIR = ''
VAL_LABELS = ''



def train(loader, model, optimizer, loss, scaler):
    loop = tqdm(loader)

    data, targets = next(iter(loader))

    # for batch_index, (data, targets) in enumerate(loop):
    #     data = data.to(device=DEVICE)
    #     targets = targets.float().unsqueeze(1).to(device=DEVICE)

    # Forward propagation
    with torch.cuda.amp.autocast_mode.autocast():       # Using FP16
        predictions = model(data)
        loss = loss(predictions, targets)

    print(f'Loss : {loss}')

    # Backward Propagation
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    loop.set_postfix(loss = loss.item())


if __name__ == '__main__':
    train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=45, p=0.75),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.4),
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
        in_channels=5
        )
    
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    print('Loading train set')
    dataset = GSQDataset(img_path=TRAIN_IMG_DIR, labels_path=TRAIN_LABELS, transforms=train_transforms)
    train_set, test_set = torch.utils.data.random_split(dataset, [20, 3])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY)
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    print('Starting the training')
    for epoch in range(NUM_EPOCHS):
        print(f'epoch : {epoch}/{NUM_EPOCHS}')
        train(
            train_loader, 
            model, 
            optimizer, 
            loss, scaler
            )