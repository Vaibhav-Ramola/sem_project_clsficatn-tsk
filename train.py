import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import VIT



"""
HYPERPARAMETERS
"""

LEARNING_RATE = 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 24
NUM_EPOCHS = 25
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
EMBED_SIZE = 786
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = ''
VAL_IMG_DIR = ''



def train(loader, model, optimizer, loss, scaler):
    loop = tqdm(loader)

    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # Forward propagation
        with torch.cuda.amp.autocast_mode.autocast():       # Using FP16
            predictions = model(data)
            loss = loss(predictions, targets)

        # Backward Propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss = loss.item())


def main():
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

    train_loader = 
    scaler = torch.cuda.amp.grad_scaler.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train(
            train_loader, 
            model, 
            optimizer, 
            loss, scaler
            )