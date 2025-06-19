import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import reduce
import wandb
import tqdm
from torch import nn

from config import Config
from preprocess.dataset import PetDataset
from preprocess.utils import *
from model.cbam import CBAMSkip
from model.basic_unet import UNet
from model.mbconv import MBConv
from model.residual_block import ResBottleneck



def run_epoch(epoch, net, loader, criterion, optimizer, scheduler, device, is_train):
    """
    Проходит одну полную эпоху

    :param epoch: номер эпохи
    :param net: модель
    :param loader: лоадер
    :param criterion: критерий
    :param optimizer: оптимизатор
    :param scheduler: планировщик
    :param device: устройство для вычислений
    :param is_train: является ли тренировкой

    :return: None
    """

    if is_train:
        net.train()
    else:
        net.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    iou = 0
    lr = config.lr

    for i, data in enumerate(tqdm.tqdm(loader)):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        loss = criterion(outputs, labels.long()).mean()

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (scheduler is not None) and (config.scheduler_type == 'cosine'):
                scheduler.step()
                lr = scheduler.get_last_lr()[0]

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1) # выбираем наиболее вероятный класс для каждого пикселя

        iou += calculate_iou(predicted.cpu().detach().numpy(), labels.cpu().detach().numpy())

        correct += (predicted == labels).sum().item() # считаем число правильных предсказаний
        total += reduce(lambda a, b: a * b, labels.shape, 1) # считаем общее число пикселей

    acc = correct / total

    print('Loss: {:.3f}, Acc: {:.3f}'.format(running_loss / (i + 1), acc * 100.0))
    print('IOU: {:.3f}'.format(iou / (i + 1)))
    torch.save(net.state_dict(), f"model_{epoch}.pt")

    if is_train:
        if (scheduler is not None) and (config.scheduler_type != 'cosine'):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]

        wandb.log({'lr': lr}, step=epoch)
        wandb.log({'train loss': running_loss / (i + 1)}, step=epoch)
        wandb.log({'train acc': acc * 100.0}, step=epoch)
        wandb.log({'train iou': iou / (i + 1)}, step=epoch)
    else:
        wandb.log({'test loss': running_loss / (i + 1)}, step=epoch)
        wandb.log({'test acc': acc * 100.0}, step=epoch)
        wandb.log({'test iou': iou / (i + 1)}, step=epoch)


if __name__ == '__main__':
    config = Config()
    fix_everything(config.seed)
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    data_root = "data"

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets and dataloaders, use albumentations for transforms
    train_transform = A.Compose([
        A.Resize(width=224, height=224),
        A.ToFloat(max_value=255),
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(),
        A.Blur(),
        A.Sharpen(),
        A.RGBShift(),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(width=224, height=224),
        A.ToFloat(max_value=255),
        ToTensorV2(),
    ])

    train_dataset = PetDataset(data_root, mask_folder='sam_output', train=True, transform=train_transform)
    val_dataset = PetDataset(data_root, mask_folder='trimaps', train=False, transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=True, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=False, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator
    )

    # Initialize model
    model = UNet(3, 3, [32, 64, 128, 256, 512],
                 conv_block=MBConv,
                 bottleneck_block=ResBottleneck,
                 skip_block=CBAMSkip)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = get_scheduler(config, optimizer, train_loader)

    wandb.init(project='segmentation_oxford_pets', name='unet_plus')

    for epoch in range(config.n_epochs):
        print("Training epoch: ", epoch + 1, "/", config.n_epochs)
        run_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, device, is_train=True)

        print('Validation')
        with torch.no_grad():
            run_epoch(epoch, model, val_loader, criterion, optimizer, scheduler, device, is_train=False)
        print('----------------------')

    print('Finished training')

    wandb.finish()

    torch.save(model.state_dict(), "unet_plus.pt")
