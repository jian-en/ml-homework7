
import time
import copy
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from skimage import io# , transform
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


NAME = {0: "St. Stephan's Cathedral, Austria",
        1: "Teide, Spain",
        2: "Tallinn, Estonia",
        3: "Brugge, Belgium",
        4: "Montreal, Canada",
        5: "Itsukushima Shrine, Japan",
        6: "Shanghai, China",
        7: "Brisbane, Australia",
        8: "Edinburgh, Scotland",
        9: "Stockholm, Sweden"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LandmarksDataset(Dataset):
    """Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                '.'.join([self.landmarks.iloc[idx, 1], 'jpg']))
        image = io.imread(img_name)
        landmark_id = self.landmarks.iloc[idx, 3].astype('int')
        sample = {'image': image,
                  'landmark_id': landmark_id,
                  'landmark_name': NAME[landmark_id]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmark_id, landmark_name = sample['image'],\
                                            sample['landmark_id'],\
                                            sample['landmark_name']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        return {'image': image, 'landmark_id': landmark_id, 'landmark_name': landmark_name}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmark_id, landmark_name = sample['image'], sample['landmark_id'], sample['landmark_name']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmark_id': landmark_id,
                'landmark_name': landmark_name}


landmarks = LandmarksDataset(csv_file="train.csv",
                             root_dir="images/",
                             transform=transforms.Compose([RandomCrop(224),
                                                           ToTensor()]))

dataloader = DataLoader(landmarks, batch_size=4,
                        shuffle=True, num_workers=4)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for data in dataloader:
            inputs, labels = data['image'].type(torch.FloatTensor), data['landmark_id'].type(torch.FloatTensor)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(landmarks)
            epoch_acc = running_corrects.double() / len(landmarks)

            print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs = data['image']
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(NAME[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
