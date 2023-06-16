from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import csv

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    """
    WARNING: Do not perform data normalization here.
        Your code here
        Hint: Use your solution (or the master solution) to HW1
        """

    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        WARNING: Do not perform data normalization here.
        """
        self.csv_data = []
        self.data_path = dataset_path
        level_file_path = dataset_path + '/labels.csv'
        with open(level_file_path, newline='') as csv_file:
            reader_d = csv.DictReader(csv_file)
            for row in reader_d:
                self.csv_data.append(row)
        # print(self.csv_data)
        # raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        """
        Your code here
        """
        return len(self.csv_data)
        # raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        data = self.csv_data[idx]
        image_path = self.data_path + '/' + data['file']
        label_index = LABEL_NAMES.index(data['label'])
        # Read the image
        image = Image.open(image_path)
        # Define a transform to convert the image to tensor
        transform = transforms.ToTensor()
        # Convert the image to PyTorch tensor
        img = transform(image)
        return [img, label_index]
        # raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
