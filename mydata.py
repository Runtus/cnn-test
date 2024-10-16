from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np



def find_label(str):
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        # 找到标识符
        if str[i] == '%' and str[i - 1] == '.':
            last = i - 1
        # 找到找到cat或dog的起点
        if (str[i] == 'c' or str[i] == 'd') and str[i - 1] == '/':
            first = i
            break
    name = str[first: last]
    if name == 'dog':
        return 1
    else:
        return 0

def init_process(path, lens):
    data = []
    name = find_label(path)
    for i in range(lens[0], lens[1]):
        data.append([path % i, name])
    return data


def MyLoader(path):
    return Image.open(path).convert("RGB")


class MyDataset(Dataset):
    def __init__(self, data, transform, loader):
        self.data = data;
        self.transform = transform
        self.loader = loader
    
    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.data)





def load_data():
    print("data processing....")
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    path1 = "data/training_data/cats/cat.%d.jpg"
    path2 = "data/training_data/dogs/dog.%d.jpg"
    path3 = "data/testing_data/cats/cat.%d.jpg"
    path4 = "data/testing_data/dogs/dog.%d.jpg"
    data_cat_train = init_process(path1, [0, 500])
    data_dog_train = init_process(path2, [0, 500])
    data_cat_test = init_process(path3, [1000, 1200])
    data_dog_test = init_process(path4, [1000, 1200])
    data = data_cat_train + data_dog_train + data_cat_test + data_dog_test
    np.random.shuffle(data)

    train_data, val_data, test_data = data[: 900], data[900: 1100], data[1100: ]
    train_data = MyDataset(train_data, transform=transform, loader=MyLoader)
    Dtr = DataLoader(dataset=train_data, batch_size=50, shuffle=True, num_workers=0)
    val_data = MyDataset(val_data,  transform=transform, loader=MyLoader)
    Val = DataLoader(dataset=val_data, batch_size=50, shuffle=True, num_workers=0)
    test_data = MyDataset(test_data, transform=transform, loader=MyLoader)
    Dte = DataLoader(dataset=test_data, batch_size=50, shuffle=True, num_workers=0)

    return Dtr, Val, Dte
