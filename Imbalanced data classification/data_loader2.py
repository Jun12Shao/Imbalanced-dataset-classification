from torch.utils import data
from torchvision import transforms
import os
import os.path
from PIL import Image
import random
import pickle
import cv2
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler



def read_cv2_img(path):
    '''
    Read color images
    :param path: Path to image
    :return: Only returns color images
    '''
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)

    if img is not None:
        if len(img.shape) != 3:
            img = np.resize(img, (img.shape[0], img.shape[1], 3))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_loader(config, dataset_name='Knot', mode='train', num_workers=1):
    """Build and return a data loader."""
    if dataset_name=='Knot':
        dataset = KnotDataset(config, mode, dataset_name)
        if mode=='train':
            bt=config.batch_size
        elif mode=='test':
            bt=1
        else:
            bt=len(dataset)
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=bt,
                                      # shuffle=False,
                                      shuffle=(mode == 'train'),
                                      drop_last=True,
                                      num_workers=num_workers,

                                      )
    else:
        raise ValueError("Dataset [%s] not recognized." % dataset_name)


    return data_loader


class DatasetBase(data.Dataset):
    def __init__(self):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self.root = None

        # self.create_transform()

        self.IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    @property
    def name(self):
        return self._name

    @property
    def path(self):
        return self.root



    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def is_csv_file(self, filename):
        return filename.endswith('.csv')

    def get_all_files_in_subfolders(self, dir, is_file):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images


class KnotDataset(DatasetBase):
    def __init__(self, config, mode, dataset_name):
        super(KnotDataset, self).__init__()
        self.dataset_name = dataset_name
        self.config=config
        self.root =config.root
        self.mode = mode

        self.create_transform()
        self.read_dataset()

        if mode=='train' and config.oversampling:
            self.balance_dataset(config)

    def __getitem__(self, index):
        assert (index < self.dataset_size)

        # start_time = time.time()
        real_img = None
        label = None
        n=0
        while real_img is None or label is None:
            # if sample randomly: overwrite index
            if not self.config.serial_batches:
                index = random.randint(0, self.dataset_size - 1)

            # get sample data
            # sample_id = self.ids[index]
            # real_img= self.get_img_by_id(sample_id)
            # label = self.labels[sample_id]

            if self.mode=='train' and self.config.oversampling:
                real_img=self.X[index]
                label=self.y[index]
            else:
                sample_id = self.ids[index]
                real_img= self.get_img_by_id(sample_id)
                label = self.labels[sample_id]

            if real_img is None:
                print('error reading image %s, skipping sample' % sample_id)
                n+=1
            if label is None:
                print('error reading aus %s, skipping sample' % sample_id)
                n+=1
            if n>100:
                raise


        # transform data
        img = self.transform(Image.fromarray(real_img))


        return img,label

    def __len__(self):
        return self.dataset_size

    def read_dataset(self):
        # read ids
        ids_filepath = os.path.join(self.root, self.config.ids_file)

        with open(ids_filepath, 'rb') as f:
            ids = pickle.load(f, encoding='utf-8')
            f.close()

        if self.mode=='train':
            self.ids=ids[0]
        elif self.mode=='valid':
            self.ids = ids[1]
        else:
            self.ids=ids[2]


        # read labels
        labels_filepath = os.path.join(self.root, self.config.labels_file)
        with open(labels_filepath, 'rb') as f:
            self.labels = pickle.load(f, encoding='latin1')
            f.close()

        # dataset size
        self.dataset_size = len(self.ids)

        print('Finished preprocessing the {} dataset...'.format(self.name))

    def balance_dataset(self,config):
        X = np.zeros((self.dataset_size, 115, 115, 3))
        y = np.zeros(self.dataset_size)
        i = 0
        for key in self.ids:
            X[i] = self.get_img_by_id(key)

            y[i] = self.labels[key]
            i += 1
        X = X.reshape((self.dataset_size, -1))

        if config.oversampling_mode=='SMOTE':
            over = SMOTE()
        else:
            over = ADASYN()
        under = RandomUnderSampler()

        # steps = [('u', under), ('o', over)]

        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        # transform the dataset
        X, y = pipeline.fit_resample(X, y, )
        print(Counter(y))
        self.X = X.reshape((-1, 115, 115, 3)).astype(np.uint8)
        self.y = y.astype(np.int64)
        self.dataset_size = len(y)


    def create_transform(self):
        if self.mode=='train':
            transform_list = [transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        else:
            transform_list = [transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        self.transform = transforms.Compose(transform_list)


    def get_img_by_id(self, id):
        img = None
        filepath =self.root + id
        if os.path.exists(filepath):
            img=read_cv2_img(filepath)
        return img




