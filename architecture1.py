import os
from skimage import io
import pandas as pd
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import sys
sys.path.insert(0, "/home/OM/projects/facial_feature_impact_comparison/modelling")
from local_model_store import LocalModelStore

################################## This is the most updated version. Done in 31/07. should be documented and create something generic ##############################

class DecisionTreeVgg(torch.nn.Module):

    def __init__(self, binary_weights_path, first_race_weights_path, second_race_weights_path, n):

        obj = LocalModelStore("", "", "")

        vgg16_binary = torchvision.models.vgg16(num_classes=2).train(False)
        vgg16_binary.cuda()
        vgg16_binary.features = torch.nn.DataParallel(vgg16_binary.features)
        self.binary_network = obj.load_model_and_optimizer_loc(model=vgg16_binary, model_location=binary_weights_path)
        
        vgg16_first = torchvision.models.vgg16(num_classes=n).train(False)
        vgg16_binary.cuda()
        vgg16_first.features = torch.nn.DataParallel(vgg16_first.features)
        self.first_race_network = obj.load_model_and_optimizer_loc(model=vgg16_first, model_location=first_race_weights_path)

        vgg16_second = torchvision.models.vgg16(num_classes=n).train(False)
        vgg16_binary.cuda()
        vgg16_second.features = torch.nn.DataParallel(vgg16_second.features)
        self.second_race_network = obj.load_model_and_optimizer_loc(model=vgg16_second, model_location=second_race_weights_path)

    # method that gets pairs of tensors and returns  their predictions
    def forward(self, first_images, second_images):

        first_img_class_prob = self.__forward_binary(first_images)# for each photo in
        second_img_class_prob = self.__forward_binary(second_images)

        _, first_img_class_pred = torch.max(first_img_class_prob, 1)
        _, second_img_class_pred = torch.max(second_img_class_prob, 1)

        results = pd.DataFrame({'img1_class': first_img_class_pred.cpu(), 'img2_class': second_img_class_pred.cpu(), 'fc7_cos_sim': np.nan})
        pairs_for_first_network_indices = list(results[(results['img1_class'] == results['img2_class']) & (results['img1_class'] == 0)].index)
        pairs_for_second_network_indices = list(results[(results['img1_class'] == results['img2_class']) & (results['img1_class'] == 1)].index)

        cos = nn.CosineSimilarity()

        if pairs_for_first_network_indices:
            first_images_for_first_network = torch.index_select(first_images, 0, torch.tensor(pairs_for_first_network_indices))
            second_images_for_first_network = torch.index_select(second_images, 0, torch.tensor(pairs_for_first_network_indices))
            first_fc7_first_network = self.__forward_first_race_fc7(first_images_for_first_network)
            second_fc7_first_network = self.__forward_first_race_fc7(second_images_for_first_network)
            first_network_output = cos(first_fc7_first_network, second_fc7_first_network)
            results.loc[pairs_for_first_network_indices, 'fc7_cos_sim'] = first_network_output.cpu().detach().numpy()

        if pairs_for_second_network_indices:
            first_images_for_second_network = torch.index_select(first_images, 0, torch.tensor(pairs_for_second_network_indices))
            second_images_for_second_network = torch.index_select(second_images, 0, torch.tensor(pairs_for_second_network_indices))
            first_fc7_second_network = self.__forward_second_race_fc7(first_images_for_second_network)
            second_fc7_second_network = self.__forward_second_race_fc7(second_images_for_second_network)
            second_network_output = cos(first_fc7_second_network, second_fc7_second_network)
            results.loc[pairs_for_second_network_indices, 'fc7_cos_sim'] = second_network_output.cpu().detach().numpy()
            
        return results

    def __forward_binary(self, images):
        return self.binary_network[0](images)

    def __forward_first_race_fc7(self, images):
        tmp_classifier_saver = self.first_race_network[0].classifier
        self.first_race_network[0].classifier = nn.Sequential(*[self.first_race_network[0].classifier[i] for i in range(5)])
        self.first_race_network[0].cuda()
        fc7_results = self.first_race_network[0](images)
        self.first_race_network[0].classifier = tmp_classifier_saver
        return fc7_results

    def __forward_second_race_fc7(self, images):
        tmp_classifier_saver = self.second_race_network[0].classifier
        self.first_race_network[0].classifier = nn.Sequential(*[self.second_race_network[0].classifier[i] for i in range(5)])
        self.second_race_network[0].cuda()
        fc7_results = self.second_race_network[0](images)
        self.second_race_network[0].classifier = tmp_classifier_saver
        return fc7_results


class VerificationDataset(Dataset):

    def __init__(self, dataset_path, first_images, second_images, transform):
        self.dataset_path = dataset_path
        self.first_images = first_images
        self.second_images = second_images
        self.transform = transform
        self.num_pairs = len(first_images)

    def __getitem__(self, idx):

        first_image = io.imread(os.path.join(self.dataset_path, self.first_images[idx]))
        second_image = io.imread(os.path.join(self.dataset_path, self.second_images[idx]))

        jpg_to_pil = transforms.ToPILImage()

        return self.transform(jpg_to_pil(first_image)), self.transform(jpg_to_pil(second_image))

    def __len__(self):
        return self.num_pairs



def verification_test(binary_weights_path, first_race_weights_path, second_race_weights_path, n, dir_pairs_path, pairs_file_name, dataset_path, test_type):

    first_images = []
    second_images = []
    my_network = DecisionTreeVgg(binary_weights_path, first_race_weights_path, second_race_weights_path, n)

    pairs_file = open(dir_pairs_path+pairs_file_name, 'r')
    lines = pairs_file.readlines()
    for line in lines:
        images = line.split(" ")
        first_images.append(images[0])
        second_images.append(images[1].rstrip())

    df = pd.DataFrame(columns = ['img1_class', 'img2_class', 'fc7_cos_sim'])

    transform = transforms.Compose([transforms.Resize([256, 256]), transforms.CenterCrop([224, 224]), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    verification_dataset = VerificationDataset(dataset_path, first_images, second_images, transform)
    verification_dataloader = DataLoader(verification_dataset, batch_size=4)
    
    for first_images, second_images in verification_dataloader:
        first_images.cuda()
        second_images.cuda()
        scores = my_network.forward(first_images, second_images)
        df = pd.concat([df, scores])
 
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    df['type'] = test_type
    df.to_csv(f"/home/ssd_storage/experiments/students/OM/first_arch_exp/{test_type}.csv")
    print("Done")



binary_weights_path = "/home/ssd_storage/experiments/students/OM/binary_expirement/vgg16/models/109.pth"
first_race_weights_path = "/home/ssd_storage/experiments/students/OM/2_expirement/vgg16/models/109.pth"
second_race_weights_path = "/home/ssd_storage/experiments/students/OM/3_expirement/vgg16/models/109.pth"
dir_pairs_path = "/home/ssd_storage/datasets/students/OM/"
n = 477
pairs_file_name = "same_2.txt"
dataset_path = "/home/administrator/datasets/vggface2_mtcnn/"
test_type = "same_first_eth"
verification_test(binary_weights_path, first_race_weights_path, second_race_weights_path, n, dir_pairs_path, pairs_file_name, dataset_path, test_type)


pairs_file_name = "diff_2.txt"
test_type = "diff_first_eth"
verification_test(binary_weights_path, first_race_weights_path, second_race_weights_path, n, dir_pairs_path, pairs_file_name, dataset_path, test_type)

pairs_file_name = "same_3.txt"
test_type = "same_second_eth"
verification_test(binary_weights_path, first_race_weights_path, second_race_weights_path, n, dir_pairs_path, pairs_file_name, dataset_path, test_type)


pairs_file_name = "diff_3.txt"
test_type = "diff_second_eth"
verification_test(binary_weights_path, first_race_weights_path, second_race_weights_path, n, dir_pairs_path, pairs_file_name, dataset_path, test_type)

pairs_file_name = "diff_mixed.txt"
test_type = "diff_eth_eth"
verification_test(binary_weights_path, first_race_weights_path, second_race_weights_path, n, dir_pairs_path, pairs_file_name, dataset_path, test_type)
