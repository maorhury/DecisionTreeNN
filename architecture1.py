import io
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torchvision.datasets import Dataset
from torch.utils.data import DataLoader
import torchvision
import transforms
from local_model_store import load_model_and_optimizer_loc

class DecisionTreeVgg(torch.nn.Module):

    def __init__(self, binary_weights_path, first_race_weights_path, second_race_weights_path, n):
        obj = local_model_store() #might need some parameters
        self.binary_network = obj.load_model_and_optimizer_loc(model=torchvision.models.vgg16(num_classes=2),
                                                           model_location=binary_weights_path)
        self.first_race_network = obj.load_model_and_optimizer_loc(model=torchvision.models.vgg16(num_classes=n),
                                                               model_location=first_race_weights_path)
        self.second_race_network = obj.load_model_and_optimizer_loc(model=torchvision.models.vgg16(num_classes=n),
                                                                model_location=second_race_weights_path)
        self.cos = nn.CosineSimilarity()

    # method that gets pairs of tensors and returns  their predictions
    def forward(self, first_images, second_images):

        first_img_class_prob = self.__forward_binary(first_images)# for each photo in
        second_img_class_prob = self.__forward_binary(second_images)

        _, first_img_class_pred = torch.max(first_img_class_prob, 1)
        _, second_img_class_pred = torch.max(second_img_class_prob, 1)

        results = pd.DataFrame({'img1_class': first_img_class_pred, 'img2_class': second_img_class_pred, 'fc7_cos_sim': np.nan})
        pairs_for_first_network_indices = list(results[(results['img1_class'] == results['img2_class']) & (results['img1_class'] == 0)].index)
        pairs_for_second_network_indices = list(results[(results['img1_class'] == results['img2_class']) & (results['img1_class'] == 1)].index)

        first_images_for_first_network = torch.index_select(first_images, 0, torch.tensor(pairs_for_first_network_indices))
        second_images_for_first_network = torch.index_select(second_images, 0, torch.tensor(pairs_for_first_network_indices))

        first_images_for_second_network = torch.index_select(first_images, 0, torch.tensor(pairs_for_second_network_indices))
        second_images_for_second_network = torch.index_select(second_images, 0, torch.tensor(pairs_for_second_network_indices))

        first_fc7_first_network = self.____forward_first_race_fc7(first_images_for_first_network)
        second_fc7_first_network = self.____forward_first_race_fc7(second_images_for_first_network)
        first_network_output = self.cos(first_fc7_first_network, second_fc7_first_network)
        results.loc[pairs_for_first_network_indices, 'fc7_cos_sim'] = first_network_output

        first_fc7_second_network = self.____forward_second_race_fc7(first_images_for_second_network)
        second_fc7_second_network = self.____forward_second_race_fc7(second_images_for_second_network)
        second_network_output = self.cos(first_fc7_second_network, second_fc7_second_network)
        results.loc[pairs_for_second_network_indices, 'fc7_cos_sim'] = second_network_output

        return results

    def __forward_binary(self, images):
        return self.binary_network(images)

    def __forward_first_race_fc7(self, images):
        tmp_classifier_saver = self.first_race_network.classifier
        self.first_race_network.classifier = nn.Sequential(*[self.first_race_network.classifier[i] for i in range(4)])
        fc7_results = self.first_race_network.classifier(images)
        self.first_race_network.classifier = tmp_classifier_saver
        return fc7_results

    def __forward_second_race_fc7(self, images):
        tmp_classifier_saver = self.second_race_network.classifier
        self.first_race_network.classifier = nn.Sequential(*[self.second_race_network.classifier[i] for i in range(4)])
        fc7_results = self.second_race_network.classifier(images)
        self.second_race_network.classifier = tmp_classifier_saver
        return fc7_results


class VerificationDataset(Dataset):

    def __init__(self, dataset_path, first_images, second_images, transform):
        self.dataset_path = dataset_path
        self.first_images = first_images
        self.second_images = second_images
        self.transform = transform

    def __getitem__(self, idx):

        first_image = io.imread(os.path.join(self.dataset_path, self.first_images[idx]))
        second_image = io.imread(os.path.join(self.dataset_path, self.second_images[idx]))

        return self.transform(first_image), self.transform(second_image)


def verification_test(binary_weights_path, first_race_weights_path, dir_pairs_path, second_race_weights_path, n, pairs_file_name, dataset_path):

    first_images = []
    second_images = []
    my_network = DecisionTreeVgg(binary_weights_path, first_race_weights_path, second_race_weights_path, n)

    pairs_file = open(dir_pairs_path+pairs_file_name, 'r')
    lines = pairs_file.readlines()
    for line in lines:
        images = line.split(" ")
        first_images.append(images[0])
        second_images.append(images[1])

    transform = transforms.Compose([transforms.Resize([256, 256]), transforms.CenterCrop([224, 224]), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    verification_dataset = VerificationDataset(dataset_path, first_images, second_images, transform)
    verification_dataloader = DataLoader(verification_dataset, batch_size=4)
    for first_images, seoond_images in verification_dataloader:
        result = my_network.forward(first_images, seoond_images)

