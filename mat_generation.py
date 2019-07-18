import torch


import argparse
import os
from scipy.io import savemat
from models import SRCNN, VDSR, EDSR
from dataset import Data, ToTensor, RandomHorizontalFlip
from torchvision import transforms

import numpy as np


def main(args):
	device = torch.device(args.device)
	#we first see if there is a model with needed name in our models folder,
	model = torch.load(f"{args.model_folder}/{args.model_name}", map_location=device)
	model.eval()


	data_transforms = transforms.Compose([ToTensor()])
	
	dataset = Data(args.filename_x, args.filename_y, args.data_root,transform=data_transforms)

	output = {"Super_resolution": []}


	for sample in dataset:
		lores = sample['x'].to(device).float()
		print(lores.shape)
		sures = model(lores.unsqueeze(0)).squeeze(0)
		output["Super_resolution"].append(sures.detach().cpu().data.numpy())
	
	savemat(f"{args.model_folder}/{args.filename_out}", output)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument(
			'--data_root', type=str, default='blind_data/',
			help="Root directory of the data.")

	parser.add_argument(
			'--filename_x', '-x', type=str, default='data_20_big',
			help="Name of the low resolution data file (without the '.mat' "
			"extension).")

	parser.add_argument(
			'--filename_y', '-y', type=str, default='data_20_big',
			help="Name of the high resolution data filee (without the '.mat' "
			"extension).")

	parser.add_argument(
			'--filename_out', '-o', type=str, default='processed_blind',
			help="Name of the high resolution data filee (without the '.mat' "
			"extension).")
	
	parser.add_argument("--model_folder", type = str, default="results/result_2",
			help="Folder with the model to generate SR from")


	parser.add_argument("--model_name", type = str, default="generator.pth",
			help="The name of the model to generate the images")


	parser.add_argument('--device', type=str, default="cuda:0",
			help="Training device 'cpu' or 'cuda:0'")


	args = parser.parse_args()
	main(args)
