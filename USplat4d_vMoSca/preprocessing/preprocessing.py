from PIL import Image
import os
import argparse
import numpy as np
from tqdm import tqdm

def downsize_image(image_path, w_h: list):
    # Open an image file
    with Image.open(image_path) as img:
        # Calculate the new size
        new_size = (int(w_h[0]), int(w_h[1]))
        # Resize the image
        downsized_img = img.resize(new_size, Image.ANTIALIAS)
        # Save the image back to the same path
        downsized_img.save(image_path)

def npy_to_npz(npy_path, npz_path=None):
    if npz_path is None:
        npz_path = npy_path.replace('.npy', '.npz')
    # Read .npy file
    data = np.load(npy_path)
    # Save the data to .npz file
    np.savez_compressed(npz_path, dep=data.squeeze())
    # Remove the original .npy file
    os.remove(npy_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir_name', type=str, default='./demo/davis_dataset')
    parser.add_argument('--w_h', type=list, default=[854, 480]) 
    parser.add_argument('--depth_dir_name', type=str, default='/home/agenuinedream/repo/MoSca/data/iphone/pillow_/sensor_depth')
    parser.add_argument('--output_dir', type=str, default='/home/agenuinedream/repo/MoSca/data/iphone/pillow_/sensor_depth')
    args = parser.parse_args()
    
    # for dir_name in os.listdir(args.root_dir_name):
    #     for file_name in os.listdir(os.path.join(args.root_dir_name, dir_name, 'images')):
    #         if file_name.endswith('.jpg'):
    #             file_path = os.path.join(args.root_dir_name, dir_name, 'images', file_name)
    #             downsize_image(file_path, args.w_h)
    
    for file_name in tqdm(os.listdir(os.path.join(args.depth_dir_name))):
        if file_name.endswith('.npy'):
            file_path = os.path.join(args.depth_dir_name, file_name)
            npz_path = os.path.join(args.output_dir, file_name.replace('.npy', '.npz'))
            npy_to_npz(file_path, npz_path)

if __name__ == '__main__':
    main()