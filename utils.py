import numpy as np
import os
from PIL import Image

def generate_grid( arr ):
    for i in range(len(arr)):
        arr[i] = np.hstack(arr[i])
    return np.vstack(arr)

def arr_to_2d( arr, width, projection=None):
    result = [[] for i in range(int(arr.shape[0]/width))]
    for i in range(arr.shape[0]):
        if(projection is None):
            result[int(i/width)].append(arr[i])
        else:
            result[int(i/width)].append(arr[i][projection])
    return result

def write_ref(reference, img_path, folder):
	os.mkdir(img_path + "/" + folder)
	for i in range(len(reference)):
		Image.fromarray(reference[i]*255).convert("RGB").save(img_path + '/' + folder + "/" + str(i) + ".png")
