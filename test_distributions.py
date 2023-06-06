import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras import models
import tensorflow as tf
import os

# script will iterate over all .hdf5 and .h5 models in 'directory' and list classification distribution for test images
f = open('test_set_distributions.txt', 'x')

filepath_dict = {
    0 : 'fire',
    1: 'frost',
    2: 'none',
    3: 'storm'
}

directory = r'**Path to Directory**'
for filename in os.listdir(directory):
    if filename.endswith(".hdf5") or filename.endswith(".h5"):
        model = load_model(filename)
        print(str(model))
        print(filename)
        f.write(filename+"\n")
        z=0
        for y in range(0, len(filepath_dict)):
            f.write(str(filepath_dict[y])+'\n')
            for x in range(1, 101):
                img_path = '**Path to Test Image Subdirectory**'+filepath_dict[y]+'\\test-'+filepath_dict[y]+'-%i.png' %x
                img = image.load_img(img_path, target_size=(480, 854)) # models must have uniform image dimensions
                img_tensor = image.img_to_array(img)
                img_tensor = np.expand_dims(img_tensor, axis=0)
                img_tensor /= 255.
                answer = model.predict(img_tensor)
                answer = answer.flatten()
                #for i in range(0, len(filepath_dict)):
                 #   f.write(str(filepath_dict[i])+': {:.0%}'.format(answer[i])+', ') #writing all classifications
                if np.argmax(answer) != y:
                    f.write(str(x)+' -- ')
                    z=z+1
                    for i in range(0, len(filepath_dict)):
                        f.write(str(filepath_dict[i])+': {:.0%}'.format(answer[i])+', ') # writing only misclassifications
                    f.write('\n')
            f.write('\n')
        f.write("Misclassified: "+str(z)+" images.\n")
        f.write("\n")
        print("Misclassified: "+str(z)+" images.")
    else:
        continue
f.close()
