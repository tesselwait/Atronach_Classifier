from PIL import Image
import os

dataset_directory = "PATH"
directories = [os.path.abspath(x[0]) for x in os.walk(dataset_directory)]
for i in directories:
    os.chdir(i)
    for item in os.listdir(i):
        if os.path.isfile(item):
            img = Image.open(item)
            f, e = os.path.splitext(item)
            img.resize((360,360), Image.ANTIALIAS).save(f+'.png')
    print(str(i)+" resized")
print("Job Complete")
