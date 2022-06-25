from PIL import Image
import random
import os
import hashlib
base_dir = 'PATH' #dataset path

duplicate_dir = 'DUPLICATE' #path to move duplicate images
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_fire_dir = os.path.join(train_dir, 'fire')
train_frost_dir = os.path.join(train_dir, 'frost')
train_storm_dir = os.path.join(train_dir, 'storm')
train_none_dir = os.path.join(train_dir, 'none')
train_guard_dir = os.path.join(train_dir, 'guard')
train_bear_dir = os.path.join(train_dir, 'bear')
validation_fire_dir = os.path.join(validation_dir, 'fire')
validation_frost_dir = os.path.join(validation_dir, 'frost')
validation_storm_dir = os.path.join(validation_dir, 'storm')
validation_none_dir = os.path.join(validation_dir, 'none')
validation_guard_dir = os.path.join(validation_dir, 'guard')
validation_bear_dir = os.path.join(validation_dir, 'bear')
test_fire_dir = os.path.join(test_dir, 'fire')
test_frost_dir = os.path.join(test_dir, 'frost')
test_storm_dir = os.path.join(test_dir, 'storm')
test_none_dir = os.path.join(test_dir, 'none')
test_guard_dir = os.path.join(test_dir, 'guard')
test_bear_dir = os.path.join(test_dir, 'bear')

dir_list = []
dir_list.append(train_fire_dir)
dir_list.append(train_frost_dir)
dir_list.append(train_none_dir)
dir_list.append(train_guard_dir)
dir_list.append(train_bear_dir)
dir_list.append(train_storm_dir)
dir_list.append(validation_fire_dir)
dir_list.append(validation_frost_dir)
dir_list.append(validation_none_dir)
dir_list.append(validation_bear_dir)
dir_list.append(validation_guard_dir)
dir_list.append(validation_storm_dir)
dir_list.append(test_fire_dir)
dir_list.append(test_frost_dir)
dir_list.append(test_none_dir)
dir_list.append(test_guard_dir)
dir_list.append(test_bear_dir)
dir_list.append(test_storm_dir)

hash_map = []
collisions = []
for x in dir_list:
    for filename in os.listdir(x):
        f = os.path.join(x, filename)
        if os.path.isfile(f):
            hash_map.append([hashlib.md5(Image.open(f).tobytes()).hexdigest(), f])
m=0
for y in hash_map:
    for z in hash_map:
        if y[0] == z[0] and y[1] != z[1]:
            collisions.append([y[1], z[1]])
            z[0] = random.random()
            #os.rename(z[1], os.path.join(duplicate_dir, 'image_'+str(m)+'.png'))
            m+=1

for i in collisions:
    print('collision at: '+ i[0]+', '+ i[1])
