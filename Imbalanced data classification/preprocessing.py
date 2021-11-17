# -*- coding:utf-8 -*-
import pickle
import os
import random
import glob

# train=[]
# valid=[]
# test=[]
# root='./data'
# classes={0:'/LargeKnots/', 1:'/NoDefect/',2:'/SmallKnots/'}
# meta_dict={}
# for key in classes:
#     # list_dir=os.listdir(root+classes[key])
#     list_dir=glob.glob(root+classes[key]+'*.bmp')
#     list_dir=[name[6:] for name in list_dir]
#     for name in list_dir:
#             meta_dict[name] = key
#
#     random.shuffle(list_dir)
#     K=int(len(list_dir)/10)
#     train+=list_dir[3*K:]
#     valid+=list_dir[:1*K]
#     test+=list_dir[1*K:3*K]
#
# random.shuffle(train)
# random.shuffle(valid)
# random.shuffle(test)
#
# print(len(train),len(valid),len(test))
#
# with open(root+'/labels_v2-2.pkl', 'wb') as f:
#     pickle.dump(meta_dict, f)
#     f.close()
#
#
# with open(root+'/train_valid_test_id_v2-2.pkl', 'wb') as f:
#     pickle.dump([train,valid,test], f)
#     f.close()

##################################################################### meta-learning data set preparing  ######
train=[]
test=[]
root='C:/Users/sh_jun/PycharmProjects/Face/prototypical_networks/data/miniImagenet/data'
# classes={0:'/LargeKnots/', 1:'/NoDefect/',2:'/SmallKnots/'}
# meta_dict={}
# for key in classes:
#     list_dir=os.listdir(root+classes[key])
#     # list_dir=glob.glob(root+classes[key]+'*.bmp')
#     # list_dir=[name[6:] for name in list_dir]
#     # for name in list_dir:
#     #         meta_dict[name] = key
#     list_dir=[name for name in list_dir if name[-4:]=='.bmp']
#
#     random.shuffle(list_dir)
#     K=int(len(list_dir)/5)
#     train=list_dir[K:]
#     test=list_dir[:K]
#     for name in train:
#         os.rename(root+classes[key]+name, root+classes[key][:-1]+'-train/'+name)
#     for name in test:
#         os.rename(root+classes[key]+name, root+classes[key][:-1]+'-test/'+name)

#####generate class labels
with open(root+'/val.txt','w') as f:
    for folder in ['/LargeKnots-valid/','/NoDefect-valid/','/SmallKnots-valid/']:
        for rot in ['rot000','rot090','rot180','rot270']:
            class_name=folder+rot+'\n'
            f.write(class_name)
    f.close()

with open(root+'/test.txt','w') as f:
    for folder in ['/LargeKnots-test/','/NoDefect-test/','/SmallKnots-test/']:
            class_name=folder+'rot000'+'\n'
            f.write(class_name)
    f.close()

img_dir='E:/internship/other/decathlon-1.0-data-imagenet/imagenet12/train'
folders=os.listdir(img_dir)
with open(root+'/train.txt','w') as f:
    for folder in folders[:100]:
        for rot in ['rot000', 'rot090', 'rot180', 'rot270']:
            class_name = '/'+folder +'/'+ rot + '\n'
            f.write(class_name)
    f.close()
############