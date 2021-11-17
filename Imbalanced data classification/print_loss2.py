# -*- coding:utf-8 -*-

"""
@author: Jun
@file: .py
@time: 5/30/20194:11 PM
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
root='C:/Users/sh_jun/PycharmProjects/Face/internship/results/'

# logs=['train_logs_cnn-no-1.pkl','train_logs_cnn-cs-1.pkl','train_logs_cnn-os-SMOTE-1.pkl','train_logs_cnn-os-ADASYN-1.pkl','train_logs_res-ADASYN-1.pkl']
#
# train_loss=[]
# train_acc=[]
# train_f1=[]
#
# valid_loss=[]
# valid_acc=[]
# valid_f1=[]
#
# for log in logs:
#     with open(root+log, 'rb') as f:
#         errors= pickle.load(f, encoding='latin1')
#         f.close()
#
#
#
#     train_loss.append(errors['train_loss'])
#     train_acc.append(errors['train_accuracy'])
#
#     train_c1_f1=errors['train_c1_f1']
#     train_c2_f1 = errors['train_c2_f1']
#     train_c3_f1 = errors['train_c3_f1']
#
#     valid_loss.append(errors['valid_loss'])
#     valid_acc.append(errors['valid_accuracy'])
#
#
#
#     valid_c1_f1 = errors['valid_c1_f1']
#     valid_c2_f1 = errors['valid_c2_f1']
#     valid_c3_f1 = errors['valid_c3_f1']
#
#
#     train_f1.append((np.array(train_c1_f1)+np.array(train_c2_f1)+np.array(train_c3_f1))/3)
#     valid_f1.append((np.array(valid_c1_f1)+np.array(valid_c2_f1)+np.array(valid_c3_f1))/3)
#
#
# n = len(train_c1_f1)
# x = range(1, n + 1)
# # Plot the result of experiment 4.
# fig, ax = plt.subplots(3, 1)
#
# ax[0].plot(x, train_loss[0], 'r-',)
# ax[0].plot(x, valid_loss[0], 'r^-')
#
# ax[0].plot(x, train_loss[1], 'b-',)
# ax[0].plot(x, valid_loss[1], 'b^-')
#
# ax[0].plot(x, train_loss[4], 'g-')
# ax[0].plot(x, valid_loss[4], 'g^-')
#
# # ax[0].plot(x, train_loss[2], 'b-')
# # ax[0].plot(x, valid_loss[2], 'b^-')
# #
# # ax[0].plot(x, train_loss[3], 'c-')
# # ax[0].plot(x, valid_loss[3], 'c^-')
#
#
#
#
# ax[1].plot(x, train_f1[0], 'r-')
# ax[1].plot(x, valid_f1[0], 'r^-')
#
# ax[1].plot(x, train_f1[1], 'b-')
# ax[1].plot(x, valid_f1[1], 'b^-')
#
# ax[1].plot(x, train_f1[4], 'g-')
# ax[1].plot(x, valid_f1[4], 'g^-')
#
#
# # ax[1].plot(x, train_f1[2], 'b-')
# # ax[1].plot(x, valid_f1[2], 'b^-')
# #
# # ax[1].plot(x, train_f1[3], 'c-')
# # ax[1].plot(x, valid_f1[3], 'c^-')
#
#
#
# ax[2].plot(x, train_acc[0], 'r-',label='bc_train')
# ax[2].plot(x, valid_acc[0], 'r^-',label='bc_val')
#
# ax[2].plot(x, train_acc[1], 'b-',label='cs_train')
# ax[2].plot(x, valid_acc[1], 'b^-',label='cs_val')
#
# ax[2].plot(x, train_acc[4], 'g-',label='Res_train')
# ax[2].plot(x, valid_acc[4], 'g^-',label='Res_val')
#
# # ax[2].plot(x, train_acc[2], 'b-',label='SMOTE_train')
# # ax[2].plot(x, valid_acc[2], 'b^-',label='SMOTE_va')
# #
# # ax[2].plot(x, train_acc[3], 'c-',label='ADA_train')
# # ax[2].plot(x, valid_acc[3], 'c^-',label='ADA_val')
#
#
# # for m in range(3):
# leg = ax[2].legend(loc="lower center", ncol=6, shadow=True)
# leg.get_title().set_color("red")
#
#
# ax[0].set_ylim(0.5, 1.0)
# ax[1].set_ylim(0.4, 1)
# ax[2].set_ylim(0.5, 1.0)
#
# ax[0].set_ylabel('Loss')
# ax[1].set_ylabel('F1 Value')
# ax[2].set_ylabel('Accuracy')
# ax[2].set_xlabel('Training Epochs')
# # plt.legend()
# plt.show()


############Print out result of meta-learning ##################################################
import ast

root='C:/Users/sh_jun/PycharmProjects/Face/prototypical_networks/scripts/train/few_shot/results/'
train_loss=[]
train_acc=[]
val_loss=[]
val_acc=[]

with open(root+'trace.txt', 'r') as f:
    lines=f.readlines()

    # contents = f.read()
    for line in lines:
        dictionary = ast.literal_eval(line)
        train_loss.append(dictionary['train']['loss'])
        train_acc.append(dictionary['train']['acc'])
        val_loss.append(dictionary['val']['loss'])
        val_acc.append(dictionary['val']['acc'])
    f.close()

n=len(train_loss)
x=range(1,n+1)


# Plot the result of experiment 4.
fig, ax = plt.subplots(2, 1)

ax[0].plot(x, train_loss, 'ro-')
ax[0].plot(x, val_loss, 'b^-')

ax[1].plot(x, train_acc, 'ro-',label='train')
ax[1].plot(x, val_acc, 'b^-',label='val')

leg = ax[1].legend(loc="lower center", ncol=2, shadow=True)
leg.get_title().set_color("red")

ax[0].set_ylim(0, 4)
ax[1].set_ylim(0, 1)

ax[0].set_ylabel('Loss')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Training Epochs')
# plt.legend()
plt.show()