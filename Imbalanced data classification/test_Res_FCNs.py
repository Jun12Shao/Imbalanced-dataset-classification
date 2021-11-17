# -*- coding:utf-8 -*-

from data_loader2 import get_loader
import argparse
from torch.backends import cudnn
import numpy as np
from models.resnet import resnet18
import onnxruntime
import time

def compute_precision_recall_and_F1(y, prec):
    results=np.zeros((3,3))
    N=len(y)
    a= float(sum((y==0)&(prec==0)))
    b= float(sum((y==1)&(prec==0)))
    c= float(sum((y==2)&(prec==0)))
    d= float(sum((y==0)&(prec==1)))
    e= float(sum((y==1)&(prec==1)))
    f= float(sum((y==2)&(prec==1)))
    g= float(sum((y==0)&(prec==2)))
    h= float(sum((y==1)&(prec==2)))
    l= float(sum((y==2)&(prec==2)))

    ### precision, recall and F1 for LargeKnots class
    if a+b+c>0 and a+d+g>0:
        p=a/(a+b+c)
        r = a / (a + d + g)

        results[0, 0]=p
        results[0, 1]=r
        results[0,2]=2*p*r/(p+r)

    ### precision, recall and F1 for Nodefect class
    if d + e + f > 0 and b + e + h > 0:
        p = e / (d + e + f)
        r = e / (b + e + h)
        results[1, 0] = p
        results[1, 1] = r
        results[1, 2] = 2 * p * r / (p + r)

    ### precision, recall and F1 for SmallKnots class
    if g + h + l > 0 and c + f + l > 0:
        p = l / (g + h + l)
        r = l / (c + f + l)
        results[2, 0] = p
        results[2, 1] = r
        results[2, 2] = 2 * p * r / (p + r)

    return results



def main():
    parser = argparse.ArgumentParser(description='Train FCN networks')

    # train args
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--best_model', type=str, default='./checkpoints/res-ADASYN-3.onnx', help='the relative path of saved best model')
    parser.add_argument('--pretrained', default=True,  help='if true, load the pretrained best model')
    parser.add_argument('--serial_batches', default=True, help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--oversampling', default=False,  help='if true, take oversampling and undersampling for imbalanced dataset')
    # Directories.
    parser.add_argument('--root', type=str, default='./data', help='path to dataset')
    parser.add_argument('--ids_file', type=str, default='train_valid_test_id_v2-2.pkl', help='file containing samples id')
    parser.add_argument('--labels_file', type=str, default='labels_v2-2.pkl', help='file containing labes dictionary')

    config = parser.parse_args()

    # For fast training.
    cudnn.benchmark = True
    time1 = time.time()

    # Data loader.
    loader_test = get_loader(config, 'Knot', 'test')

    res=resnet18(pretrained=True)
    res = res.cuda()
    res.eval()

    ort_session = onnxruntime.InferenceSession(config.best_model)

    ## Test loss and accuracy
    test_accuracy = 0
    total_y = None
    total_pred = None

    for step, (X, y) in enumerate(loader_test):
        X=X.cuda()
        outputs = res(X)
        outputs = ort_session.run(None, {'actual_input_1': outputs.detach().cpu().numpy()})

        y=y.numpy()
        predictions = np.argmax(outputs[0],axis=1)
        acc = np.mean(y == predictions)
        test_accuracy += acc

        if total_y is None:
            total_y = y
        else:
            total_y = np.concatenate([total_y, y], axis=0)

        if total_pred is None:
            total_pred = predictions
        else:
            total_pred = np.concatenate([total_pred, predictions], axis=0)

    ## calculate precision, recall and f1 value
    test_p_r_f = compute_precision_recall_and_F1(total_y, total_pred)

    print('Test_accuracy', test_accuracy/(step+1))
    print('Test_f1:', test_p_r_f[0, 2] , test_p_r_f[1, 2] , test_p_r_f[2, 2])

    v_f1=test_p_r_f[:,2].mean()
    print('Average f1:' , v_f1)
    time2 = time.time()
    print(time2 - time1)

if __name__ == '__main__':
    main()

