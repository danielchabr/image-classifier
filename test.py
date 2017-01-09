import cv2
import os, urllib
from os import listdir
from os.path import isfile, join
import numpy as np
import mxnet as mx
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

IMAGE_SIZE = 100
MODEL = 'model/resnet'
MXNET_HOME = './..'
DATA_DIR = 'data/categories'
IMAGE_SIZE_STR = str(IMAGE_SIZE)

def get_image(url, show=True):
    filename = 'test/' +  url.split("/")[-1]
    urllib.urlretrieve(url, filename)
    img = cv2.imread(filename)
    if img is None:
        print('failed to download ' + url)
    return filename

def predict(filename, mod, classes):
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    if img is None:
        return None
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2) 
    img = img[np.newaxis, :] 
    
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    #print(prob)
    prob = np.squeeze(prob)

    a = np.argsort(prob)[::-1]    
    #for i in a[0:5]:
        #print('probability=%f, class=%s' %(prob[i], classes[i]))
    return classes[a[0:5][0]]

if __name__ == '__main__':
    sym, arg_params, aux_params = mx.model.load_checkpoint(MODEL, 1)
    mod = mx.mod.Module(symbol=sym)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,IMAGE_SIZE,IMAGE_SIZE))])
    mod.set_params(arg_params, aux_params)

    TEST_PATH = 'test/categories/'

    # TEST DATASET
    os.system('python ' + MXNET_HOME + '/tools/im2rec.py --list=1 --recursive=1 --shuffle=1  test/list test/categories')
    os.system('python ' + MXNET_HOME + '/tools/im2rec.py --num-thread=4 --resize=' + IMAGE_SIZE_STR + ' --color=1 test/list test/categories')

    total = .0
    correct = .0
    classes = [f for f in listdir(TEST_PATH) if os.path.isdir(join(TEST_PATH, f))]
    for c in classes:
        files = [f for f in listdir(TEST_PATH + c) if isfile(join(TEST_PATH + c, f))]
        for f in files:
            filepath = TEST_PATH + c + '/' + f
            result = predict(filepath, mod, classes)
            total += 1.0
            if result == c:
                #print('CORRECT ' + filepath)
                correct += 1.0
            else:
                print('FALSE ' + filepath)

    print('Validation accuracy: ' + str(correct/total))



