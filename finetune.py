import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx
import requests
import json
import codecs
import sys, time
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

MXNET_HOME = './..'
DATA_DIR = 'data/categories'
IMAGE_SIZE = 200
IMAGE_SIZE_STR = str(IMAGE_SIZE)
NETWORK = 'resnet'

def get_iterators(batch_size, data_shape=(3, IMAGE_SIZE, IMAGE_SIZE)):
    train = mx.io.ImageRecordIter(
        #path_imgrec         = './data/list_train.rec',
        path_imgrec         = './data/list.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        shuffle             = True,
        rand_crop           = True,
        rand_mirror         = True)
    val = mx.io.ImageRecordIter(
        #path_imgrec         = './data/list_val.rec',
        path_imgrec         = './test/list.rec',
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        rand_crop           = False,
        rand_mirror         = False)
    return (train, val)

def get_model(prefix, epoch):
    fname1 = prefix+'-symbol.json'
    fname2 = prefix+'-%04d.params' % (epoch,)
    if not os.path.isfile(fname1.split('/')[-1]):
        download_file(prefix+'-symbol.json')
    if not os.path.isfile(fname2.split('/')[-1]):
        download_file(prefix+'-%04d.params' % (epoch,))

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)

def _save_model():
    model_prefix = 'model/resnet'
    dst_dir = os.path.dirname(model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(model_prefix)

def _load_model():
    model_prefix = 'model/resnet'
    epoch = 10
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, epoch)
    return (sym, arg_params, aux_params)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = mx.cpu() if num_gpus is None or args.gpus is 0 else [
        mx.gpu(i) for i in range(num_gpus)]
    #devs = [mx.gpu(i) for i in range(num_gpus)]

    model_prefix   =  'model/' + NETWORK
    # save model
    checkpoint = _save_model()

    new_sym, arg_params, aux_params = _load_model()

    mod = mx.mod.Module(symbol=new_sym, context=devs)
    mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
    mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
    mod.set_params(arg_params, aux_params, allow_missing=True)
    mod.fit(train, None, 
        num_epoch=10,
        batch_end_callback = mx.callback.Speedometer(batch_size, 1),        
        epoch_end_callback = checkpoint,
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        eval_metric='acc')
    return mod.score(val, 'acc')

if __name__ == '__main__':

    os.system('python ' + MXNET_HOME + '/tools/im2rec.py --list=1 --recursive=1 --shuffle=1 data/list ' + DATA_DIR)
    os.system('python ' + MXNET_HOME + '/tools/im2rec.py --num-thread=16 --resize=' + IMAGE_SIZE_STR + ' data/list ' + DATA_DIR)

    os.system('python ' + MXNET_HOME + '/tools/im2rec.py --list=1 --recursive=1 --shuffle=1  test/list test/categories')
    os.system('python ' + MXNET_HOME + '/tools/im2rec.py --num-thread=16 --resize=' + IMAGE_SIZE_STR + ' --color=1 test/list test/categories')

    get_model('http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50', 0)
    get_model('http://data.mxnet.io/models/imagenet-11k/resnet-152/resnet-152', 0)
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)

    # Output may vary
    num_classes = 3
    #batch_per_gpu = 16
    #num_gpus = 8

    (new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)

    #Additional for local only
    #batch_size = batch_per_gpu * num_gpus
    num_gpus = None
    batch_size = 30

    (train, val) = get_iterators(batch_size)
    mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)
    assert mod_score > 0.77, "Low training accuracy."
