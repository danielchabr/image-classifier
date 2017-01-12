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
from requests.packages.urllib3.exceptions import InsecureRequestWarning

LABELS = []#['beagle dog', 'husky dog', 'dalmatian dog']
API_KEY = '4196410-648a310def2eb58655fe4fa70'
MXNET_HOME = './..'
DATA_DIR = 'data/categories'
IMAGE_SIZE = '200'
NETWORK = 'resnet'

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def downloadImages(label, links):
    for link in links:
        name = os.path.join(DATA_DIR, label, link.split('/')[-1])
        download_file(link, name)

def getImagesForLabel(label):
    URL = "https://pixabay.com/api/?key="+API_KEY+"&per_page=200&q="+label
    resp = requests.post(URL, verify=False)
    if resp.status_code != 200:
        print "Failed to get images for label " + label
        print resp.text
        sys.exit(-1)
    respJson = resp.json()
    imagesJson = respJson['hits']
    imageLinks = []
    for imageJson in imagesJson:
        imageLinks.append(imageJson['webformatURL'])
    return imageLinks

if __name__ == '__main__':

    for label in LABELS:
        imageLinks = getImagesForLabel(label)
        downloadImages(label, imageLinks)

    # TRAIN DATASET
    os.system('python ' + MXNET_HOME + '/tools/im2rec.py --list=1 --recursive=1 --shuffle=1 data/list ' + DATA_DIR)
    os.system('python ' + MXNET_HOME + '/tools/im2rec.py --num-thread=4 --resize=' + IMAGE_SIZE + ' --quality 90 --color=1 data/list ' + DATA_DIR)

    # TEST DATASET
    os.system('python ' + MXNET_HOME + '/tools/im2rec.py --list=1 --recursive=1 --shuffle=1  test/list test/categories')
    os.system('python ' + MXNET_HOME + '/tools/im2rec.py --num-thread=4 --resize=' + IMAGE_SIZE + ' --color=1 test/list test/categories')

    # parse args
    parser = argparse.ArgumentParser(description="train dogs",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 3)
    parser.set_defaults(
        #gpus           = '0',
        # network
        network        = NETWORK,
        num_layers     = 50,
        # data
        data_train     = 'data/list.rec',
        data_val       = 'test/list.rec',
        num_classes    = 3,
        num_examples   = 150,
        image_shape    = '3,' + IMAGE_SIZE + ',' + IMAGE_SIZE,
        pad_size       = 4,
        min_random_scale = 1, # if input image has min size k, suggest to use

        # train
        batch_size     = 50,
        num_epochs     = 12,
        lr             = .02,
        lr_factor      = 0.8,
        lr_factor_epoch = 0.5,
        lr_step_epochs = '4,8',
        #save model
        model_prefix   =  'model/' + NETWORK
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbol.'+args.network)
    sym = net.get_symbol(**vars(args))
    
    # train
    fit.fit(args, sym, data.get_rec_iter)
