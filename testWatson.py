import cv2
import os, urllib
from os import listdir
from os.path import isfile, join
from collections import namedtuple
import requests
import base64
from random import randint
from time import sleep
import pickle

def predict(filename):
    url = "https://visual-recognition-demo.mybluemix.net/api/classify"
    print(filepath)

    if not os.path.isfile('test_status.pkl'):
        print('creating file')
        status = {}
        with open('test_status.pkl', 'wb') as f:
            pickle.dump(status, f)

    with open('test_status.pkl', 'rb') as f:
        status = pickle.load(f)

    if filename in status:
        #print('already analyzed')
        return status[filename]
    #else:
        #print('not analyzed yet')

    sleep(randint(1,5))

    with open(filename, "rb") as image_file:
        encoded_image = "data:" + "image/jpeg" + ";" + "base64," + base64.b64encode(image_file.read())

    payload = {
        'classifier_id': 'DogBreeds_1883911605',
        'url': '',
        'threshold': 0.0,
        'image_data': encoded_image
    }
    headers = {
        'pragma': "no-cache",
        'origin': "https://visual-recognition-demo.mybluemix.net",
        'csrf-token': "njfvfEq4-YoTIsVSLWeSFYOLZq7Ohnl_TUkY",
        'accept-encoding': "gzip, deflate, br",
        'accept-language': "en-US,en;q=0.8",
        'user-agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36",
        'content-type': "application/x-www-form-urlencoded; charset=UTF-8",
        'accept': "*/*",
        'cache-control': "no-cache",
        'x-requested-with': "XMLHttpRequest",
        'cookie': "mmapi.store.p.0=%7B%22mmparams.d%22%3A%7B%7D%2C%22mmparams.p%22%3A%7B%22mmid%22%3A%221489590613265%7C%5C%22-182437930%7CAwAAAAq5TBjHIQ0AAA%3D%3D%5C%22%22%2C%22pd%22%3A%221489590613268%7C%5C%221661070281%7CAwAAAAoBQrlMGMchDd9sJvABAL0L5OjjTNNIDwAAAJgZmLDjTNNIAAAAAP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FACZhbmFseXRpY3MtaWJtLXByb2QudzNpYm0ubXlibHVlbWl4Lm5ldAIhDQEAAAAAAAAAAAAA%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FAAAAAAABRQ%3D%3D%5C%22%22%2C%22srv%22%3A%221489590613270%7C%5C%22ldnvwcgus01%5C%22%22%7D%7D; mmapi.store.s.0=%7B%22mmparams.d%22%3A%7B%7D%2C%22mmparams.p%22%3A%7B%7D%7D; CoreID6=64538312940614580545490&ci=50200000|REPLACE_50200000|DISCOVER-IOT_50200000|IBM_GlobalMarketing; 50200000_clogin=v=37&l=1470390700&e=1470392500491; cvo_sid1=FRZMV7DDZSD4; utag_main=v_id:01537ad2cd22001c291452930aa905079003b07100e50$_sn:3$_ss:1$_st:1470392500350$dc_visit:3$_pn:1%3Bexp-session$ses_id:1470390698077%3Bexp-session$dc_event:2%3Bexp-session$dc_region:eu-central-1%3Bexp-session$ttd_uuid:57ea338c-acd2-41c8-89c6-ac3a7b3c6f45%3Bexp-session; cvo_tid1=BNhvPeXj1Tg|1470390700|1470390700|1; user=s%3A%7B%22id%22%3A%22daniel_chabr%40cz.ibm.com%22%7D.1xb6H3S5VBU6uphfijCr4l%2FGc65BGh%2BBeC7NH194mkk; _ga=GA1.2.490206474.1461064015; _csrf=VdEGlL_KjcYx9RJnu0ftI5lA; TLTSID=1i4mBjYLiz7rcag69RYQramYLMYkDFiV",
        'connection': "keep-alive",
        'referer': "https://visual-recognition-demo.mybluemix.net/train?classifier_id=DogBreeds_269347328&name=Dog%20Breeds&kind=dogs&expires=Mon%20Jan%2009%202017%2013%3A40%3A59%20GMT%2B0100%20(CET)",
        'postman-token': "6b62d9f6-e968-c80b-2212-9a652a799e57"
        }

    response = requests.request("POST", url, data=payload, headers=headers)

    respJson = response.json()
    print(respJson)

    classifiers = respJson['images'][0]['classifiers']
    if len(classifiers) == 0:
        return 0

    classifications = classifiers[0]['classes']

    if len(classifications) == 0:
        status[filename] = 'none'
        print('writing none')
        with open('test_status.pkl', 'wb') as f:
            pickle.dump(status, f)
        return 0
    else:
        breed = classifications[0]['class']
        status[filename] = breed
        print('writing ' + breed)
        with open('test_status.pkl', 'wb') as f:
            pickle.dump(status, f)
        return breed

if __name__ == '__main__':
    TEST_PATH = 'test/categories/'

    total = .0
    correct = .0
    wrongFiles = []
    classes = [f for f in listdir(TEST_PATH) if os.path.isdir(join(TEST_PATH, f))]
    for c in classes:
        files = [f for f in listdir(TEST_PATH + c) if isfile(join(TEST_PATH + c, f))]
        for f in files:
            filepath = TEST_PATH + c + '/' + f
            if filepath.split('/')[-1] == '.DS_Store':
                continue
            result = predict(filepath)
            total += 1.0
            if result == c:
                print('CORRECT ' + filepath)
                correct += 1.0
            else:
                print('FALSE ' + filepath)
                wrongFiles.append(filepath)
            # wait before sending next request

    print(wrongFiles)
    print('Validation accuracy: ' + str(correct/total))



