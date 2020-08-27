import boto3
import json
import cv2
import os
import time
import sys

import logging
import zipfile
try:
    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

import numpy as np

from io import BytesIO


# EVALUATION SYSTEM SETTINGS
# YOU CAN ONLY CHANGE LINE 95 - 116

WORKSPACE_BUCKET = 'celeba-spoof-eval-workspace'
IMAGE_LIST_PATH = 'files/challenge_test_path_crop_20796.txt'
IMAGE_PREFIX = 'test_data/'
UPLOAD_PREFIX = 'test_output/'
TMP_PATH = '/tmp'
# 
LOCAL_IMAGE_PREFIX = 'test_data/'
LOCAL_ROOT = '/'
LOCAL_IMAGE_LIST_PATH = 'test_data/test_example.txt'
LOCAL_LABEL_LIST_PATH = 'test_data/test_example_label.json'



def _get_s3_image_list(s3_bucket, s3_path):
    s3_client = boto3.client('s3', region_name='us-west-2')
    f = BytesIO()
    # logging.info("s3_bucket and  s3_path {}{}".format(s3_bucket, s3_path))
    s3_client.download_fileobj(s3_bucket, s3_path, f)
    lines = f.getvalue().decode('utf-8').split('\n')
    return [x.strip() for x in lines if x != '']


def _download_s3_image(s3_bucket, s3_path, filename):
    s3_client = boto3.client('s3', region_name='us-west-2')
    image_name = filename.split('/')[-1]
    local_path = os.path.join(TMP_PATH, image_name)
    #download required image from s3 to local
    s3_client.download_file(s3_bucket, s3_path, local_path)

def _upload_output_to_s3(data, filename, s3_bucket, s3_prefix):
    s3_client = boto3.client('s3', region_name='us-west-2')
    local_path = os.path.join(TMP_PATH, filename)
    s3_path = os.path.join(s3_prefix, filename)
    
    # Put data into binary file
    data_str = json.dumps(data)
    encode_data = data_str.encode()
    with open(local_path, 'wb') as f:
        f.write(encode_data)
    s3_client.upload_file(local_path, s3_bucket, s3_path)


def get_job_name():
    return os.environ['CELEBASPOOF_EVAL_JOB_NAME']


def upload_eval_output(output_probs, job_name):
    """
    This function uploads the testing output to S3 to trigger evaluation.
    
    params:
    - output_probs (dict): dict of probability of every image
    - output_times (dict): dict of processing time of every image
    - job_name (str)

    """
    upload_data = {
        i: {
            "prob": output_probs[i],
        } for i in output_probs
    }

    filename = '{}.bin'.format(job_name)

    _upload_output_to_s3(upload_data, filename, WORKSPACE_BUCKET, UPLOAD_PREFIX)

    logging.info("output uploaded to {}{}".format(UPLOAD_PREFIX, filename))

def read_image(image_path):
    """
    Read an image from input path

    params:
        - image_local_path (str): the path of image.
    return:
        - image: Required image.
    """
    ########################################################################################################
    # print(image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    # Your change ends here.
    ########################################################################################################

def get_image():
    """
    This function returns a iterator of test images.
    Each iteration provides a tuple of (video_id, image), image will be in RGB color format with array shape of (height, width, 3).
    
    return: tuple(video_id: str, frames: numpy.array)
    """
    image_list = _get_s3_image_list(WORKSPACE_BUCKET, IMAGE_LIST_PATH)
    logging.info("got image list, {} image".format(len(image_list)))
    Batch_size = 2048
    logging.info("Batch_size=, {}".format(Batch_size))
    n = 0
    final_image = []
    final_image_buff = []
    final_image_id = []
    final_image_id_buff = []

    for idx,image_id in enumerate(image_list):
        # get video from s3
        st = time.time()
        try:
            #download required image from s3 to local
            _download_s3_image(WORKSPACE_BUCKET, os.path.join(IMAGE_PREFIX, image_id), image_id)
        except:
            logging.info("Failed to download image: {}".format(os.path.join(IMAGE_PREFIX, image_id)))
            raise
        image_name = image_id.split('/')[-1]
        image_local_path = os.path.join(TMP_PATH, image_name) # local path of the video named video_id
        image = read_image(image_local_path)
        final_image_buff.append(image)
        final_image_id_buff.append(image_id)
        if len(final_image_buff) == 500:
            final_image.extend(final_image_buff)
            final_image_buff = []
            final_image_id.extend(final_image_id_buff)
            final_image_id_buff = []

        n += 1
        elapsed = time.time() - st
        logging.info("image downloading & image reading time: {}".format(elapsed))

        if n == Batch_size or idx == len(image_list) - 1:
            final_image.extend(final_image_buff)
            final_image_id.extend(final_image_id_buff)
            np_final_image_id = np.array(final_image_id)
            np_final_image = np.array(final_image)
            n = 0
            yield np_final_image_id, np_final_image

            try:
                for i in final_image_id:
                    os.remove(i) # remove the local image 
            except:
                logging.info("Failed to delete this image, error: {}".format(sys.exc_info()[0]))

            final_image = []
            final_image_buff = []
            final_image_id = []
            final_image_id_buff = []

def get_local_image(max_number=None):
    """
    This function returns a iterator of image.
    It is used for local test of participating algorithms.
    Each iteration provides a tuple of (image_id, image), each image will be in RGB color format with array shape of (height, width, 3)
    
    return: tuple(image_id: str, image: numpy.array)
    """
    image_list = [x.strip() for x in open(LOCAL_IMAGE_LIST_PATH)]
    logging.info("got local image list, {} image".format(len(image_list)))
    Batch_size = 1024
    logging.info("Batch_size=, {}".format(Batch_size))
    n = 0
    final_image = []
    final_image_id = []
    for idx,image_id in enumerate(image_list):
        # get image from local file
        try:
            image = read_image(os.path.join(LOCAL_IMAGE_PREFIX, image_id))
            final_image.append(image)
            final_image_id.append(image_id)
            n += 1
        except:
            logging.info("Failed to read image: {}".format(os.path.join(LOCAL_IMAGE_PREFIX, image_id)))
            raise

        if n == Batch_size or idx == len(image_list) - 1:
            np_final_image_id = np.array(final_image_id)
            np_final_image = np.array(final_image)
            n = 0
            final_image = []
            final_image_id = []
            yield np_final_image_id, np_final_image



def verify_local_output(output_probs):
    """
    This function prints the groundtruth and prediction for the participant to verify, calculates average FPS.

    params:
    - output_probs (dict): dict of probability of every video
    - output_times (dict): dict of processing time of every video
    - num_frames (dict): dict of number of frames extracting from every video
    """
    # gts = json.load(open('test-data/local_test_groundtruth.json'), parse_int=float)
    with open (LOCAL_LABEL_LIST_PATH,'r') as f:
        gts = json.load(f)



    for k in gts:
        #import pdb;pdb.set_trace()
        assert k in output_probs, ValueError("The detector doesn't work on image {}".format(k))


        logging.info("Image ID: {}".format(k))
        logging.info("\tgt: {}".format(gts[k]))
        logging.info("\toutput probability: {}".format(output_probs[k]))
        # logging.info("\toutput time: {}".format(output_times[k]))

        logging.info(" ")


    logging.info("Done")


