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

WORKSPACE_BUCKET = 'deeperforensics-eval-workspace'
IMAGE_LIST_PATH = 'test-data/deeperforensicis_runtime_eval_video_list.txt'
IMAGE_PREFIX = 'test-data/'
UPLOAD_PREFIX = 'test-output/'
TMP_PATH = '/tmp'


def _get_s3_image_list(s3_bucket, s3_path):
    s3_client = boto3.client('s3', region_name='us-west-2')
    f = BytesIO()
    s3_client.download_fileobj(s3_bucket, s3_path, f)
    lines = f.getvalue().decode('utf-8').split('\n')
    return [x.strip() for x in lines if x != '']


def _download_s3_image(s3_bucket, s3_path, filename):
    s3_client = boto3.client('s3', region_name='us-west-2')
    local_path = os.path.join(TMP_PATH, filename)
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
    return os.environ['DEEPERFORENSICS_EVAL_JOB_NAME']


def upload_eval_output(output_probs, output_times, job_name):
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
            "runtime": output_times[i],
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

    for image_id in image_list:
        # get video from s3
        st = time.time()
        try:
            #download required image from s3 to local
            _download_s3_image(WORKSPACE_BUCKET, os.path.join(IMAGE_PREFIX, image_id), image_id)
        except:
            logging.info("Failed to download image: {}".format(os.path.join(IMAGE_PREFIX, image_id)))
            raise
        image_local_path = os.path.join(TMP_PATH, image_id) # local path of the video named video_id
        image = read_image(image_local_path)
        elapsed = time.time() - st
        logging.info("image downloading & image reading time: {}".format(elapsed))
        yield image_id, image
        try:
            os.remove(image_local_path) # remove the video named video_id
        except:
            logging.info("Failed to delete this image, error: {}".format(sys.exc_info()[0]))

def get_local_image(max_number=None):
    """
    This function returns a iterator of image.
    It is used for local test of participating algorithms.
    Each iteration provides a tuple of (image_id, image), each image will be in RGB color format with array shape of (height, width, 3)
    
    return: tuple(video_id: str, image: numpy.array)
    """
    image_list = [x.strip() for x in open(IMAGE_LIST_PATH)]
    logging.info("got local image list, {} image".format(len(image_list)))

    for image_id in image_list:
        # get video from local file
        try:
            image = read_image(os.path.join(IMAGE_PREFIX, image_id))
        except:
            logging.info("Failed to read image: {}".format(os.path.join(IMAGE_PREFIX, image_id)))
            raise
        yield image_id, image


def verify_local_output(output_probs, output_times):
    """
    This function prints the groundtruth and prediction for the participant to verify, calculates average FPS.

    params:
    - output_probs (dict): dict of probability of every video
    - output_times (dict): dict of processing time of every video
    - num_frames (dict): dict of number of frames extracting from every video
    """
    # gts = json.load(open('test-data/local_test_groundtruth.json'), parse_int=float)
    gts = json.load(open('test-data/local_test_groundtruth.json'))

    all_time = 0
    all_num_frames = 0
    for k in gts:

        assert k in output_probs and k in output_times, ValueError("The detector doesn't work on image {}".format(k))

        all_time += output_times[k]
        all_num_frames += num_frames[k]

        logging.info("Image ID: {}, Runtime: {}".format(k, output_times[k]))
        logging.info("\tgt: {}".format(gts[k]))
        logging.info("\toutput probability: {}".format(output_probs[k]))
        logging.info("\toutput time: {}".format(output_times[k]))

        logging.info(" ")

    logging.info("Done")


