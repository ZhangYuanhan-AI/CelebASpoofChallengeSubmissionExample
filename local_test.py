"""
This script provides a local test routine so you can verify the algorithm works before pushing it to evaluation.

It runs your detector on several local videos and verify whether they have obvious issues, e.g:
    - Fail to start
    - Wrong output format

It also prints out the runtime for the algorithms for your references.


The participants are expected to implement a face forgery detector class. The sample detector illustrates the interface.
Do not modify other part of the evaluation toolkit otherwise the evaluation will fail.

Author: Yuanjun Xiong, Zhengkui Guo
Contact: zhengkui_guo@outlook.com

DeeperForensics Challenge
"""

import time
import sys
import logging

import numpy as np
from eval_kit.client import get_local_frames_iter, verify_local_output

logging.basicConfig(level=logging.INFO)

sys.path.append('model')
########################################################################################################
# please change these lines to include your own face detector extending the eval_kit.detector.DeeperForensicsDetector base class.
from tsn_predict import TSNPredictor as CelebASpoofDetector
########################################################################################################


def run_local_test(detector_class, image_iter):
    """
    In this function we create the detector instance. And evaluate the wall time for performing DeeperForensicsDetector.
    """

    # initialize the detector
    logging.info("Initializing detector.")
    try:
        detector = detector_class()
    except:
        # send errors to the eval frontend
        raise
    logging.info("Detector initialized.")


    # run the images one-by-one and get runtime
    output_probs = {}
    output_times = {}
    eval_cnt = 0

    logging.info("Starting runtime evaluation")
    for image_id, image in image_iter:
        time_before = time.time()
        try:
            prob = detector.predict(image)
            assert isinstance(prob, float)
            output_probs[video_id] = prob
        except:
            # send errors to the eval frontend
            logging.error("Video id failed: {}".format(video_id))
            raise
        elapsed = time.time() - time_before
        output_times[image_id] = elapsed
        logging.info("image {} run time: {}".format(image_id, elapsed))

        eval_cnt += 1

        if eval_cnt % 100 == 0:
            logging.info("Finished {} images".format(eval_cnt))

    logging.info("""
    ================================================================================
    all images finished, showing verification info below:
    ================================================================================
    """)

    # verify the algorithm output
    verify_local_output(output_probs, output_times)


if __name__ == '__main__':
    celebA_spoof_image_iter = get_local_image()
    run_local_test(DeeperForensicsDetector, celebA_spoof_image_iter)
