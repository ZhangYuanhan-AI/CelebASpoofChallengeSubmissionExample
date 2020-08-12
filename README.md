# DeeperForensics Challenge Submission Example
This repo provides an example Docker image for submission of DeeperForensics Challenge 2020. The submission is in the form of online evaluation.

**Note**: **We highly recommend the participants to test their Docker images before submission.** Or your code may not be run properly on the online evaluation platform. Please refer to [Test the Docker image locally](#test-the-docker-image-locally) for instructions.

## Before you start: request resource provision

First, create an account on the [challenge website](https://competitions.codalab.org/competitions/22955), as well as an [AWS account](https://aws.amazon.com/account/) (in any region except Beijing and Ningxia). Then, send your **AWS account id (12 digits)** and **an email address** to the orgnizers' email address: [deeperforensics@gmail.com](mailto:deeperforensics@gmail.com). We will allocate evaluation resources for you.

## Install and configure AWS CLI
Then you should install AWS CLI (we recommend version 2). Please refer to https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html.

After installation, you should configure the settings that the AWS Command Line Interface (AWS CLI) uses to interact with AWS. Please refer to https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html.

## Install Docker Engine
In order to build your Docker image, you should install Docker Engine first. Please refer to [Install Docker Engine](https://docs.docker.com/engine/install/).

## Obtain this example

Run the following command to clone this submission example repo:

```bash
git clone https://github.com/Drdestiny/DeeperForensicsChallengeSubmissionExample
```

## Include your own algorithm & Build Docker image

- Your algorithm codes should be put in `model`.
- Your detector should inherit `DeeperForensicsDetector` class in `eval_kit/detector.py`. For example:

```python
sys.path.append('..')
from eval_kit.detector import DeeperForensicsDetector

class TSNPredictor(DeeperForensicsDetector): # You can give your detector any name.
    ...
```
You need to implement the abstract function `predict(self, video_frames)` in your detector class:

```python
  @abstractmethod
  def predict(self, video_frames):
      """
      Process a list of video frames, the evaluation toolkit will measure the runtime of every call to this method.
      The time cost will include any thing that's between the image input to the final bounding box output.
      The image will be given as a numpy array in the shape of (H, W, C) with dtype np.uint8.
      The color mode of the image will be **RGB**.
      
      params:
          - video_frames (list): a list of numpy arrays with dtype=np.uint8 representing frames of **one** video
      return:
          - probablity (float)
      """
      pass
```

- Modify Line 29 in `run_evalution.py` to import your own detector for evaluation.

```python
########################################################################################################
# please change this line to include your own detector extending the eval_kit.detector.DeeperForensicsDetector base class.
from tsn_predict import TSNPredictor as DeeperForensicsDetector
########################################################################################################
```

- Also, you can implement frame extracting using your own method. If so, you should modify Line 97 - 114 in `eval_kit/client.py` (optional). **Make sure that return type remains the same.**
```python
def extract_frames(video_path):
    """
    Extract frames from a video

    params:
        - video_local_path (str): the path of video.
    return:
        - frames (list): a list containing frames cut from the video.
    """
    ########################################################################################################
    # Please change these lines below to implement your own frame extracting method, or just use it.  
    vid = cv2.VideoCapture(video_path)
    cnt = 0
    skip_cnt = 0
    frames = []
    while True:
        success, frame = vid.read()
        if not success:
            break
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            cnt += 1
        skip_cnt += 1 # prevent time out
        # detect first 50 frames
        if cnt >= 50 or skip_cnt >= 300:
            break
    vid.release()
    return frames
    # Your change ends here.
    ########################################################################################################
```

- Edit `Dockerfile` to build your Docker image. If you don't know how to use Dockerfile, please refer to:
  -  [Best practices for writing Dockerfiles (English)](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#dockerfile-instructions)
  -  [使用 Dockerfile 定制镜像 (Chinese)](https://yeasy.gitbook.io/docker_practice/image/build)
  -  [Dockerfile 指令详解 (Chinese)](https://yeasy.gitbook.io/docker_practice/image/dockerfile)

## Test the Docker image locally

The online evaluation for submissions may take several hours. It is slow to debug on the cloud. Here we provide some tools for the participants to locally test the correctness of the submission.

Before running, modify Line 32 in `local_test.py`.

To verify your algorithm can be run properly, run the following command:

```bash
docker run -it deeperforensics-challenge-<your_aws_id> python3 local_test.py
```

**Please refer to step 2 and 3 in** [Submit the Docker image](#submit-the-docker-image) **to learn how to tag your Docker image.**

It will run the algorithms in the evaluation workflow on some sample images and print out the results.
You can compare the output of your algorithm with the ground truth for the sample images. 

The output will look like:

```
    ================================================================================
    all videos finished, showing verification info below:
    ================================================================================
    
INFO:root:Video ID: 000000.mp4, Runtime: 0.7639188766479492
INFO:root:	gt: 0
INFO:root:	output probability: 0.00022679567337036133
INFO:root:	number of frame: 50
INFO:root:	output time: 0.7639188766479492
INFO:root: 
INFO:root:Video ID: 000001.mp4, Runtime: 0.26082539558410645
INFO:root:	gt: 1
INFO:root:	output probability: 0.9986529353773221
INFO:root:	number of frame: 50
INFO:root:	output time: 0.26082539558410645
INFO:root: 
INFO:root:Video ID: 000002.mp4, Runtime: 0.25582146644592285
INFO:root:	gt: 1
INFO:root:	output probability: 0.9995381628687028
INFO:root:	number of frame: 50
INFO:root:	output time: 0.25582146644592285
INFO:root: 
INFO:root:Video ID: 000003.mp4, Runtime: 0.2605171203613281
INFO:root:	gt: 0
INFO:root:	output probability: 0.00013506412506103516
INFO:root:	number of frame: 50
INFO:root:	output time: 0.2605171203613281
INFO:root: 
INFO:root:Done. Average FPS: 129.779

```

## Submit the Docker image

First, you should install AWS CLI version 2. Please refer to [Installing the AWS CLI version 2 on Linux](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html).

After installing, set your AWS ID and credential. Please refer to [Configuration and credential file settings](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html). **Note: Please use your root access keys.** (refer to [Creating Access Keys for the Root User](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_root-user.html#id_root-user_manage_add-key)) Or your submission may fail.

Then, you can push your Docker image to the allocated ECR repo:

1. Retrieve the login command to use to authenticate your Docker client to your registry.
Use the AWS CLI:

```bash
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 212923332099.dkr.ecr.us-west-2.amazonaws.com
```

2. Build your Docker image using the following command. For information on building a Docker file from scratch see the instructions here. You can skip this step if your image is already built:

```bash
cd ../DeeperForensicsChallengeSubmissionExample
docker build -t deeperforensics-challenge-<your_aws_id> .  # . means the current path. Please don't lose it.
```

For example:

```bash
docker build -t deeperforensics-challenge-123412341234 . 
```

3. After the build is completed, tag your image so you can push the image to the repository:

```bash
docker tag deeperforensics-challenge-<your_aws_id>:latest 212923332099.dkr.ecr.us-west-2.amazonaws.com/deeperforensics-challenge-<your_aws_id>:latest
```

4. Run the following command to push this image to your the AWS ECR repository:

```bash
docker push 212923332099.dkr.ecr.us-west-2.amazonaws.com/deeperforensics-challenge-<your_aws_id>:latest
```

After you pushed to the repo, the evaluation will automatically start. In **3 hours** you should receive a email with the evaluation result if the evaluation is successful. Finally, you can submit the evaluation result to the [challenge website](https://competitions.codalab.org/competitions/22955).
