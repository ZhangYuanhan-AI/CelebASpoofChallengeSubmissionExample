# CelebA-Spoof Challenge Submission Example
This repo provides an example Docker image for submission of CelebA-Spoof Challenge 2020. The submission is in the form of online evaluation.

**Note**: **We highly recommend the participants to test their Docker images before submission.** Or your code may not be run properly on the online evaluation platform. Please refer to [Test the Docker image locally](#test-the-docker-image-locally) for instructions.

## Before you start: request resource provision

1. Create an account on the [challenge website (CodaLab)](https://competitions.codalab.org/competitions/26210), as well as an [AWS account](https://aws.amazon.com/account/) (in any region except Beijing and Ningxia). 
2.  [Register](https://competitions.codalab.org/competitions/26210#participate) CelebA-Spoof Challenge 2020 using the created CodaLab account.
3. Send yours following information to the orgnizers' email address: [celebaspoof@gmail.com](mailto:celebaspoof@gmail.com). We will check and allocate evaluation resources for you. (Very Important!)
   - CodaLab user name (i.e. team name)
   - The number of team members
   - Affiliation
   - CodaLab email address
   - AWS account id (12 digits)


## Install and configure AWS CLI
AWS CLI is required. ([version 2](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) is recommended)

After installation, you should configure the settings that the AWS Command Line Interface (AWS CLI) uses to interact with AWS:

1. Generate the AWS Access Key ID and AWS Secret Access Key in IAM: AWS Management Console -> Find Services -> Enter 'IAM' -> Choose IAM (Manage access to AWS resources) -> Delete your root access keys -> Manage security credentials -> Access keys (access key ID and secret access key) -> Create New Access Key.
2. Run this command:
   `aws configure`
   
   Then it will require you to input AWS Access Key ID, AWS Secret Access Key, Default region name (please input us-west-2) and Default output format (left it empty). If you still have questions, please refer to [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).

## Install Docker Engine
In order to build your Docker image, you should install Docker Engine first. Please refer to [Install Docker Engine](https://docs.docker.com/engine/install/).

## Install nvidia-docker
Because GPU is necessary for both the local test and online evaluation, we also need to install nvidia-docker. Please refer to [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

## Obtain this example

Run the following command to clone this submission example repo:

```bash
git clone https://github.com/Davidzhangyuanhan/CelebASpoofChallengeSubmissionExample.git
```

## Include your own algorithm & Build Docker image

- Your algorithm codes should be put in `model`.
- Your detector should inherit CelebASpoofDetector class in `eval_kit/detector.py`. For example:

```python
sys.path.append('..')
from eval_kit.detector import CelebASpoofDetector

class AENetPredictor(CelebASpoofDetector): # You can give your detector any name.
    ...
```
You need to implement the abstract function `predict(self, image)` in your detector class:

```python
    @abstractmethod
    def predict(self, image):
        """
        Process a list of image, the evaluation toolkit will measure the runtime of every call to this method.
        The time cost will include any thing that's between the image input to the final prediction score.
        The image will be given as a numpy array in the shape of (H, W, C) with dtype np.uint8.
        The color mode of the image will be **RGB**.
        
        params:
            - image (np.array): numpy array of required image
        return:
            - probablity
        """
        pass

```

- Modify Line 29 in `run_evalution.py` to import your own detector for evaluation.

```python
########################################################################################################
# please change this line to include your own detector extending the eval_kit.detector.CelebASpoofDetector base class.
from predictor import AENetPredictor as CelebASpoofDetector
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
docker run -it celeba-spoof-challenge-<your_aws_id> python3 local_test.py
```

**Please refer to step 2 and 3 in** [Submit the Docker image](#submit-the-docker-image) **to learn how to tag your Docker image.**

It will run the algorithms in the evaluation workflow on some sample images and print out the results.

The output will look like:

```
    ================================================================================
    All images finished, showing verification info below:
    ================================================================================

INFO:root:Image ID: 494405.png
INFO:root:      gt: 1
INFO:root:      output probability: 0.9999998807907104
INFO:root:
INFO:root:Image ID: 494406.png
INFO:root:      gt: 1
INFO:root:      output probability: 0.999998927116394
INFO:root:
INFO:root:Image ID: 494410.png
INFO:root:      gt: 0
INFO:root:      output probability: 6.803428732382599e-06
INFO:root:
INFO:root:Image ID: 494415.png
INFO:root:      gt: 0
INFO:root:      output probability: 3.3887470181070967e-06
INFO:root:
INFO:root:Done
```

## Submit the Docker image

You can push your Docker image to the allocated ECR repo as following steps:

**(692230297653 is organizers' AWS account, please don't change)**

- **Step 1**. Retrieve the login command to use to authenticate your Docker client to your registry. Use the AWS CLI:

```bash
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 692230297653.dkr.ecr.us-west-2.amazonaws.com
```

- **Step 2**. Build your Docker image using the following command. For information on building a Docker file from scratch see the instructions here. You can skip this step if your image has already been built:

```bash
cd ../CelebASpoofChallengeSubmissionExample
docker build -t celeba-spoof-challenge-<your_aws_id> .  # . means the current path. Please don't lose it.
# Example
# docker build -t celeba-spoof-challenge-123412341234 . 
```

- **Step 3**. After the build is completed, tag your image so you can push the image to the repository:

```bash
docker tag celeba-spoof-challenge-<your_aws_id>:latest 692230297653.dkr.ecr.us-west-2.amazonaws.com/celeba-spoof-challenge-<your_aws_id>:latest
```

- **Step 4**. Run the following command to push this image to your AWS ECR repository:

```bash
docker push 692230297653.dkr.ecr.us-west-2.amazonaws.com/celeba-spoof-challenge-<your_aws_id>:latest
```

After you push to the repo, the evaluation will automatically start. In **45 minutes** you should receive a email with the evaluation result if the evaluation is successful. Finally, you can submit the evaluation result to the [challenge website](https://competitions.codalab.org/competitions/26210).

## Acknowledgments

We sincerely thank the codebase from [DeeperForensics Challenge](https://github.com/Drdestiny/DeeperForensicsChallengeSubmissionExample), especially for the instruction from [Zhengkui Guo](https://drdestiny.github.io/) and [Liming Jiang](https://liming-jiang.com/).