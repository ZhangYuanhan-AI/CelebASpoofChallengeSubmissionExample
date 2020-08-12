from abc import ABC, abstractmethod


class CelebASpoofDetector(ABC):
    def __init__(self):
        """
        Participants should define their own initialization process.
        During this process you can set up your network. The time cost for this step will
        not be counted in runtime evaluation
        """

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
