import sys
import numpy as np
import torchvision
#from sklearn.metrics import confusion_matrix

from models import TSN
from transforms import *
from ops import ConsensusModule

sys.path.append('..')
from eval_kit.detector import CelebASpoofDetector

def pretrain(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(name, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")

class TSNPredictor(CelebASpoofDetector):

    def __init__(self):
        self.num_class = 2
        self.net = AENet(num_classes = self.num_class)
        checkpoint = torch.load('./model/fake_bninception_AF_rgb_model_best.pth.tar')
        print("model step {} best prec@1: {}".format(checkpoint['step'], checkpoint['best_prec1']))
        pretrain(self.net,checkpoint['state_dict'])


        self.transform = torchvision.transforms.Compose([
            transforms.Resize((self.new_width, self.new_height)),
            transforms.ToTensor(),
            ])
        
        self.net.cuda()
        self.net.eval()



    def preprocess_data(self, image):
        image = Image.fromarray(frame).resize((224,224), Image.BICUBIC)
        return processed_data

    def eval_image(self, image):
        data = image.unsqueeze(0)
        channel = 3
        input_var = data.view(-1, channel, data.size(2), data.size(3))
        with torch.no_grad():
            rst = self.net(input_var).detach().cpu().numpy().copy()
        return rst.reshape(1, self.num_class)

    def predict(self, image):
        data = self.preprocess_data(image)
        rst = self.eval_image(data)
        probability = np.array(rst)
        return probability[0][1]