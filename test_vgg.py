import argparse
import easydict

from utils.config import *
from utils.count_model import *
from agents import *
from torchsummaryX import summary

if __name__ == '__main__':
    config = easydict.EasyDict()
    config.exp_name = "vgg16_exp_cifar100_0"
    config.log_dir = os.path.join("experiments", config.exp_name, "logs/")

    config.load_file = os.path.join("experiments", "vgg16_exp_cifar100_0","checkpoints", "checkpoint.pth")
    config.cuda = True
    config.gpu_device = 0
    config.seed = 1
    config.milestones = [120,160]
    config.gamma = 0.9
    config.img_size = 32
    config.num_classes = 100
    config.data_mode = "download"
    config.data_loader_workers = 4
    config.pin_memory = True
    config.async_loading = True
    config.batch_size = 128
    config.async_loading = True
    config.max_epoch = 200

    #torch.cuda.init()

    #agent = globals()["VGG_BN_cifar"](config)
    agent = VGG_BN_cifar(config)
    agent.init_graph()
    best,history = agent.train(specializing=False, freeze_conv=False)

    torch.save(agent.model.state_dict(),"vgg16_pretrained.model")

    agent.load_checkpoint(config.load_file)
    agent.compress(method = 'greedy',k=0.62)
    summary(agent.model, torch.zeros((1, 3, 32, 32)).to(torch.device("cuda")))
    config.max_epoch = 50
    config.milestones = [30,45]
    agent.current_epoch = 0
    agent.current_iteration = 0
    best,history = agent.train(specializing=False, freeze_conv=False)
    torch.save(agent.model.state_dict(),"vgg16.model")

    print(count_model_param_nums(agent.model) / 1e6)
    print(count_model_flops(agent.model, input_res=32)/ 1e9)
