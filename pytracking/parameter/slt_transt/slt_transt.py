from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()
    params.debug = 0
    params.visualization = False
    params.use_gpu = True
    net_name = '/home/ubuntu/finalSubmissionRepo/SLTRLProject/checkpoints/slt_transt.pth'
    params.net = NetWithBackbone(net_path=net_name, use_gpu=params.use_gpu)
    return params
