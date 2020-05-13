dependencies = ['torch']

import torch.hub

from srresnet import _NetG

_pretrained_url = 'https://github.com/valgur/pytorch-SRResNet/blob/master/model/model_srresnet.pth?raw=true'


def SRResNet(pretrained=False, progress=True, map_location=None):
    if not pretrained:
        return _NetG()

    model = torch.hub.load_state_dict_from_url(
        _pretrained_url, map_location=map_location, progress=progress)["model"]
    return model
