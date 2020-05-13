dependencies = ['torch']

import torch.hub

from srresnet import _NetG

_pretrained_url = 'https://github.com/valgur/pytorch-SRResNet/releases/download/v1.0/srresnet-9cdfd5af.pt'


def SRResNet(pretrained=False, progress=True, map_location=None):
    if not pretrained:
        return _NetG()

    model = torch.hub.load_state_dict_from_url(
        _pretrained_url, map_location=map_location, progress=progress, check_hash=True)["model"]
    return model
