from mmengine import Registry
from torch.optim.lr_scheduler import LambdaLR, StepLR

LR_SCHEUDLERS = Registry("lr_schedulers")

LR_SCHEUDLERS.register_module(name="LambdaLR", module=LambdaLR)
LR_SCHEUDLERS.register_module(name="StepLR", module=StepLR)

VOCODERS = Registry("vocoders")
DATASETS = Registry("datasets")
DIFFUSIONS = Registry("diffusions")
ENCODERS = Registry("encoders")