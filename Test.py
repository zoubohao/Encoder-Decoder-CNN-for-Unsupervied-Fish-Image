from torch.optim.swa_utils import SWALR
import torch.optim as optim
import torchvision as tv
model  = tv.models.resnet50(num_classes=3)
optimizer = optim.SGD(model.parameters(),lr = 0.1)
swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=1, anneal_strategy="cos")
for i in range(100):
    print(optimizer.param_groups[0]["lr"])
    swa_scheduler.step()


