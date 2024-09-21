import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from fvcore.nn import FlopCountAnalysis

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = model().cuda()  # 替换为你的模型类
input = torch.randn(1, 3, 256, 256).cuda()  # 例如输入是 [batch_size, channels, height, width]
print(f"Model has {count_parameters(model)} trainable parameters")
flops = FlopCountAnalysis(model, input)
print(f"FLOPs: {flops.total()}")
checkpoint.done()

