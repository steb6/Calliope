import torch
from compressive_transformer_pytorch import CompressiveTransformer
from compressive_transformer_pytorch import AutoregressiveWrapper

model = CompressiveTransformer(
    num_tokens = 20000,
    dim = 512,
    depth = 6,
    seq_len = 1024,
    mem_len = 1024,
    cmem_len = 256,
    cmem_ratio = 4,
    memory_layers = [5,6]
).cuda()

model = AutoregressiveWrapper(model)

inputs = torch.randint(0, 20000, (1, 1024)).cuda()

optimizer = torch.optim.Adam(model.parameters())

for loss, aux_loss, _ in model(inputs, return_loss = True):
    optimizer.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    print("OPTIMIZED BY LOSS ************************************************************")
    for module_name, parameter in model.named_parameters():
        if parameter.grad is not None:
            print(module_name)
    optimizer.zero_grad(set_to_none=True)
    aux_loss.backward(retain_graph=True)
    print("OPTIMIZED BY AUX_LOSS ************************************************************")
    for module_name, parameter in model.named_parameters():
        if parameter.grad is not None:
            print(module_name)
