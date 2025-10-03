from lightning.fabric import Fabric

# fabric = Fabric()
# Running with TPU Accelerator using 8 TPU cores
# fabric = Fabric(devices=8, accelerator="tpu")
def get_fabric():
    fabric = Fabric(strategy="ddp_find_unused_parameters_true", 
                    accelerator="gpu", 
                    devices=2,
                    precision="16-mixed",  # T4 GPUs doesn't support bf16
                    )

    fabric.launch()
    return fabric

def convert_for_distributed(fabric, model,optimizer, lr_scheduler):
    model, optimizer, lr_scheduler = fabric.setup(model, optimizer, lr_scheduler)
    return model, optimizer, lr_scheduler

def convert_data_for_distributed(fabric, train_dataloader = None, val_dataloader = None, test_dataloader = None):
    if train_dataloader is not None:
        train_dataloader = fabric.setup(train_dataloader)
    if val_dataloader is not None:
        val_dataloader = fabric.setup(val_dataloader)
    if test_dataloader is not None:
        test_dataloader = fabric.setup(test_dataloader)
    return train_dataloader, val_dataloader, test_dataloader