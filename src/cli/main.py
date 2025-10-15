from ..checkpoints_utils.checkpoint_load import load_model_from_checkpoint
from  ..pruning.tensor_prunning  import global_unstructured_prune,make_pruning_permanent
from ..report_utils.measure_latency import measure_latency
from .. report_utils.report_sparsity import report_sparsity
from ..pruning.channel_pruning import prune_model_channels
from  omegaconf import OmegaConf
import torch
cfg = OmegaConf.load("/home/temp/MyDir/Projects/aspdfpwjfpwejfwpefwef/Drons/tb_logs_big/efficientnet/version_6/hparams.yaml")


model = load_model_from_checkpoint(checkpoint_path="tb_logs_big/efficientnet/version_0/checkpoints/epoch=1-step=88.ckpt")

example_inputs = torch.randn(1, 3, 224, 224)



lat0 = measure_latency(model = model, input_size= (1,3,224,224))
print(f"Latency before: {lat0:.6f}s")

model = prune_model_channels(
    model,
    example_inputs=example_inputs,
    amount=2000,
    exclude_name_patterns=[ ],
    min_out_channels=8,
    importance="l1",
    device="cpu",
)

lat1 = measure_latency(model = model, input_size = (1,3,224,224))
print(f"Latency after:  {lat1:.6f}s  (×{lat0/max(lat1,1e-9):.2f})")