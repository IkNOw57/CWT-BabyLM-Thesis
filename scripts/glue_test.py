from engine.tasks.benchmark.glue import GlueBenchmark
from transformers import AutoTokenizer, AutoModel
from engine.lit.lightning_module import TaskTrainer
import torch
import argparse
import os
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'

parser = argparse.ArgumentParser()
parser.add_argument("--model_ckpt")
parser.add_argument("--params")
parser.add_argument("--mode")
parser.add_argument("--train_batch_size")
parser.add_argument("--run_name")
args = parser.parse_args()
model_ckpt = args.model_ckpt
params = args.params
mode = args.mode
train_bs = args.train_batch_size
run_name = args.run_name
#unsafe_modules = torch.serialization.get_unsafe_globals_in_checkpoint(model_ckpt)
#print(unsafe_modules)
#torch.serialization.add_safe_globals(unsafe_modules)

#torchtrainer load 
#task_trainer = torch.load(model_ckpt,map_location='cuda',weights_only=False)

print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
  print(f"GPU: {i}: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

task_trainer = TaskTrainer.load_from_checkpoint(model_ckpt, map_location='cuda:0',weights_only=False)

tokenizer = task_trainer.task.tokenizer

if mode == "mlm":
    model = task_trainer.task.mlm_model
    backbone = 'mlm'
else: 
    model = task_trainer.task.lm_model

backbone = model



def main():
  GlueBenchmark(  tokenizer, backbone, logger='wandb', logger_args={'project': 'GLUE'+run_name}, train_batch_size=int(train_bs), accumulate_grad_batches=1,
        learning_rate=1e-5
    )

if __name__ =='__main__':
  main()
  #mp.spawn(main()#,args=(args,),nprocs=1)