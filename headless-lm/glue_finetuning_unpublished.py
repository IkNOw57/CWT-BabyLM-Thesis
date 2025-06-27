from engine.tasks.benchmark.glue import GlueBenchmark
# from transformers import AutoTokenizer, AutoModel
from engine.lit.lightning_module import TaskTrainer
import argparse

# model_id = 'nthngdy/headless-bert-bs64-owt2'
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# mlm_model = AutoModel.from_pretrained(model_id)
# backbone = mlm_model


parser = argparse.ArgumentParser()
parser.add_argument("--model_ckpt")
parser.add_argument("--mode")
args = parser.parse_args()
model_ckpt = args.model_ckpt
mode = args.mode

task_trainer = TaskTrainer.load_from_checkpoint(model_ckpt,weights_only=True, map_location="cpu")
tokenizer = task_trainer.task.tokenizer

if mode == "mlm":
    model = task_trainer.task.mlm_model
else: 
    model = task_trainer.task.lm_model

backbone = model



def main():
    GlueBenchmark(
        tokenizer, backbone, logger='wandb', logger_args={'project': 'GLUE'}, train_batch_size=32, accumulate_grad_batches=1,
        learning_rate=1e-5, loss_type = 'emb'
    ) #,weighted_ce=True, weight_decay=0.01, shuffle=True

if __name__ =='__main__':
    main()