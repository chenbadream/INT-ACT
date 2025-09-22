"""
The script to step into the training or evaluation.
Has model factory featue to select the model to train or evaluate.

"""
import os
import sys

import draccus
from modeling_pi0 import ActionPolicy

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 向上回溯一级目录，即到达项目根目录
project_root_dir = os.path.join(current_dir, "..")

# 将项目根目录添加到 sys.path
sys.path.append(project_root_dir)


from src.agent.configuration_pipeline import TrainPipelineConfig


@draccus.wrap()
def main(pipeline_cfg: TrainPipelineConfig):
    model_type = pipeline_cfg.model_cfg.type

    model_map = {
        "action": ActionPolicy,
        }

    if pipeline_cfg.eval_cfg is None:
        # only training
        from trainer import ActionTrainer
        trainer_map = {
            "action": ActionTrainer,
        }

        model_class = model_map.get(model_type, None)

        trainer_class = trainer_map.get(model_type, None)
        if trainer_class is None:
            raise ValueError(f"Model type {model_type} not supported for training.")

        trainer = trainer_class(train_cfg=pipeline_cfg, model_class=model_class)
        trainer.train()

if __name__ == "__main__":
    main()
