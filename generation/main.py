import os
import hydra
from omegaconf import DictConfig
from Generators.generator import Generator
from Generators.generator_more_info import GeneratorMoreInfo
from Generators.postprocess import PostProcessor, TaskCreation

@hydra.main(version_base=None, config_path="./config", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    task_name = cfg.experiment.name
    
    if task_name == "generate":
        Generator(cfg)

    elif task_name == "generate_more_info":
        GeneratorMoreInfo(cfg)
        
    elif task_name == "server":
        command = [
            "python",
            "-m", "vllm.entrypoints.openai.api_server",
            "--host", cfg.host,
            "--port", str(cfg.port),
            "--model", cfg.model.model_id,
            "--tensor-parallel-size", str(cfg.model.tensor_parallel_size),
            "--trust-remote-code"
        ]
        command = " ".join(command)
        os.system(command)
        
    elif task_name == "postprocess":
        PostProcessor(cfg)
    
    elif task_name == "task_creation":
        TaskCreation(cfg)
        
    else:
        raise ValueError(f"Unknown experiment name: {task_name}")

if __name__ == "__main__":
    run_experiment()