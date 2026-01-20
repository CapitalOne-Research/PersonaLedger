import os
import time
import warnings

def get_lm_provider(cfg):
    name = cfg.experiment.lm_provider_name
    if name == "LocalServer":
        return LocalServer(cfg)
    else:
        raise Exception("Invalid lm_provider name.")
    
class LocalServer:
    def __init__(self, cfg):
        import openai
        self.client = openai.OpenAI(
            base_url=f"http://localhost:8000/v1",
            api_key="not-needed"
        )
        self.cfg = cfg
        self.max_tokens=self.cfg.experiment.max_tokens
        self.temperature=self.cfg.experiment.temperature
        self.top_k=self.cfg.experiment.top_k
        self.top_p=self.cfg.experiment.top_p
    
    def main(self, messages):
        response = self.client.chat.completions.create(
            model=self.cfg.model.model_id,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        response_text = response.choices[0].message.content
        return response_text
    
