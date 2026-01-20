import os
import json
import multiprocessing
from utils.dataobjs import UserPersona, UserFinancialProfile, Transaction
from tqdm import tqdm
from Generators.generation_tracker import GenerationTracker

class Generator:
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.experiment.name == "generate":
            from utils.lm_providers import get_lm_provider
            self.lm = get_lm_provider(self.cfg)
            user_persona_path = self.cfg.data.augmented_persona_path
            json_id = int(self.cfg.experiment.json_name)
            start_idx = self.cfg.experiment.start_idx
            end_idx = self.cfg.experiment.end_idx
            processes = []
            for user_id in range(start_idx, end_idx):
                global_id = 1000 * json_id + user_id
                p = multiprocessing.Process(target=self.main_generate, args=(json_id, user_id, global_id))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
        else:
            raise ValueError("Invalid experiment name is passed.")
        
    def main_generate(self, json_id, user_id, global_id):
        generation_inputs = self.get_generation_inputs()
        terminate = False
        for gen_id, (user_persona, user_financial_profile) in enumerate(generation_inputs):
            if gen_id != user_id:
                continue
            
            savepath = os.path.join(self.cfg.data.export_path, self.cfg.experiment.json_name)
            if not os.path.exists(savepath):
                os.system(f"mkdir -p {savepath}")
                
            tracker = GenerationTracker(user_persona, user_financial_profile, self.cfg)
            conversations = [
                {"role": "system", "content": tracker.get_system_prompt()},
                {"role": "user", "content": tracker.get_running_user_prompt()},
            ]
            trxn_id = 0
            while trxn_id < self.cfg.experiment.num_transactions:
                if len(conversations) > self.cfg.context_days * 2 + 1:
                    truncate_index = self.cfg.context_days * 2 - 1
                    conversations_input = conversations[:1] + conversations[-truncate_index:]
                else:
                    conversations_input = conversations

                response_text = self.lm.main(conversations_input)
                
                user_prompt = conversations[-1]["content"]
                success = tracker.update_history(user_prompt, response_text)
                if not success:
                    continue
                new_user_prompt = tracker.get_running_user_prompt()
                conversations = conversations + [
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": new_user_prompt}
                ]
                
                if tracker.applied_rules[-1].charge_off_count == 3 or tracker.applied_rules[-3].balance < 0:
                    tracker.transactions.append(Transaction(
                        timestamp="charge-off",
                        merchant_name="charge-off",
                        merchant_type="charge-off",
                        card_present_or_not="charge-off",
                        amount=0
                    ))
                    terminate = True
                
                export_json = os.path.join(savepath, f"{str(global_id).zfill(6)}.json")
                tracker.export(export_json)
                with open(os.path.join(savepath, f"conversation_{str(global_id).zfill(6)}.json"), "w") as f:
                    json.dump(conversations, f, indent=4)
                
                trxn_id += 1
                
                print(f"generated transactions saved to {export_json}. Day: {trxn_id}.")
                    
                if terminate:
                    break
        
    def get_generation_inputs(self):
        file = os.path.join(
            self.cfg.data.augmented_persona_path, 
            f"{self.cfg.experiment.json_name}.json")
        with open(file, "r") as f:
            data = json.load(f)
        return [(
            UserPersona(**item["user_persona"]), 
            UserFinancialProfile(**item["user_financial_profile"])
        ) for item in data]