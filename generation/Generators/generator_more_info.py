import os
import json
import dspy
import numpy as np
from tqdm import tqdm
from typing import List
from utils.dataobjs import UserPersona, UserFinancialProfile, Subscription
from pydantic import BaseModel, Field

class IntField(BaseModel):
    value: int = Field(description="The integer value for the field.")
    reasoning: str = Field(description="The justification for why this value was chosen based on the persona.")

class StringField(BaseModel):
    value: str = Field(description="The string value for the field.")
    reasoning: str = Field(description="The justification for why this value was chosen based on the persona.")

class BoolField(BaseModel):
    value: bool = Field(description="The boolean value for the field.")
    reasoning: str = Field(description="The justification for why this value was chosen based on the persona.")

class ListOfSubscriptionsField(BaseModel):
    value: List[Subscription] = Field(description="The list of strings for the field.")
    reasoning: str = Field(description="The justification for why this list was chosen based on the persona.")
    
class ListOfStringField(BaseModel):
    value: List[str] = Field(description="The list of strings for the field.")
    reasoning: str = Field(description="The justification for why this list was chosen based on the persona.")

class GeneratorMoreInfoDspy(dspy.Signature):
    """
    You are an expert financial analyst. Your task is to meticulously analyze the provided user persona and generate a structured financial profile.

    **Instructions:**
    1.  **Analyze the Persona:** Carefully read the entire persona to understand the user's lifestyle, values, and explicit statements.
    2.  **Infer Attributes:** For each output field, infer the reasonable value based on the persona.
    3.  **Provide Justification:** For each inference, provide a detailed 'reasoning' that explicitly quotes or references specific details from the persona description. Your reasoning is the most important part of the output.
    4.  **Strictly Adhere to Format:** Your final output must be a valid JSON object that follows the defined structure for every field.
    """
    
    # --- Input Field ---
    user_persona: UserPersona = dspy.InputField()
    
    # --- Output Fields ---
    income_level: StringField = dspy.OutputField(
        desc="A brief description of the user's income status. Choose from 1) no income, 2) low income, 3) med income, 4) high income, 5) very high income."
    )
    credit_limit: IntField = dspy.OutputField(
        desc="A reasonable integer credit limit for the user (e.g., 6000)."
    )
    payment_habit: StringField = dspy.OutputField(
        desc="""A description of how the user pays their bills. Choose from
1) automatic_payment (pay in-full at the end of the date), 
2) manual_payment, which avoids interest, 
3) minimum_payment, which induces interest."""
    )
    subscriptions: ListOfSubscriptionsField = dspy.OutputField(
        desc="""A dictionary of fixed recurring subscriptions. This can include e.g. Amazon, Netflix, Xfinity, Spotify, AT&T, etc.."""
    )
    recurring_variable_bills: ListOfSubscriptionsField = dspy.OutputField(
        desc="""A dictionary of recurring bills with variable amounts, such as utility bill, water bill, property taxes, home or car insurance, etc.."""
    )
    life_events: ListOfStringField = dspy.OutputField(
        desc="Significant life events impacting spending (e.g., 'newborn_baby_6_months_ago', 'moved_to_new_city_3_months_ago')."
    )
    car_ownership: StringField = dspy.OutputField(
        desc="Car ownership status (e.g., 'owns_1_car', 'no_car', 'owns_2_cars')."
    )
    spending_patterns: StringField = dspy.OutputField(
        desc="A general description of the user's typical spending behavior. Examples: 'frequent small purchases (e.g., daily coffee, snacks)', 'balanced spending with occasional large purchases (e.g., travel, electronics)', 'heavy online spending', 'budget-conscious and infrequent major purchases', 'primarily in-person transactions', 'impulsive spending'. This should reflect their lifestyle, hobbies, and income level."
    )
    
class GeneratorMoreInfo:
    def __init__(self, cfg):
        self.scorer = dspy.ChainOfThought(GeneratorMoreInfoDspy)
        self.examples = self.load_training_samples("Generators/fewshot_examples")
        if cfg.experiment.name == "generate_more_info":
            choices = [
                "Spenders: comfortably cover their basic needs and consistently enjoy discretionary purchases, but they save a relatively small portion of their income.",
                "Accumulators: income vastly exceeds their lifestyle needs, allowing them to save and invest the overwhelming majority of their money.",
                "Balancers: intentionally prioritize saving and investing for the future while still maintaining a comfortable current lifestyle.",
                "Survivors: live paycheck to paycheck, spending nearly all of their income on immediate necessities with little to no ability to save.",
                "Minimalists: consciously live on far less than they earn, enabling a very high savings rate to achieve financial independence."
            ]
            lm = dspy.LM(
                f"openai/{cfg.model.model_id}",
                api_base=f"http://{cfg.host}:{cfg.port}/v1",
                api_key="EMPTY",
                model_type='chat',
                max_tokens=8192  
            )
            dspy.settings.configure(lm=lm)
            import json
            with open(os.path.join(cfg.data.persona_path, f"{cfg.experiment.file_idx}.json"), "r") as f:
                data = json.load(f)
            new_data = []
            for user_persona in tqdm(data):
                user_financial_profile = self.main(user_persona)
                user_financial_profile["subscriptions"] = [s.asdict() for s in user_financial_profile["subscriptions"]]
                user_financial_profile["recurring_variable_bills"] = [s.asdict() for s in user_financial_profile["recurring_variable_bills"]]
                user_financial_profile["spending_patterns"] = str(np.random.choice(choices))
                user_financial_profile["credit_limit"] = user_financial_profile["credit_limit"] + 4000
                new_data.append({
                    "user_persona": user_persona,
                    "user_financial_profile": user_financial_profile
                })
                with open(os.path.join(cfg.data.augmented_persona_path, f"{cfg.experiment.file_idx}.json"), "w") as f:
                    json.dump(new_data, f, indent=4)
    
    def main(self, user_persona: UserPersona) -> UserFinancialProfile:
        out = self.scorer(user_persona=user_persona, demos=self.examples)
        return {
            "income_level": out.income_level.value,
            "credit_limit": out.credit_limit.value,
            "payment_habit": out.payment_habit.value,
            "subscriptions": out.subscriptions.value,
            "recurring_variable_bills": out.recurring_variable_bills.value,
            "life_events": out.life_events.value,
            "car_ownership": out.car_ownership.value,
            "spending_patterns": out.spending_patterns.value
        }
        
    def load_training_samples(self, examples_path):
        files = os.listdir(examples_path)
        examples = []
        for file in files:
            with open(os.path.join(examples_path, file), "r") as f:
                data = json.load(f)
            user_persona = data["user_persona"]
            user_financial_profile = data["user_financial_profile"]
            example = dspy.Example(
                user_persona=user_persona,
                **user_financial_profile
            ).with_inputs("user_persona")
            examples.append(example)
        return examples
    