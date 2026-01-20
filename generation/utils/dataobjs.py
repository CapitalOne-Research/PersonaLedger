import os
import json
from typing import List, Dict
from dataclasses import dataclass, asdict

list_of_fields = [
    "persona", 
    "professional_persona", 
    "sports_persona", 
    "arts_persona", 
    "travel_persona", 
    "culinary_persona", 
    "skills_and_expertise_list", 
    "hobbies_and_interests_list", 
    "career_goals_and_ambitions", 
    "sex", 
    "age", 
    "marital_status", 
    "education_level", 
    "bachelors_field", 
    "occupation"
]

@dataclass
class UserPersona:
    """ A class that stores the user personas.
    """
    persona: str
    professional_persona: str
    sports_persona: str
    arts_persona: str
    travel_persona: str
    culinary_persona: str
    skills_and_expertise_list: str
    hobbies_and_interests_list: str
    career_goals_and_ambitions: str
    sex: str
    age: int
    marital_status: str
    education_level: str
    bachelors_field: str
    occupation: str
    
    def asdict(self):
        return asdict(self)

@dataclass
class Transaction:
    """ A class that stores transaction information.
    """
    # relevant_to: str
    timestamp: str
    merchant_name: str
    merchant_type: str
    card_present_or_not: str # yes or no
    amount: float
    
    def asdict(self):
        return asdict(self)

    def __repr__(self):
        return "{" + f"timestamp: {self.timestamp}, merchant_name: {self.merchant_name}, merchant_type: {self.merchant_type}, card_present_or_not: {self.card_present_or_not}, amount: {self.amount}" + "}"

@dataclass
class Rule:
    """ A class that stores a credit card rule. Add properties by yourself if needed.
    """
    rule: str
    
@dataclass    
class Subscription:
    date_to_charge: int
    amount: float
    charge_frequency_month: int
    std: float
    merchant_name: str
    product_description: str
    
    def asdict(self):
        return asdict(self)
    
@dataclass
class UserFinancialProfile:
    income_level: str
    credit_limit: int
    payment_habit: str
    subscriptions: List[Subscription]
    recurring_variable_bills: List[Subscription]
    life_events: List[str]
    car_ownership: str
    spending_patterns: str
    
    def asdict(self):
        return asdict(self)