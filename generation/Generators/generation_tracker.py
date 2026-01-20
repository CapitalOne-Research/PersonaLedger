from typing import List
import math
import numpy as np
import json
from datetime import datetime, timedelta
from dateutil.rrule import rrule, MONTHLY
from bisect import bisect_left
from utils.dataobjs import UserPersona, UserFinancialProfile, Transaction

FORMAT_CODE = "%Y-%m-%d %H:%M:%S"

class Rule:
    def __init__(self, user_financial_profile):
        self.user_financial_profile = user_financial_profile
    
    def update(self, transactions, all_transactions, today):
        """Given the newly created transactions and historical transactions (before adding the new transactions),
        1) update the internal state,
        2) potentially modify transactions,
        3) MUST NOT modify all_transactions. It is only for the context for 1) and 2).
        Args:
            transactions (List[Transaction]): newly created transactions
            all_transactions (List[Transaction]): historical transactions
            today (datetime object): today's date
        """
        pass
    
    def prompt_text(self, **kwargs):
        return None
    
class Holiday(Rule):
    def __init__(self, user_financial_profile):
        from datetime import date
        import holidays
        self.us_holidays = holidays.US()
    def prompt_text(self, **kwargs):
        today = kwargs.get('today')
        if today in self.us_holidays:
            return f"Holiday notification: Today is {self.us_holidays.get(today)}."
    
class AmountPolarity(Rule):
    """ Modify any positive payment into negative. Modify any negative transaction into positive.
    """
    def __init__(self, user_financial_profile):
        super(AmountPolarity, self).__init__(user_financial_profile)
    def update(self, transactions, all_transactions, today):
        for transaction in transactions:
            if transaction.merchant_type == "payment":
                transaction.amount = -abs(transaction.amount)
            else:
                transaction.amount = round(abs(transaction.amount+np.random.rand()), 2)
    
class CreditBalance(Rule):
    def __init__(self, user_financial_profile):
        super(CreditBalance, self).__init__(user_financial_profile)
        self.credit_balance = 0
    def update(self, transactions, all_transactions, today):
        for transaction in transactions:
            self.credit_balance += transaction.amount
        if self.credit_balance < -50:
            raise Exception("Bad generation.")
    def prompt_text(self, **kwargs):
        return f"Your running balance: {round(self.credit_balance, 2)}."
        
class CreditAvailability(Rule):
    def __init__(self, user_financial_profile):
        super(CreditAvailability, self).__init__(user_financial_profile)
        self.credit_avail = self.user_financial_profile.credit_limit
        self.charge_off_count = 0
    def update(self, transactions, all_transactions, today):
        for transaction in transactions:
            self.credit_avail -= transaction.amount
        if self.credit_avail < 0:
            transactions.insert(0, Transaction(
                timestamp=today.strftime(FORMAT_CODE),
                merchant_name="charge-off-warning",
                merchant_type="charge-off-warning",
                card_present_or_not="charge-off-warning",
                amount=0
            ))
    def prompt_text(self, **kwargs):
        if self.credit_avail < 0:
            self.charge_off_count += 1
            payment_to_make = abs(self.credit_avail) + self.user_financial_profile.credit_limit
            payment_to_make = round(payment_to_make, 2)
            return f"WARNING: You spend over your credit limit. Please submit a payment of ${payment_to_make}. This is the {self.charge_off_count} time you trigger the warning. If you trigger the warning three times, your account will be locked and you will be prosecuted."
        else:
            return f"Your available credit: {round(self.credit_avail, 2)}."
    
class Subscriptions(Rule):
    def __init__(self, user_financial_profile):
        super(Subscriptions, self).__init__(user_financial_profile)
        self.recurring_transactions = self.plan_recurring_transactions()
        self.charged_subscriptions = []
    def update(self, transactions, all_transactions, today):
        for transaction in self.recurring_transactions:
            if today == datetime.strptime(transaction.timestamp, FORMAT_CODE):
                transactions.insert(0, transaction)
                self.charged_subscriptions.append(transaction)
    def prompt_text(self, **kwargs):
        return None
        charged_subscriptions = [(t.merchant_name, t.amount) for t in self.charged_subscriptions]
        if charged_subscriptions:
            self.charged_subscriptions = []
            return f"Notification: You are charged the following subscriptions today: {str(charged_subscriptions)}."
                
    def plan_recurring_transactions(self):
        def get_charge_dates(
            charge_every_n_months: int,
            day_of_month_to_charge: int,
            starting_timestamp: str = "2023-01-01 00:00:00",
            ending_timestamp: str = "2029-12-31 23:59:59",
        ) -> List[datetime]:
            start_date = datetime.strptime(starting_timestamp, FORMAT_CODE)
            end_date = datetime.strptime(ending_timestamp, FORMAT_CODE)
            charge_dates = rrule(
                freq=MONTHLY,
                dtstart=start_date,
                until=end_date,
                interval=charge_every_n_months,
                bymonthday=[day_of_month_to_charge]
            )
            return list(charge_dates)
        
        subscriptions = self.user_financial_profile.subscriptions
        for i in range(len(subscriptions)): subscriptions[i]["std"] = 0
        billing = self.user_financial_profile.recurring_variable_bills
        recurring_transactions = []
        for item in subscriptions+billing:
            charge_date = get_charge_dates(
                item["charge_frequency_month"], 
                item["date_to_charge"])
            for date in charge_date:
                recurring_transactions.append(Transaction(
                    # relevant_to="Automatic charge of subscription.",
                    timestamp= date.strftime(FORMAT_CODE),
                    merchant_name=item["merchant_name"],
                    merchant_type=item["product_description"],
                    card_present_or_not= "no",
                    amount= round(item["amount"] + np.random.normal()*item["std"], 2)
                ))
        return recurring_transactions
    
class PaymentAndInterest(Rule):
    def __init__(self, user_financial_profile):
        self.payment_date = np.random.randint(1, 26)
        self.interest_charge_date = self.payment_date + 2  # interest_charge_date is two days later than the payment due date, to give user some buffer to pay.
        self.interest_rate = 0.05
        self.payment_due, self.interest_due = 0, 0
        self.autopay = user_financial_profile.payment_habit == "automatic_payment"
    def update(self, transactions, all_transactions, today):
        if today.day == self.payment_date:
            self.payment_due = self.compute_payment_due_n_days_ago(all_transactions, today, n=15)
            if self.payment_due > 0.001 and self.autopay:
                transactions.insert(0, Transaction(
                    # relevant_to="auto payment posted by the system.",
                    timestamp=today.strftime(FORMAT_CODE),
                    merchant_name="payment thank you",
                    merchant_type="payment",
                    card_present_or_not="no",
                    amount=-round(self.payment_due, 2)
                ))
        if today.day == self.interest_charge_date:
            late_payment_due = self.compute_payment_due_n_days_ago(all_transactions, today, n=17)
            self.interest_due = late_payment_due * self.interest_rate
            if self.interest_due > 0.001:
                transactions.insert(0, Transaction(
                    # relevant_to="auto interest charge by the system",
                    timestamp=today.strftime(FORMAT_CODE),
                    merchant_name="credit interest",
                    merchant_type="credit interest",
                    card_present_or_not="no",
                    amount=round(self.interest_due, 2)
                ))
    def prompt_text(self, **kwargs):
        # self.today in `prompt_text` will be one day later than self.today in `update`.
        today = kwargs.get('today')
        if today.day - 1 == self.payment_date and self.payment_due > 0.001:
            if self.autopay:
                return f"Reminder: An autopayment of ${round(self.payment_due, 2)} is posted today. Thank you! Please also pay attention to your debit card balance."
            else:
                return f"Reminder: Your payment of ${round(self.payment_due, 2)} is due today. Please submit a payment immediately or you would be charged an interest."
        if today.day - 1 == self.interest_charge_date and self.interest_due > 0.001:
            return f"Notification: You are charged an interest of ${round(self.interest_due, 2)}."
    def compute_payment_due_n_days_ago(self, all_transactions, today, n):
        # step 1: get the latest transaction before (today - ndays).
        target_date_n_days_ago = today - timedelta(days=n)
        relevant_transactions_start_index = bisect_left(
            [datetime.strptime(t.timestamp, FORMAT_CODE) for t in all_transactions], 
            target_date_n_days_ago
        )
        if relevant_transactions_start_index == 0: return 0 # if no transaction happens before target_date, then no payment is due.
        else:
            starting_running_balance = sum([t.amount for t in all_transactions[:relevant_transactions_start_index]])
        # step 2: compute the cumulative payment after that transaction.
        cumulative_payment_after_cutoff = 0.0
        for transaction in all_transactions[relevant_transactions_start_index:]:
            if transaction.merchant_type == "payment":
                cumulative_payment_after_cutoff += abs(transaction.amount)
        # step 3: the payment due is the running_balance - cum_payment.
        payment_due = starting_running_balance - cumulative_payment_after_cutoff
        return max(0.0, payment_due)
        
class LastTimeGas(Rule):
    def __init__(self, user_financial_profile):
        self.activate = user_financial_profile.car_ownership != "no_car"
        self.n_days_ago = 0
    def update(self, transactions, all_transactions, today):
        if self.activate:
            self.n_days_ago += 1
            for transaction in transactions:
                if "gas" in transaction.merchant_type:
                    self.n_days_ago = 0
    def prompt_text(self, **kwargs):
        if self.activate:
            n_days_ago = int(self.n_days_ago) if not math.isnan(self.n_days_ago) else "nan"
            if self.n_days_ago > 30:
                return f"You did gas refilling {n_days_ago} days ago. Your gas is running low."
            else:    
                return f"You did gas refilling {n_days_ago} days ago. A normal frequency will be 1x per 2-4 weeks."
        else:
            return None
    
class LastTimeGrocery(Rule):
    def __init__(self, user_financial_profile):
        self.n_days_ago = 0
    def update(self, transactions, all_transactions, today):
        self.n_days_ago += 1
        for transaction in transactions:
            if "grocery" in transaction.merchant_type:
                self.n_days_ago = 0
                
    def prompt_text(self, **kwargs):
        n_days_ago = int(self.n_days_ago) if not math.isnan(self.n_days_ago) else "nan"
        return f"You did grocery shopping {n_days_ago} days ago. A normal frequency will be 1x per 1-2 weeks."
    
class RandomEvents(Rule):
    def __init__(self, user_financial_profile, random_event_p):
        with open("./Generators/random_events.json", "r") as f:
            self.events = json.load(f)
        self.random_event_p = random_event_p
    def update(self, transactions, all_transactions, today):
        pass
    def prompt_text(self, **kwargs):
        if np.random.rand() < self.random_event_p:
            event = np.random.choice(self.events)["event"]
            return f"A random event occurs: {event} Please make your decision based on your user and financial persona. You must respond to this in your trajectory plan and transactions."
        else:
            return None
        
class DebitCard(Rule):
    def __init__(self, user_financial_profile):
        if user_financial_profile.income_level == "no income":
            self.salary = 1000
        elif user_financial_profile.income_level == "low income":
            self.salary = 2000
        elif user_financial_profile.income_level == "med income":
            self.salary = 4000
        elif user_financial_profile.income_level == "high income":
            self.salary = 6000
        else:
            self.salary = 16000
        self.salary_arrives_in_n_days = 1
        self.balance = 5000
    def update(self, transactions, all_transactions, today):
        self.salary_arrives_in_n_days -= 1
        if self.salary_arrives_in_n_days == 0:
            self.balance += self.salary
            self.salary_arrives_in_n_days = 14
        for transaction in transactions:
            if "payment" in transaction.merchant_name:
                self.balance -= abs(transaction.amount)
            self.balance = round(self.balance, 2)
    def prompt_text(self, **kwargs):
        if self.salary > 0:
            out = f"Your current debit card balance is ${self.balance}."
            if self.balance < 1000:
                out += f"\nYour debit card balance is running low as ${self.balance}. Remember to have enough debit card balance to pay off your credit card balance. Please reduce spending. If you cannot pay off, you will be prosecuted crime."
            return out
        
class GenerationTracker:
    def __init__(self, user_persona: UserPersona, user_financial_profile: UserFinancialProfile, cfg):
        self.cfg = cfg
        self.user_persona = user_persona
        self.user_financial_profile = user_financial_profile
        self.applied_rules = [
            Holiday           (None),
            AmountPolarity    (None),
            Subscriptions     (user_financial_profile),
            PaymentAndInterest(user_financial_profile),
            LastTimeGas       (user_financial_profile),
            LastTimeGrocery   (user_financial_profile),
            RandomEvents      (user_financial_profile, self.cfg.experiment.random_event_p),
            DebitCard         (user_financial_profile),
            CreditBalance     (user_financial_profile),
            CreditAvailability(user_financial_profile) # DebitCard, CreditBalance and CreditAvailability must be placed at last.
        ]
        # self.today = datetime.strptime("2025-01-01", "%Y-%m-%d")
        start_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
        self.today = start_date + timedelta(days=np.random.randint(0, 365))
        print(self.today)
        self.transactions = []
        self.trajectory_plans = []
        
    def get_system_prompt(self):
        from Generators.prompt import system_prompt
        user_persona = self.user_persona.asdict()
        user_persona["income_level"] = self.user_financial_profile.income_level
        user_persona["payment_habit"] = self.user_financial_profile.payment_habit
        user_persona["car_ownership"] = self.user_financial_profile.car_ownership
        user_persona["spending_patterns"] = self.user_financial_profile.spending_patterns
        return system_prompt(user_persona)
    
    def get_running_user_prompt(self):
        today_str = self.today.strftime("%Y-%m-%d")
        day_str   = self.today.strftime("%A")
        out = f"Today is {today_str}, {day_str}. Here is a summary of your account information:\n"
        for ruleobj in self.applied_rules:
            prompt_text = ruleobj.prompt_text(today=self.today)
            if prompt_text:
                out += f"* {prompt_text}\n"
        out += "Remember the 200-word trajectory plan shall balance routine activities with novel experiences, and balance work with rest. Avoid repetitive activities and vary the character's routine to make their week feel more natural. The transactions must closely align the trajectory plan."
        return out
    
    def update_history(self, user_prompt, response_text):
        try:
            trajectory_plan, transactions = self.parse_transactions(response_text)
            transactions = [Transaction(**t) for t in transactions]
            if any(["deposit" in t.merchant_name.lower() or "deposit" in t.merchant_type.lower() for t in transactions]):
                raise Exception("Very likely a 'deposit' event is generated, which will mess up the generation.")
            for t in transactions:
                datetime.strptime(str(t.timestamp), FORMAT_CODE) # validate all the timestamp has correct format.
        except:
            print("unsuccessful generation.")
            return False
            
        for ruleobj in self.applied_rules:
            ruleobj.update(transactions, self.transactions, self.today)
        self.transactions = self.transactions + transactions
        self.trajectory_plans.append({
            "date": self.today.strftime("%Y-%m-%d"),
            "user_prompt": user_prompt,
            "trajectory_plan": trajectory_plan
        })
        self.today = self.today + timedelta(days=1)
        return True
        
    def parse_transactions(self, response_text):
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            response_text = response_text[start_index:end_index+1]
        data = json.loads(response_text)
        trajectory_plan = data["trajectory_plan"]
        transactions = data["transactions"]
        return trajectory_plan.strip(), transactions
    
    def export(self, export_json: str):
        assert export_json.endswith(".json")
        out = {}
        out["user_persona"] = self.user_persona.asdict()
        out["user_financial_profile"] = self.user_financial_profile.asdict()
        out["transactions"] = [t.asdict() for t in self.transactions]
        out["trajectory_plans"] = self.trajectory_plans
        with open(export_json, "w") as f:
            json.dump(out, f, indent=4)
        