import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class PostProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        batch_list = os.listdir(self.cfg.data.export_path)
        global_idx = 0
        liquid_personas, liquid_transactions = [], []
        illiquid_personas, illiquid_transactions = [], []
        for path in batch_list:
            path = os.path.join(self.cfg.data.export_path, path)
            files = os.listdir(path)
            files = [file for file in files if file.endswith("json") and "conversation" not in file]
            for file in tqdm(files):
                json_file = os.path.join(path, file)
                illiquid, persona, transactions = self.process_one_json(json_file)
                if illiquid == -1:
                    continue
                elif illiquid == 0:
                    transactions["user_id"] = global_idx
                    persona["user_id"] = global_idx
                    liquid_personas.append(persona)
                    liquid_transactions.append(transactions)
                    global_idx += 1
                elif illiquid == 1:
                    transactions["user_id"] = global_idx
                    persona["user_id"] = global_idx
                    illiquid_personas.append(persona)
                    illiquid_transactions.append(transactions)
                    global_idx += 1
        
        illiquid_personas = pd.DataFrame(illiquid_personas).set_index("user_id")
        liquid_personas = pd.DataFrame(liquid_personas).set_index("user_id")
        illiquid_transactions = pd.concat(illiquid_transactions, axis=0)
        liquid_transactions = pd.concat(liquid_transactions, axis=0)
        
        illiquid_personas.to_parquet(os.path.join(self.cfg.data.export_path, "raw_generation", "illiquid", "personas.parquet"))
        liquid_personas.to_parquet(os.path.join(self.cfg.data.export_path, "raw_generation", "normal", "personas.parquet"))
        illiquid_transactions.to_parquet(os.path.join(self.cfg.data.export_path, "raw_generation", "illiquid", "transactions.parquet"))
        liquid_transactions.to_parquet(os.path.join(self.cfg.data.export_path, "raw_generation", "normal", "transactions.parquet"))
        
    def process_one_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        user_persona = data["user_persona"]
        user_financial_profile = data["user_financial_profile"]
        transactions = data["transactions"]
        out_persona = {
            **user_persona,
            **{k: user_financial_profile[k] for k in (
                "income_level", "credit_limit", "payment_habit", "car_ownership", "spending_patterns"
            )}
        }
        if transactions[-1]["timestamp"] == "charge-off":
            illiquid = True
            transactions = transactions[:-1]
        else:
            illiquid = False
            
        transactions = pd.DataFrame(transactions)
        transactions["card_present_or_not"] = transactions["card_present_or_not"].map(
            lambda x: "unknown" if x not in ["yes", "no"] else x
        )
        try:
            transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
        except:
            illiquid = -1
            
        timespan = transactions["timestamp"].iloc[-1] - transactions["timestamp"].iloc[0]
        if timespan.days < 90:
            illiquid = -1
            
        return illiquid, out_persona, transactions

def simulate_fraudulent_day(user_transactions_df, fraud_pool_df):
    """
    Randomly replaces one day of transactions in a user's history with one day of 
    transactions from a fraudulent pool.

    Args:
        user_transactions_df (pd.DataFrame): The legitimate user's transaction history.
        fraud_pool_df (pd.DataFrame): A pool of transactions to draw fraudulent activity from.

    Returns:
        pd.DataFrame: The user's transaction history with one day replaced,
                      or the original DataFrame if no valid replacement could be made.
    """
    # --- 1. Data Preparation ---
    # Ensure timestamp columns are in datetime format for proper manipulation.
    # Using .copy() to avoid SettingWithCopyWarning later.
    df1 = user_transactions_df.copy()
    df2 = fraud_pool_df.copy()
    df1['timestamp'] = pd.to_datetime(df1['timestamp'])
    df2['timestamp'] = pd.to_datetime(df2['timestamp'])

    # --- 2. Random Day Selection ---
    # Get unique dates from both dataframes to choose from.
    df1_dates = df1['timestamp'].dt.date.unique()
    df2_dates = df2['timestamp'].dt.date.unique()

    if len(df1_dates) == 0 or len(df2_dates) == 0:
        print("Warning: One of the dataframes has no dates to sample from.")
        return user_transactions_df

    target_date = np.random.choice(df1_dates)
    source_date = np.random.choice(df2_dates)
    
    # --- 3. Isolate and Filter Transactions ---
    # Get all transactions for the chosen day from the fraud pool.
    fraud_day_transactions = df2[df2['timestamp'].dt.date == source_date].copy()
    
    # Per instructions, remove any payment events (amount < 0).
    fraud_day_transactions = fraud_day_transactions[fraud_day_transactions['amount'] >= 0].copy()

    # If the chosen day from the fraud pool has no positive transactions, abort.
    if fraud_day_transactions.empty:
        return user_transactions_df

    # --- 4. Align Timestamps and Data Structure ---
    # Get the original transactions to find the seq_id to use.
    original_day_transactions = df1[df1['timestamp'].dt.date == target_date]
    seq_id_to_use = original_day_transactions['seq_id'].iloc[0]
    
    # Create new timestamps for the fraudulent transactions by combining the target date
    # with the original time of day from the source transactions.
    new_timestamps = fraud_day_transactions['timestamp'].apply(
        lambda ts: ts.replace(year=target_date.year, month=target_date.month, day=target_date.day)
    )
    
    # Assign the new, consistent timestamps.
    fraud_day_transactions['timestamp'] = new_timestamps
    
    # Add the 'seq_id' column to match the structure of the original dataframe.
    fraud_day_transactions['seq_id'] = int(seq_id_to_use)

    # --- 5. Perform the Replacement ---
    # Remove the original day's transactions from the user's history.
    df1_without_target_day = df1[df1['timestamp'].dt.date != target_date]

    # Concatenate the remaining history with the new fraudulent day's transactions.
    # Ensure the columns align perfectly.
    final_df = pd.concat([df1_without_target_day, fraud_day_transactions[df1.columns]], ignore_index=True)
    
    # Sort the final dataframe by timestamp to maintain chronological order.
    final_df = final_df.sort_values(by='timestamp').reset_index(drop=True)

    return final_df

class TaskCreation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.path = os.path.join(self.cfg.data.export_path, "raw_generation")
        self.all_normal_transactions = pd.read_parquet(os.path.join(self.path, "normal/transactions.parquet"))
        self.all_insolvency_transactions = pd.read_parquet(os.path.join(self.path, "illiquid/transactions.parquet"))
        self.normal_personas = pd.read_parquet(os.path.join(self.path, "normal/personas.parquet"))
        self.insolvent_personas = pd.read_parquet(os.path.join(self.path, "illiquid/personas.parquet"))
        self.normal_user_ids = self.all_normal_transactions["user_id"].unique()
        self.insolvent_user_ids = self.all_insolvency_transactions["user_id"].unique()
        self.largest_user_id = max(self.normal_user_ids.max(), self.insolvent_user_ids.max())
        self.month_window = self.cfg.experiment.month_window
        self.train_ratio = self.cfg.experiment.train_ratio
        
        if self.cfg.experiment.what_task == "identity_theft":
            self.identity_theft()
        elif self.cfg.experiment.what_task == "illiquidity_classification":
            self.insolvency_prediction()
        else:
            raise Exception(f"Wrong task {self.cfg.experiment.what_task} is inputted, choose from identity_theft and illiquidity_classification.")

    def insolvency_prediction(self):
        train_df, test_df, labels = [], [], {}
        global_seq_id_offset = 0
        for user_id in tqdm(range(self.largest_user_id)):
            transactions_df = self.get_transactions_by_user_id(user_id)
            transactions_df = self.label_seq_id_by_month(transactions_df, remove_last_seq_id=True)
            train = np.random.uniform() < self.train_ratio
            if transactions_df is not None:
                transactions_df["seq_id"] = transactions_df["seq_id"] + global_seq_id_offset
                global_seq_id_offset = transactions_df["seq_id"].max() + 1
                if train:
                    train_df.append(transactions_df)
                else:
                    test_df.append(transactions_df)
                label = str(user_id in self.insolvent_user_ids)
                labels.update({idx: label for idx in transactions_df["seq_id"]})
        savepath = os.path.join(self.path, f"illiquidity_prediction_{self.month_window}months")
        if not os.path.exists(savepath):
            os.system(f"mkdir -p {savepath}")
        self.save(train_df, test_df, labels, savepath)

    def identity_theft(self):
        train_df, test_df, labels = [], [], None
        global_seq_id_offset = 0
        for user_id in tqdm(range(self.largest_user_id)):
            transactions_df = self.get_transactions_by_user_id(user_id)
            transactions_df = self.label_seq_id_by_month(transactions_df, remove_last_seq_id=False)
            train = np.random.uniform() < self.train_ratio
            for seq_id in range(transactions_df["seq_id"].max()):
                cur_df = transactions_df[transactions_df["seq_id"] == seq_id]
                random_insolvent_user_id = np.random.choice(self.insolvent_user_ids)
                fake_df = self.get_transactions_by_user_id(random_insolvent_user_id)
                cur_df["is_fraud"] = False
                fake_df["is_fraud"] = True
                final_df = simulate_fraudulent_day(cur_df, fake_df)
                final_df["seq_id"] = global_seq_id_offset
                global_seq_id_offset += 1
                if train:
                    train_df.append(final_df)
                else:
                    test_df.append(final_df)
        savepath = os.path.join(self.path, f"identity_theft_{self.month_window}months")
        if not os.path.exists(savepath):
            os.system(f"mkdir -p {savepath}")
        self.save(train_df, test_df, labels, savepath)

    def save(self, train_df, test_df, labels, savepath):
        train_df = pd.concat(train_df, ignore_index=True)
        test_df = pd.concat(test_df, ignore_index=True)
        train_df = train_df.drop(["user_id"], axis=1)
        test_df = test_df.drop(["user_id"], axis=1)
        train_df.to_parquet(f"{savepath}/train.parquet")
        test_df.to_parquet(f"{savepath}/test.parquet")
        if labels is not None:
            with open(f"{savepath}/labels.json", "w") as f:
                json.dump(labels, f, indent=4)

    def get_transactions_by_user_id(self, user_id):
        if user_id in self.normal_user_ids:
            transactions = self.all_normal_transactions[self.all_normal_transactions['user_id'] == user_id]
        elif user_id in self.insolvent_user_ids:
            transactions = self.all_insolvency_transactions[self.all_insolvency_transactions['user_id'] == user_id]
        else:
            transactions = None
        return transactions
    
    def get_user_persona_by_user_id(self, user_id):
        if user_id < self.normal_user_id_cutoff:
            user_persona = self.normal_personas.loc[user_id]
        else:
            user_persona = self.insolvent_personas.loc[user_id]
        return user_persona
    
    def label_seq_id_by_month(self, transactions_df, remove_last_seq_id=True):
        transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
        min_date = transactions_df['timestamp'].min()
        transactions_df['seq_id'] = (
            (transactions_df['timestamp'].dt.year - min_date.year) * 12 +
            (transactions_df['timestamp'].dt.month - min_date.month)
        ) // self.month_window
        if remove_last_seq_id:
            last_idx = transactions_df['seq_id'].max()
            if last_idx > 0:
                transactions_df = transactions_df[transactions_df['seq_id'] < last_idx]
            else:
                transactions_df = None
        return transactions_df