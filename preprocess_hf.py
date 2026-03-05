import pandas as pd
from datasets import load_dataset
import random

def preprocess():
    print("Loading HuggingFace dataset 'Tobi-Bueck/customer-support-tickets'...")
    ds = load_dataset('Tobi-Bueck/customer-support-tickets', split='train')
    
    # Take a subset so training on CPU doesn't take days (e.g. 5000 rows)
    # The dataset has ~61k rows.
    ds = ds.shuffle(seed=42).select(range(5000))
    df = ds.to_pandas()

    print("Preprocessing data...")
    # Create ticket_text
    df["ticket_text"] = "Subject: " + df["subject"].fillna("") + "\n\n" + df["body"].fillna("")

    # Map target variables
    df["category"] = df["type"].fillna("Request")
    
    # Synthesize Urgency since dataset doesn't have it
    def map_urgency(t):
        if t == 'Incident': return 'High'
        elif t == 'Problem': return 'High'
        elif t == 'Change': return 'Medium'
        else: return 'Low' # Request
        
    df["urgency"] = df["type"].apply(map_urgency)

    # Final dataframe
    final_df = df[["ticket_text", "category", "urgency"]].dropna()

    final_df.to_csv("customer_support_tickets.csv", index=False)
    print(f"✅ Saved HF preprocessed data with {len(final_df)} rows to customer_support_tickets.csv")

if __name__ == "__main__":
    preprocess()
