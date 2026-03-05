import pandas as pd

def preprocess():
    print("Loading Kaggle dataset...")
    df = pd.read_csv('DATASETS/customer_support_tickets.csv')

    # Create ticket_text from Subject and Description
    df["ticket_text"] = "Subject: " + df["Ticket Subject"].fillna("") + "\n\n" + df["Ticket Description"].fillna("")

    # Map target variables
    df["category"] = df["Ticket Type"]
    df["urgency"] = df["Ticket Priority"]

    # Select only required columns and drop rows with missing targets
    final_df = df[["ticket_text", "category", "urgency"]].dropna()

    final_df.to_csv("customer_support_tickets.csv", index=False)
    print(f"✅ Saved preprocessed data with {len(final_df)} rows to customer_support_tickets.csv")

if __name__ == "__main__":
    preprocess()
