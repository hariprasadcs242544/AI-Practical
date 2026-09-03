import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

file_path = r"F:\Downloads\Most Runs - 2010.csv"
df = pd.read_csv(file_path)
print("Hariprasad Vishwakarma T127")
print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nColumn names:")
print(df.columns.tolist())

if "Player" in df.columns:
    df = df.drop("Player", axis=1)

if "POS" in df.columns:
    df = df.drop("POS", axis=1)

if "HS" in df.columns:
    df["HS"] = df["HS"].astype(str).str.replace("*", "", regex=False)
    df["HS"] = pd.to_numeric(df["HS"], errors="coerce")

for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna()

print("\nMissing values removed")

transactions = []

for _, row in df.iterrows():
    transaction = []

    if row["Runs"] >= 400:
        transaction.append("High Runs")
    elif row["Runs"] >= 250:
        transaction.append("Medium Runs")
    else:
        transaction.append("Low Runs")

    if row["Avg"] >= 40:
        transaction.append("High Average")
    elif row["Avg"] >= 25:
        transaction.append("Medium Average")
    else:
        transaction.append("Low Average")

    if row["SR"] >= 140:
        transaction.append("High Strike Rate")
    elif row["SR"] >= 120:
        transaction.append("Medium Strike Rate")
    else:
        transaction.append("Low Strike Rate")

    if row["100"] > 0:
        transaction.append("Century")
    if row["50"] > 0:
        transaction.append("Fifty")
    if row["4s"] >= 30:
        transaction.append("Many Fours")
    if row["6s"] >= 15:
        transaction.append("Many Sixes")

    transactions.append(transaction)

print("\nNumber of transactions:", len(transactions))

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

basket = pd.DataFrame(
    te_array,
    columns=te.columns_
)

print("\nTransaction Matrix:")
print(basket.head())

frequent_itemsets = apriori(
    basket,
    min_support=0.10,
    use_colnames=True
)

frequent_itemsets["itemsets_length"] = frequent_itemsets["itemsets"].apply(len)

print("\nFrequent Itemsets:")
print(
    frequent_itemsets.sort_values(
        by="support",
        ascending=False
    ).head(20)
)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.30
)

print("\nAssociation Rules:")
print(rules[[
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift"
]].head(20).to_string(index=False))

rules = rules.sort_values(
    by="lift",
    ascending=False
)

strong_rules = rules[
    (rules["confidence"] >= 0.50) &
    (rules["lift"] > 1)
]

print("\nStrong Association Rules:")
print(
    strong_rules[[
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]].head(20).to_string(index=False)
)

top_itemsets = frequent_itemsets.sort_values(
    by="support",
    ascending=False
).head(10)

itemset_labels = [
    ", ".join(list(itemset))
    for itemset in top_itemsets["itemsets"]
]

plt.figure(figsize=(10, 6))

plt.barh(
    itemset_labels[::-1],
    top_itemsets["support"].values[::-1]
)

plt.xlabel("Support")
plt.ylabel("Itemset")
plt.title("Top 10 Frequent Itemsets")

plt.tight_layout()

top_rules = rules.head(10).copy()

rule_labels = [
    ", ".join(list(row["antecedents"])) +
    " -> " +
    ", ".join(list(row["consequents"]))
    for _, row in top_rules.iterrows()
]

plt.figure(figsize=(10, 6))

plt.barh(
    rule_labels[::-1],
    top_rules["lift"].values[::-1]
)

plt.xlabel("Lift")
plt.ylabel("Association Rule")
plt.title("Top 10 Association Rules by Lift")

plt.tight_layout()

print("\n==============================")
print("ASSOCIATION RULE MINING SUMMARY")
print("==============================")
print("Total transactions:", len(transactions))
print("Number of frequent itemsets:", len(frequent_itemsets))
print("Number of association rules:", len(rules))
print("Number of strong rules:", len(strong_rules))
print("==============================")

plt.show()
