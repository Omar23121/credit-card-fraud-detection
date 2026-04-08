import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load dataset
    df = pd.read_csv("creditcard.csv")

    print("Dataset loaded!\n")

    # -----------------------------
    # 1. Class distribution
    # -----------------------------
    class_counts = df["Class"].value_counts()

    print("Class distribution:")
    print(class_counts)

    # Plot class distribution
    plt.figure()
    class_counts.plot(kind='bar')
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.xlabel("Class (0 = Normal, 1 = Fraud)")
    plt.ylabel("Count")
    plt.show()

    # -----------------------------
    # 2. Transaction Amount Analysis
    # -----------------------------
    plt.figure()
    df["Amount"].hist(bins=50)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.show()

    # -----------------------------
    # 3. Compare Fraud vs Non-Fraud Amounts
    # -----------------------------
    fraud = df[df["Class"] == 1]
    normal = df[df["Class"] == 0]

    print("\nAverage transaction amount:")
    print("Fraud:", fraud["Amount"].mean())
    print("Normal:", normal["Amount"].mean())

    # Boxplot comparison
    plt.figure()
    plt.boxplot([normal["Amount"], fraud["Amount"]], tick_labels=["Normal", "Fraud"])
    plt.title("Transaction Amount Comparison")
    plt.ylabel("Amount")
    plt.show()


if __name__ == "__main__":
    main()