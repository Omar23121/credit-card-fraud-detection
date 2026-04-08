import pandas as pd

def main():
    try:
        # Load dataset (make sure creditcard.csv is in the SAME folder)
        df = pd.read_csv("creditcard.csv")

        print("✅ Dataset loaded successfully!\n")

        # Show first 5 rows
        print("🔹 First 5 rows:")
        print(df.head())

        # Dataset shape
        print("\n🔹 Dataset shape (rows, columns):")
        print(df.shape)

        # Column names
        print("\n🔹 Column names:")
        print(df.columns.tolist())

        # Check missing values
        print("\n🔹 Missing values per column:")
        print(df.isnull().sum())

        # Class distribution
        print("\n🔹 Class distribution (0 = normal, 1 = fraud):")
        print(df["Class"].value_counts())

        # Percentage distribution
        print("\n🔹 Class distribution (%):")
        print(df["Class"].value_counts(normalize=True) * 100)

    except FileNotFoundError:
        print("❌ ERROR: 'creditcard.csv' not found.")
        print("👉 Make sure the dataset is in the SAME folder as load_data.py")

if __name__ == "__main__":
    main()