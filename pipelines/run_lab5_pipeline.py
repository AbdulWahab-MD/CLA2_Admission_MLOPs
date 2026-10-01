import pandas as pd

def run():
    df = pd.read_csv("data/raw/admission.csv")

    df.columns = df.columns.str.strip()

    required_columns = [
        "Serial No.",
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR",
        "CGPA",
        "Research",
        "Chance of Admit"
    ]

    for column in required_columns:
        assert column in df.columns, f"Missing column: {column}"

    print("[SUCCESS] Admission dataset validation completed.")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Columns: {len(df.columns)}")

if __name__ == "__main__":
    run()