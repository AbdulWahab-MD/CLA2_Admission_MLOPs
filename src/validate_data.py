import pandas as pd


def run():
    # Load the Admission dataset
    df = pd.read_csv("data/raw/admission.csv")

    # Check for missing values
    assert df.isnull().sum().sum() == 0, "Null values detected"

    # Admission dataset contains 9 columns
    assert len(df.columns) >= 9, "Missing columns"

    # Check required columns
    required_columns = [
        "Serial No.",
        "GRE Score",
        "TOEFL Score",
        "University Rating",
        "SOP",
        "LOR ",
        "CGPA",
        "Research",
        "Chance of Admit "
    ]

    for column in required_columns:
        assert column in df.columns, f"Missing column: {column}"

    print("[SUCCESS] Schema Validation PASSED")


if __name__ == "__main__":
    run()
