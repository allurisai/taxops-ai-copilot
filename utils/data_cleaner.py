from io import StringIO
import re

import pandas as pd


CATEGORY_KEYWORDS = {
    "Travel": ["uber", "lyft", "delta", "southwest", "airbnb", "hotel"],
    "Office Supplies": ["amazon", "staples", "office depot", "best buy"],
    "Meals": ["doordash", "ubereats", "starbucks", "restaurant", "cafe"],
    "Software": ["zoom", "microsoft", "google", "quickbooks", "dropbox", "slack"],
    "Fuel": ["shell", "chevron", "exxon", "fuel", "gas station"],
    "Utilities": ["comcast", "verizon", "at&t", "electric", "internet"],
    "Professional Services": ["stripe atlas", "legal", "attorney", "consulting"],
}
VENDOR_NORMALIZATION_MAP = {
    "amazon": "Amazon",
    "amazon marketplace": "Amazon",
    "uber trip": "Uber",
    "uber": "Uber",
    "lyft": "Lyft",
    "quickbooks": "QuickBooks",
    "quickbooks online": "QuickBooks",
    "starbucks": "Starbucks",
    "doordash": "DoorDash",
    "google workspace": "Google Workspace",
    "google": "Google",
    "delta airlines": "Delta Air Lines",
}


def read_csv_file(uploaded_file):
    """Read an uploaded CSV file into a DataFrame."""
    decoded_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    return pd.read_csv(StringIO(decoded_text))


def read_csv_path(file_path):
    """Read a local CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def normalize_text_value(value):
    """Lowercase and trim text values while leaving non-text values unchanged."""
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _find_column(dataframe, keywords):
    """Find the first column whose name contains one of the given keywords."""
    for column in dataframe.columns:
        lowered_column = str(column).lower()
        if any(keyword in lowered_column for keyword in keywords):
            return column
    return None


def _normalize_vendor_name(raw_value):
    """Create a cleaner vendor label for review purposes."""
    if pd.isna(raw_value):
        return ""

    normalized_value = str(raw_value).strip().lower()
    normalized_value = re.sub(r"[^a-z0-9& ]+", " ", normalized_value)
    normalized_value = re.sub(r"\s+", " ", normalized_value).strip()

    for alias, canonical_name in VENDOR_NORMALIZATION_MAP.items():
        if alias in normalized_value:
            return canonical_name

    if not normalized_value:
        return ""

    return normalized_value.title()


def suggest_category_from_row(row):
    """Suggest a category by scanning all text in the row for known keywords."""
    row_text = " ".join(str(value) for value in row.values if pd.notna(value)).lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in row_text for keyword in keywords):
            return category

    return "Review Needed"


def _detect_unusual_amounts(dataframe):
    """Flag unusually large amounts using a simple IQR-based rule."""
    amount_column = _find_column(dataframe, ["amount", "debit", "credit", "total"])
    if amount_column is None:
        return pd.Series([False] * len(dataframe), index=dataframe.index), amount_column

    numeric_amounts = pd.to_numeric(dataframe[amount_column], errors="coerce").abs()
    valid_amounts = numeric_amounts.dropna()

    if valid_amounts.empty:
        return pd.Series([False] * len(dataframe), index=dataframe.index), amount_column

    q1 = valid_amounts.quantile(0.25)
    q3 = valid_amounts.quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + (1.5 * iqr) if iqr > 0 else valid_amounts.quantile(0.95)

    return numeric_amounts > threshold, amount_column


def process_dataframe(dataframe):
    """Clean a DataFrame, add category suggestions, and produce a summary report."""
    cleaned_df = dataframe.copy()
    cleaned_df.columns = [str(column).strip() for column in cleaned_df.columns]

    object_columns = cleaned_df.select_dtypes(include=["object"]).columns
    for column in object_columns:
        cleaned_df[column] = cleaned_df[column].apply(normalize_text_value)

    missing_value_counts = cleaned_df.isna().sum()
    duplicate_mask = cleaned_df.duplicated()

    vendor_column = _find_column(cleaned_df, ["vendor", "merchant", "payee", "description"])
    if vendor_column:
        cleaned_df["Normalized Vendor"] = cleaned_df[vendor_column].apply(_normalize_vendor_name)
    else:
        cleaned_df["Normalized Vendor"] = ""

    cleaned_df["Suggested Category"] = cleaned_df.apply(suggest_category_from_row, axis=1)
    unusual_amount_mask, amount_column = _detect_unusual_amounts(cleaned_df)

    cleaned_df["Duplicate Row"] = duplicate_mask
    cleaned_df["Unusual Amount"] = unusual_amount_mask

    review_reasons = []
    for row_index, row in cleaned_df.iterrows():
        reasons = []
        if row["Suggested Category"] == "Review Needed":
            reasons.append("Missing clear category match")
        if row["Duplicate Row"]:
            reasons.append("Possible duplicate")
        if row.isna().any():
            reasons.append("Missing values")
        if row["Unusual Amount"]:
            reasons.append("Unusual amount")
        review_reasons.append("; ".join(reasons))

    cleaned_df["Review Notes"] = review_reasons
    cleaned_df["Needs Review"] = cleaned_df["Review Notes"].ne("")

    vendor_suggestions = []
    if vendor_column:
        vendor_pairs = (
            cleaned_df[[vendor_column, "Normalized Vendor"]]
            .dropna()
            .rename(columns={vendor_column: "raw_vendor", "Normalized Vendor": "normalized_vendor"})
        )
        vendor_pairs = vendor_pairs[
            vendor_pairs["raw_vendor"].astype(str).str.strip().ne("")
            & vendor_pairs["normalized_vendor"].astype(str).str.strip().ne("")
        ]
        vendor_pairs = vendor_pairs[
            vendor_pairs["raw_vendor"].astype(str).str.casefold()
            != vendor_pairs["normalized_vendor"].astype(str).str.casefold()
        ]

        if not vendor_pairs.empty:
            vendor_suggestions = (
                vendor_pairs.value_counts()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(8)
                .to_dict("records")
            )

    report = {
        "total_rows": int(len(cleaned_df)),
        "missing_value_counts": {
            column: int(count)
            for column, count in missing_value_counts.items()
            if int(count) > 0
        },
        "duplicate_count": int(duplicate_mask.sum()),
        "review_count": int(cleaned_df["Needs Review"].sum()),
        "columns_with_missing_values": int((missing_value_counts > 0).sum()),
        "missing_category_count": int(cleaned_df["Suggested Category"].eq("Review Needed").sum()),
        "anomaly_count": int(cleaned_df["Unusual Amount"].sum()),
        "amount_column": amount_column or "Not detected",
        "vendor_suggestions": vendor_suggestions,
    }

    return cleaned_df, report


def highlight_review_rows(row):
    """Highlight review and anomaly rows in the Streamlit table."""
    if row.get("Needs Review", False):
        return ["background-color: #fff3cd"] * len(row)
    if row.get("Unusual Amount", False):
        return ["background-color: #fde2e1"] * len(row)
    return [""] * len(row)
