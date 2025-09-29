import pandas as pd

def convert_dataframe_to_string(dataframe: pd.DataFrame, string_format: str = "dataframe") -> str:
    """
    Converts a pandas DataFrame to a full, untruncated string.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to serialize.
    string_format : {"dataframe", "csv", "json", "markdown"}
        - "dataframe": Pretty-printed table via DataFrame.to_string with display options set to avoid truncation.
        - "csv": CSV via DataFrame.to_csv(index=False).
        - "json": JSON via DataFrame.to_json(orient="records", date_format="iso").
        - "markdown": Markdown table via DataFrame.to_markdown(index=False). Requires 'tabulate' package.

    Notes
    -----
    This function does not coerce or alter dtypes. Timestamps keep their original precision and timezone.
    """
    # Normalize format argument
    fmt = (string_format or "").strip().lower()

    # Use the dataframe as-is to avoid any implicit coercion or precision loss
    df = dataframe

    if fmt == "csv":
        # Return a comma-separated CSV string without the index
        # Pandas will preserve timezone-aware timestamps in ISO format with offsets
        return df.to_csv(index=False)

    if fmt == "json":
        # JSON string with records orientation; ISO dates for readability/interop
        return df.to_json(orient="records", date_format="iso")

    if fmt == "markdown":
        # Markdown table; requires optional dependency 'tabulate'
        # Always return string
        return df.to_markdown(index=False)

    if fmt == "dataframe" or fmt == "":
        # Convert entire dataframe to string with full display (no truncation)
        with pd.option_context(
            'display.max_rows', None,        # Show all rows
            'display.max_columns', None,     # Show all columns
            'display.width', None,           # Unlimited width
            'display.max_colwidth', None,    # Show full content of each cell
            'display.expand_frame_repr', True # Use all space available
        ):
            return df.to_string(index=False)

    raise ValueError(f"Unsupported string_format '{string_format}'. Supported: 'dataframe', 'csv', 'json', 'markdown'.")


# uv run python -m src.core.utils.convert_data_for_ai
if __name__ == "__main__":
    # Write a 250 row dataframe with 5 columns (one being a datetime column)
    df = pd.DataFrame({
        "col1": range(250),
        "col2": pd.date_range("2021-01-01", periods=250, tz="UTC"),
        "col3": range(250),
        "col4": range(250),
        "col5": pd.date_range("2021-01-01", periods=250, tz="UTC")
    })

    print("original dataframe")
    print(df)
    print("-"*100)


    print("\n\ndataframe string")
    df_string = convert_dataframe_to_string(df, "dataframe")
    print(df_string)
    print(f"type: {type(df_string)}")
    print("-"*100)

    print("\n\ncsv string")
    csv_string = convert_dataframe_to_string(df, "csv")
    print(csv_string)
    print(f"type: {type(csv_string)}")
    print("-"*100)

    print("\n\njson string")
    json_string = convert_dataframe_to_string(df, "json")
    print(json_string)
    print(f"type: {type(json_string)}")
    print("-"*100)

    print("\n\nmarkdown string")
    md_string = convert_dataframe_to_string(df, "markdown")
    print(md_string)
    print(f"type: {type(md_string)}")
    print("-"*100)