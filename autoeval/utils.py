import mikeio
from pathlib import Path

def inspect_dfs0(file: Path):
    """
    Inspects a dfs0 file and prints the index and name of each data item.

    This function reads a dfs0 file using mikeio and converts it to a pandas DataFrame.
    It then enumerates through the columns of the DataFrame, printing the item index
    and name for each column.

    Args:
        file (Path): The path to the dfs0 file to be inspected.

    Returns:
        None
    """
    df = mikeio.read(file).to_dataframe()
    for index, name in [(index, name) for index, name in enumerate(df.columns)]:
        print(f"Item: {index} {name}")