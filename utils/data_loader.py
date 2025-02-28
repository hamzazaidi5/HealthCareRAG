import pandas as pd
from langchain_community.document_loaders import DataFrameLoader


class OncologyDataLoader:
    def __init__(self, file_path, nrows=None):
        """
        :param file_path: Path to the CSV file.
        :param nrows: Optionally limit the number of rows to load.
        """
        self.file_path = file_path
        self.nrows = nrows  # Use this to limit rows if needed
        self.text_columns = [
            'Drug Name', 'Cancer Type', 'Number of Patients',
            'OS_Improvement (%)', 'PFS_Improvement (%)',
            'Other Outcome Measures', 'Brief Study Summary',
            'Formatted Study Results'
        ]

    def load_data(self):
        # Read the entire CSV or limited rows if nrows is provided.
        df = pd.read_csv(self.file_path, nrows=self.nrows)

        # Ensure each expected column exists and convert values to string.
        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('N/A').astype(str)
            else:
                # If a column is missing, create it with a default value.
                df[col] = 'N/A'

        # Create a combined text field from the relevant columns.
        df["text"] = df.apply(self._create_document_text, axis=1)

        # Load documents using the DataFrameLoader with the "text" column.
        loader = DataFrameLoader(df, page_content_column="text")
        return loader.load()

    def _create_document_text(self, row):
        parts = []
        for col in self.text_columns:
            parts.append(f"{col}: {row.get(col, 'N/A')}")
        return "\n".join(parts)
