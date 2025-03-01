import pandas as pd
import os
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader

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

    def _create_document_text(self, row):
        """Combine the relevant columns into a single text field."""
        parts = []
        for col in self.text_columns:
            value = row.get(col, "N/A")
            if pd.notna(value) and str(value).strip():
                parts.append(f"{col}: {value}")
        return "\n".join(parts)

    def load_data(self):
        # Read CSV file (optionally limit rows)
        df = pd.read_csv(self.file_path, nrows=self.nrows)

        # Ensure each expected column exists and convert to string.
        for col in self.text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('N/A').astype(str)
            else:
                df[col] = 'N/A'

        # Create a combined text column from the relevant columns.
        df["text"] = df.apply(self._create_document_text, axis=1)

        # Write the processed DataFrame to a temporary CSV file.
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tmp:
            temp_file_path = tmp.name
            df.to_csv(temp_file_path, index=False)

        # Load documents using CSVLoader with the combined "text" column.
        loader = CSVLoader(file_path=temp_file_path, source_column="text")
        documents = loader.load()

        # Clean up temporary file.
        os.remove(temp_file_path)

        return documents
