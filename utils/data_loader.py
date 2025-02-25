import pandas as pd
# import numpy as np
from langchain_community.document_loaders import DataFrameLoader


# class OncologyDataLoader:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.text_columns = [
#             'Drug Name', 'Cancer Type', 'Number of Patients',
#             'OS_Improvement (%)', 'PFS_Improvement (%)',
#             'Other Outcome Measures', 'Brief Study Summary',
#             'Formatted Study Results'
#         ]
#
#     def load_data(self):
#         df = pd.read_csv(self.file_path, nrows=5)
#
#         # Convert relevant columns to strings
#         for col in self.text_columns:
#             if col in df.columns:
#                 df[col] = df[col].fillna('N/A').astype(str)
#             else:
#                 df[col] = 'N/A'
#
#         df["text"] = df.apply(self._create_document_text, axis=1)
#
#         loader = DataFrameLoader(df, page_content_column="text")
#         return loader.load()
#
#     def _create_document_text(self, row):
#         parts = []
#         for col in self.text_columns:
#             parts.append(f"{col}: {row.get(col, 'N/A')}")
#         return "\n".join(parts)


import csv

class OncologyDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text_columns = [
            'Drug Name', 'Cancer Type', 'Number of Patients',
            'OS_Improvement (%)', 'PFS_Improvement (%)',
            'Other Outcome Measures', 'Brief Study Summary',
            'Formatted Study Results'
        ]

    def load_data(self):
        data = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)  # Reads CSV as a list of dictionaries
            for row in reader:
                # Ensure all columns exist and fill missing ones
                for col in self.text_columns:
                    row[col] = row.get(col, "N/A")

                # Create text representation
                row["text"] = self._create_document_text(row)
                data.append(row)

        return data  # Returns a list of dictionaries

    def _create_document_text(self, row):
        return "\n".join([f"{col}: {row[col]}" for col in self.text_columns])


from langchain.docstore.document import Document

def load_csv_as_documents(file_path):
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = "\n".join([f"{k}: {v}" for k, v in row.items()])
            documents.append(Document(page_content=text))
    return documents