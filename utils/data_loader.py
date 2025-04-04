import pandas as pd
import os
import tempfile
from langchain_community.document_loaders.csv_loader import CSVLoader
from typing import List, Optional
import re
from langchain.schema import Document


class DataLoader:
    def __init__(self, trials_path: str, oncology_path: str):
        """Initialize loader for both datasets
        Args:
            trials_path (str): Path to Active Recruiting Trials CSV
            oncology_path (str): Path to Oncology Survival dataset
        """
        self.trials_path = trials_path
        self.oncology_path = oncology_path

    def load_trials_data(self,
                         location_filter: Optional[str] = None,
                         condition_filter: Optional[str] = None) -> List[Document]:
        """Load and filter clinical trials data"""
        try:
            df = pd.read_csv(self.trials_path, low_memory=False)

            # Filter for actively recruiting trials
            df = df[df['Study Status'].str.contains('Recruiting', case=False, na=False)]

            if location_filter:
                df = df[df['Locations'].str.contains(location_filter, case=False, na=False)]

            if condition_filter:
                condition_pattern = f"({condition_filter}|solid tumor|malignancy)"
                df = df[df['Conditions'].str.contains(condition_pattern, case=False, na=False)]

            documents = []
            for _, row in df.iterrows():
                content = self._format_trial_info(row)
                documents.append(Document(
                    page_content=content,
                    metadata={
                        'source': 'clinical_trial',
                        'trial_id': row['NCT Number'],
                        'location': row['Locations'],
                        'condition': row['Conditions']
                    }
                ))

            return documents
        except Exception as e:
            print(f"Error loading trials data: {str(e)}")
            return []

    def load_oncology_data(self, cancer_type: Optional[str] = None) -> List[Document]:
        """Load and filter oncology survival data"""
        try:
            df = pd.read_csv(self.oncology_path)

            if cancer_type:
                df = df[df['Cancer Type'].str.contains(cancer_type, case=False, na=False)]

            documents = []
            for _, row in df.iterrows():
                content = self._format_oncology_info(row)
                documents.append(Document(
                    page_content=content,
                    metadata={
                        'source': 'oncology_data',
                        'cancer_type': row['Cancer Type'],
                        'drug_name': row['Drug Name']
                    }
                ))

            return documents
        except Exception as e:
            print(f"Error loading oncology data: {str(e)}")
            return []

    def _format_trial_info(self, row: pd.Series) -> str:
        """Format clinical trial information"""
        return f"""
Trial ID: {row['NCT Number']}
Title: {row['Study Title']}
Condition: {row['Conditions']}
Phase: {row['Phases']}
Location: {row['Locations']}
Status: {row['Study Status']}
Primary Outcome: {row['Primary Outcome Measures']}
Brief Summary: {row['Brief Summary']}
Eligibility:
- Age: {row['Age']}
- Sex: {row['Sex']}
Intervention Type: {row['Interventions']}
"""

    def _format_oncology_info(self, row: pd.Series) -> str:
        """Format oncology survival data"""
        return f"""
Drug: {row['Drug Name']}
Cancer Type: {row['Cancer Type']}
Overall Survival: {row['Overall Survival']}
FDA Approval: {row['FDA Approval']}
Treatment Line: {row['Treatment Line']}
Key Outcomes: {row['Key Outcomes']}
"""

    def load_combined_data(self,
                           cancer_type: Optional[str] = None,
                           location_filter: Optional[str] = None) -> List[Document]:
        """Load and combine both datasets"""
        trials_docs = self.load_trials_data(
            location_filter=location_filter,
            condition_filter=cancer_type
        )
        oncology_docs = self.load_oncology_data(cancer_type=cancer_type)
        return trials_docs + oncology_docs