import csv
import os
from typing import List, Dict
import json


class TrademarkCSVManager:
    def __init__(self, csv_file_path: str = "existing_trademarks.csv"):
        self.csv_file_path = csv_file_path
        self.fieldnames = [
            "Client / Applicant", "Application No.", "Trademark", 
            "Logo", "Class", "Status", "Validity"
        ]
        # Mapping for backward compatibility with extraction system
        self.field_mapping = {
            "name": "Client / Applicant",
            "text_in_logo": "Trademark", 
            "registration_number": "Application No.",
            "business_category": "Class",
            "legal_status": "Status"
        }
        
    def convert_extracted_to_csv_format(self, extracted_trademark: Dict) -> Dict:
        """Convert extracted trademark data to CSV format using field mapping"""
        csv_format = {}
        
        # Map extracted fields to CSV columns
        for extracted_field, csv_field in self.field_mapping.items():
            if extracted_field in extracted_trademark:
                csv_format[csv_field] = extracted_trademark[extracted_field]
        
        # Set remaining fields to empty if not mapped
        for field in self.fieldnames:
            if field not in csv_format:
                csv_format[field] = ""
                
        return csv_format
        
    def save_trademarks_to_csv(self, trademarks_data: List[Dict]):
        """Save trademark data to CSV file"""
        try:
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()
                
                for trademark in trademarks_data:
                    # Only write fields that exist in fieldnames
                    filtered_trademark = {key: trademark.get(key, '') for key in self.fieldnames}
                    writer.writerow(filtered_trademark)
                    
            print(f"Saved {len(trademarks_data)} trademarks to {self.csv_file_path}")
            
        except Exception as e:
            print(f"Error saving to CSV: {str(e)}")
            
    def load_trademarks_from_csv(self) -> List[Dict]:
        """Load trademark data from CSV file"""
        try:
            with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                trademarks = list(reader)
            print(f"Loaded {len(trademarks)} trademarks from {self.csv_file_path}")
            return trademarks
            
        except FileNotFoundError:
            print(f"CSV file {self.csv_file_path} not found.")
            return []
        except Exception as e:
            print(f"Error loading from CSV: {str(e)}")
            return []
            
    def append_trademark_to_csv(self, trademark_data: Dict):
        """Append a single trademark to the CSV file"""
        try:
            file_exists = os.path.exists(self.csv_file_path)
            
            with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Only write fields that exist in fieldnames
                filtered_trademark = {key: trademark_data.get(key, '') for key in self.fieldnames}
                writer.writerow(filtered_trademark)
                
            print(f"Appended trademark to {self.csv_file_path}")
            
        except Exception as e:
            print(f"Error appending to CSV: {str(e)}")
            
    def update_csv_with_new_extractions(self, new_trademarks: List[Dict]):
        """Update CSV file with newly extracted trademarks"""
        existing_trademarks = self.load_trademarks_from_csv()
        
        # Combine existing and new trademarks
        all_trademarks = existing_trademarks + new_trademarks
        
        # Save all to CSV
        self.save_trademarks_to_csv(all_trademarks)
        
        print(f"Updated CSV with {len(new_trademarks)} new trademarks. Total: {len(all_trademarks)}")
        
    def export_to_json(self, output_file: str = "trademarks_export.json"):
        """Export CSV data to JSON format"""
        trademarks = self.load_trademarks_from_csv()
        
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(trademarks, file, indent=2, ensure_ascii=False)
            print(f"Exported {len(trademarks)} trademarks to {output_file}")
            
        except Exception as e:
            print(f"Error exporting to JSON: {str(e)}")
            
    def get_csv_stats(self) -> Dict:
        """Get statistics about the CSV data"""
        trademarks = self.load_trademarks_from_csv()
        
        if not trademarks:
            return {"total_trademarks": 0}
            
        stats = {
            "total_trademarks": len(trademarks),
            "unique_applicants": len(set(t.get("Client / Applicant", "") for t in trademarks if t.get("Client / Applicant"))),
            "classes": {},
            "statuses": {}
        }
        
        # Count classes and statuses
        for trademark in trademarks:
            class_type = trademark.get("Class", "Unknown")
            stats["classes"][class_type] = stats["classes"].get(class_type, 0) + 1
            
            status = trademark.get("Status", "Unknown")
            stats["statuses"][status] = stats["statuses"].get(status, 0) + 1
            
        return stats 