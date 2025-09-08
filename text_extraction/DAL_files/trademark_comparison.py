import re
import csv
import json
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import jellyfish  # For Soundex, Metaphone, and Levenshtein distance
from fuzzywuzzy import fuzz, process

class TrademarkComparator:
    def __init__(self):
        self.existing_trademarks = []
    
    def load_existing_trademarks_from_csv(self, csv_file_path: str):
        """Load existing trademark data from CSV file"""
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                self.existing_trademarks = list(csv_reader)
                print(f"Loaded {len(self.existing_trademarks)} trademarks from CSV")
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            self.existing_trademarks = []
    
    def normalize_name(self, name: str) -> str:
        """Normalize name for better comparison"""
        if not name:
            return ""
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^\w\s]', ' ', name.lower())
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        return normalized
    
    def phonetic_similarity(self, name1: str, name2: str) -> Dict[str, float]:
        """Calculate phonetic similarity using Soundex and Metaphone"""
        if not name1 or not name2:
            return {
                "soundex": 0.0,
                "metaphone": 0.0,
                "avg_phonetic": 0.0
            }
        
        # Soundex comparison
        soundex1 = jellyfish.soundex(name1)
        soundex2 = jellyfish.soundex(name2)
        soundex_match = 1.0 if soundex1 == soundex2 else 0.0
        
        # Metaphone comparison
        metaphone1 = jellyfish.metaphone(name1)
        metaphone2 = jellyfish.metaphone(name2)
        metaphone_match = 1.0 if metaphone1 == metaphone2 else 0.0
        
        # Average phonetic similarity
        avg_phonetic = (soundex_match + metaphone_match) / 2 * 100
        
        return {
            "soundex": soundex_match * 100,
            "metaphone": metaphone_match * 100,
            "avg_phonetic": avg_phonetic
        }
    
    def fuzzy_similarity(self, name1: str, name2: str) -> Dict[str, float]:
        """Calculate fuzzy string similarity using various Levenshtein-based methods"""
        if not name1 or not name2:
            return {
                "levenshtein": 0.0,
                "fuzzy_ratio": 0.0,
                "partial_ratio": 0.0,
                "token_sort": 0.0,
                "avg_fuzzy": 0.0
            }
        
        # Levenshtein distance (normalized to percentage)
        levenshtein_distance = jellyfish.levenshtein_distance(name1, name2)
        max_len = max(len(name1), len(name2))
        levenshtein_similarity = (1 - (levenshtein_distance / max_len)) * 100 if max_len > 0 else 0
        
        # FuzzyWuzzy ratios
        fuzzy_ratio = fuzz.ratio(name1, name2)
        partial_ratio = fuzz.partial_ratio(name1, name2)
        token_sort_ratio = fuzz.token_sort_ratio(name1, name2)
        
        # Average fuzzy similarity
        avg_fuzzy = (levenshtein_similarity + fuzzy_ratio + partial_ratio + token_sort_ratio) / 4
        
        return {
            "levenshtein": levenshtein_similarity,
            "fuzzy_ratio": fuzzy_ratio,
            "partial_ratio": partial_ratio,
            "token_sort": token_sort_ratio,
            "avg_fuzzy": avg_fuzzy
        }
    
    def find_similar_trademarks(self, new_trademark: Dict, similarity_threshold: float = 50.0) -> List[Dict]:
        """Find similar trademarks in the existing database"""
        if not self.existing_trademarks:
            return []
        
        similar_trademarks = []
        
        # Map extracted trademark fields to comparison fields
        # For new trademark (from PDF extraction)
        new_company_name = self.normalize_name(new_trademark.get('name', ''))
        new_trademark_text = self.normalize_name(new_trademark.get('text_in_logo', ''))
        
        # Use trademark text first, fallback to company name if trademark text is empty
        if new_trademark_text:
            new_name = new_trademark_text
            fallback_used = False
        else:
            new_name = new_company_name
            fallback_used = True
        
        for existing in self.existing_trademarks:
            # Map CSV fields to comparison fields
            existing_company_name = self.normalize_name(existing.get('Client / Applicant', ''))
            existing_trademark_text = self.normalize_name(existing.get('Trademark', ''))
            
            # Use trademark text first, fallback to company name if trademark text is empty
            if existing_trademark_text:
                existing_name = existing_trademark_text
            else:
                existing_name = existing_company_name
            
            if not existing_name or not new_name:
                continue
            
            # Calculate name similarity
            phonetic_sim = self.phonetic_similarity(new_name, existing_name)
            fuzzy_sim = self.fuzzy_similarity(new_name, existing_name)
            
            name_similarity = {
                "phonetic": phonetic_sim,
                "fuzzy": fuzzy_sim
            }
            
            # For now, set logo similarity to 0 (you can implement logo comparison later)
            logo_similarity = 0.0
            
            # Calculate maximum similarity score
            all_scores = [
                phonetic_sim["avg_phonetic"],
                fuzzy_sim["avg_fuzzy"],
                logo_similarity
            ]
            max_similarity = max(all_scores)
            
            # Determine similarity type
            if phonetic_sim["avg_phonetic"] == max_similarity:
                similarity_type = "phonetic"
            elif fuzzy_sim["avg_fuzzy"] == max_similarity:
                similarity_type = "fuzzy"
            else:
                similarity_type = "logo"
            
            if max_similarity >= similarity_threshold:
                similar_trademarks.append({
                    "existing_trademark": existing,
                    "similarity_score": max_similarity,
                    "similarity_level": self.classify_similarity_level(max_similarity),
                    "similarity_type": similarity_type,
                    "fallback_used": fallback_used,
                    "detailed_scores": {
                        "name_comparison": {
                            "overall": max(phonetic_sim["avg_phonetic"], fuzzy_sim["avg_fuzzy"]),
                            "phonetic": phonetic_sim,
                            "fuzzy": fuzzy_sim
                        },
                        "logo_comparison": {
                            "overall": logo_similarity,
                            "fuzzy": {"logo_similarity": logo_similarity}
                        }
                    }
                })
        
        # Sort by similarity score (highest first)
        similar_trademarks.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similar_trademarks
    
    def classify_similarity_level(self, score: float) -> str:
        """Classify similarity level based on score"""
        if score >= 85:
            return "HIGH - Likely same brand"
        elif score >= 70:
            return "MEDIUM - Potential conflict"
        elif score >= 50:
            return "LOW - Worth reviewing"
        else:
            return "MINIMAL - No significant similarity"
    
    def generate_comparison_report(self, new_trademark: Dict, similarity_threshold: float = 50.0) -> Dict:
        """Generate a comprehensive comparison report"""
        similar_trademarks = self.find_similar_trademarks(new_trademark, similarity_threshold)
        
        # Map the new trademark to the expected format for the Streamlit app
        mapped_new_trademark = {
            "name": new_trademark.get("name", ""),
            "trademark": new_trademark.get("text_in_logo", ""),
            "application_no": new_trademark.get("registration_number", ""),
            "class": new_trademark.get("business_category", ""),
            "status": new_trademark.get("legal_status", "")
        }
        
        report = {
            "new_trademark": mapped_new_trademark,
            "matches": similar_trademarks,
            "similar_trademarks_found": len(similar_trademarks),
            "total_existing_trademarks": len(self.existing_trademarks),
            "similarity_threshold": similarity_threshold
        }
        
        return report