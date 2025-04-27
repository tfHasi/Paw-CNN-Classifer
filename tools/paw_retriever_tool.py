import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, Any

class PawRetrieverTool:
    def __init__(self):
        self.sources = {
            "akc": "https://www.akc.org/dog-breeds/",
            "dogtime": "https://dogtime.com/dog-breeds/"
        }
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def scrape_breed_info(self, breed: str) -> Dict[str, Any]:
        formatted_breed = breed.lower().replace(" ", "-")
        results = {
            "breed": breed,
            "content": {},
            "success": False,
            "error": None
        }

        for source_name, base_url in self.sources.items():
            try:
                url = f"{base_url}{formatted_breed}"
                response = requests.get(url, headers=self.headers, timeout=10)
                time.sleep(1)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    extract_method = getattr(self, f"_extract_{source_name}_content")
                    results["content"][source_name] = extract_method(soup)
                    results["success"] = True 
            except Exception as e:
                if not results["error"]:
                    results["error"] = f"Error scraping {source_name}: {str(e)}"
                    
        return results
    
    def _extract_akc_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        content = {
            "general_info": {},
            "temperament": "",
            "health": "",
            "history": ""
        }

        if breed_info := soup.find('div', class_='breed-hero-info'):
            for stat in breed_info.find_all('div', class_='attribute-list__row'):
                if label := stat.find('div', class_='attribute-list__term'):
                    if value := stat.find('div', class_='attribute-list__description'):
                        content["general_info"][label.text.strip()] = value.text.strip()

        for section_id, content_key in [
            ('temperament', 'temperament'),
            ('health', 'health'),
            ('history', 'history')
        ]:
            if section := soup.find('div', id=section_id):
                if paragraph := section.find_next('p'):
                    content[content_key] = paragraph.text.strip()
                    
        return content
    
    def _extract_dogtime_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        content = {
            "general_info": {},
            "temperament": "",
            "health": "",
            "care": ""
        }

        if vital_stats := soup.find('div', class_='vital-stat-box'):
            for stat in vital_stats.find_all('div', class_='vital-stat'):
                if label := stat.find('span', class_='vital-stat-name'):
                    if value := stat.find('span', class_='vital-stat-value'):
                        content["general_info"][label.text.strip()] = value.text.strip()

        for section in soup.find_all('div', class_='breed-characteristics-ratings-wrapper'):
            if header := section.find('h2'):
                if 'Personality' in header.text:
                    traits = section.find_all('div', class_='characteristic-title')
                    content["temperament"] = ', '.join([t.text.strip() for t in traits if t])
                
                elif 'Care' in header.text:
                    care_details = section.find_all('div', class_='characteristic-star-block')
                    care_items = []
                    for detail in care_details:
                        if trait := detail.find('div', class_='characteristic-title'):
                            care_items.append(trait.text.strip())
                    content["care"] = ", ".join(care_items)

        if health_section := soup.find('section', class_='health-section'):
            if paragraph := health_section.find('p'):
                content["health"] = paragraph.text.strip()
                
        return content