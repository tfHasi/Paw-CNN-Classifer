import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, List, Any

class PawRetrieverTool:
    def __init__(self):
        self.sources = {
            "akc": "https://www.akc.org/dog-breeds/",
            "dogtime": "https://dogtime.com/dog-breeds/"
        }
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def _format_breed_name(self, breed: str) -> str:
        return breed.lower().replace(" ", "-")
    
    def scrape_breed_info(self, breed: str) -> Dict[str, Any]:
        formatted_breed = self._format_breed_name(breed)
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
                    if source_name == "akc":
                        content = self._extract_akc_content(soup)
                    elif source_name == "dogtime":
                        content = self._extract_dogtime_content(soup)
                    results["content"][source_name] = content
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
        try:
            breed_info = soup.find('div', class_='breed-hero-info')
            if breed_info:
                stats = breed_info.find_all('div', class_='attribute-list__row')
                for stat in stats:
                    label = stat.find('div', class_='attribute-list__term')
                    value = stat.find('div', class_='attribute-list__description')
                    if label and value:
                        content["general_info"][label.text.strip()] = value.text.strip()
            temp_section = soup.find('div', id='temperament')
            if temp_section:
                content["temperament"] = temp_section.find_next('p').text.strip()
            health_section = soup.find('div', id='health')
            if health_section:
                content["health"] = health_section.find_next('p').text.strip()
            history_section = soup.find('div', id='history')
            if history_section:
                content["history"] = history_section.find_next('p').text.strip()        
        except Exception:
            pass
        return content
    
    def _extract_dogtime_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        content = {
            "general_info": {},
            "temperament": "",
            "health": "",
            "care": ""
        }
        try:
            vital_stats = soup.find('div', class_='vital-stat-box')
            if vital_stats:
                stats = vital_stats.find_all('div', class_='vital-stat')
                for stat in stats:
                    label = stat.find('span', class_='vital-stat-name')
                    value = stat.find('span', class_='vital-stat-value')
                    if label and value:
                        content["general_info"][label.text.strip()] = value.text.strip()
            sections = soup.find_all('div', class_='breed-characteristics-ratings-wrapper')
            for section in sections:
                header = section.find('h2')
                if header and 'Personality' in header.text:
                    traits = section.find_all('div', class_='characteristic-title')
                    content["temperament"] = ', '.join([t.text.strip() for t in traits if t])
                
                if header and 'Care' in header.text:
                    care_details = section.find_all('div', class_='characteristic-star-block')
                    for detail in care_details:
                        trait = detail.find('div', class_='characteristic-title')
                        if trait:
                            content["care"] += trait.text.strip() + ", "
            health_section = soup.find('section', class_='health-section')
            if health_section:
                content["health"] = health_section.find('p').text.strip() if health_section.find('p') else ""     
        except Exception:
            pass
        return content