import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# Function to scrape the specific fields from a Fandom page
def scrape_fandom_geography_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Dictionary to store the scraped fields
    scraped_data = {}

    # Scrape the name of the geography
    name_tag = soup.find('h2', {'data-source': 'name'})
    if name_tag:
        scraped_data['Name'] = name_tag.get_text(strip=True)
    else:
        scraped_data['Name'] = "Unknown"

    # Scrape the titles/nicknames
    nicknames_section = soup.find('h2', string='Titles')
    if nicknames_section:
        nicknames_content = nicknames_section.find_next_sibling('div')
        scraped_data['Nicknames/Titles'] = nicknames_content.get_text(strip=True) if nicknames_content else "Not available"
    else:
        scraped_data['Nicknames/Titles'] = "Not available"

    # Scrape the sociocultural characteristics
    sociocultural_section = soup.find('h2', string='Sociocultural characteristics')
    if sociocultural_section:
        sociocultural_content = sociocultural_section.find_next_sibling('div')
        scraped_data['Sociocultural Characteristics'] = sociocultural_content.get_text(strip=True) if sociocultural_content else "Not available"
    else:
        scraped_data['Sociocultural Characteristics'] = "Not available"

    # Scrape the physical characteristics
    physical_section = soup.find('h2', string='Physical characteristics')
    if physical_section:
        physical_content = physical_section.find_next_sibling('div')
        scraped_data['Physical Characteristics'] = physical_content.get_text(strip=True) if physical_content else "Not available"
    else:
        scraped_data['Physical Characteristics'] = "Not available"

    # Scrape the lore
    lore_section = soup.find('span', {'id': 'Lore'})
    if lore_section:
        lore_content = lore_section.find_parent().find_next_sibling('p')
        scraped_data['Lore'] = lore_content.get_text(strip=True) if lore_content else "Not available"
    else:
        scraped_data['Lore'] = "Not available"

    # Scrape the history
    history_section = soup.find('span', {'id': 'History'})
    if history_section:
        history_content = history_section.find_parent().find_next_sibling('p')
        scraped_data['History'] = history_content.get_text(strip=True) if history_content else "Not available"
    else:
        scraped_data['History'] = "Not available"

    # Scrape the history in Arcane
    arcane_history_section = soup.find('span', {'id': 'History_in_Arcane'})
    if arcane_history_section:
        arcane_history_content = arcane_history_section.find_parent().find_next_sibling('p')
        scraped_data['History in Arcane'] = arcane_history_content.get_text(strip=True) if arcane_history_content else "Not available"
    else:
        scraped_data['History in Arcane'] = "Not available"

    # Scrape the locations
    locations_section = soup.find('span', {'id': 'Locations'})
    if locations_section:
        locations_content = locations_section.find_parent().find_next_sibling('p')
        scraped_data['Locations'] = locations_content.get_text(strip=True) if locations_content else "Not available"
    else:
        scraped_data['Locations'] = "Not available"

    # Add static value for category
    scraped_data['Category'] = "LoL_locations"

    # Add the original URL
    scraped_data['URL'] = url

    return scraped_data

# Function to scrape multiple pages and save to a JSON file
def scrape_and_save_to_json(urls, output_file='scraped_geography_data.json'):
    all_data = []
    for url in urls:
        print(f"Scraping: {url}")
        scraped_data = scrape_fandom_geography_page(url)
        all_data.append(scraped_data)

        # Add a 10-15 second delay between each request
        time.sleep(10 + 5 * random.random())  # Sleep for 10 to 15 seconds

    # Save to JSON file
    pd.DataFrame(all_data).to_json(output_file, orient='records')
    print(f"Data saved to {output_file}")

# List of URLs to scrape
urls = [
    "https://leagueoflegends.fandom.com/wiki/Runeterra",
    "https://leagueoflegends.fandom.com/wiki/Piltover",
    "https://leagueoflegends.fandom.com/wiki/Zaun",
    "https://leagueoflegends.fandom.com/wiki/The_Void",
    "https://leagueoflegends.fandom.com/wiki/Bilgewater",
    "https://leagueoflegends.fandom.com/wiki/Shadow_Isles",
    "https://leagueoflegends.fandom.com/wiki/Shurima",
    "https://leagueoflegends.fandom.com/wiki/Noxus",
    "https://leagueoflegends.fandom.com/wiki/Targon",
    "https://leagueoflegends.fandom.com/wiki/Ixtal",
    "https://leagueoflegends.fandom.com/wiki/Ionia",
    "https://leagueoflegends.fandom.com/wiki/Freljord",
    "https://leagueoflegends.fandom.com/wiki/Demacia",
    "https://leagueoflegends.fandom.com/wiki/Bandle_City"
]

# Scrape the pages and save the data into a JSON file
scrape_and_save_to_json(urls, output_file='LoL_geography_data.json')
