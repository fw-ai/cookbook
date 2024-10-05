import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# Function to scrape the specific fields from a Fandom page
def scrape_fandom_lol_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Dictionary to store the scraped fields
    scraped_data = {}

    # Scrape the champion name
    name_tag = soup.find('h2', class_='pi-item pi-item-spacing pi-title pi-secondary-background')
    if name_tag:
        scraped_data['Name'] = name_tag.find('span').get_text(strip=True)
    else:
        scraped_data['Name'] = "Unknown"

    # Scrape the champion title
    title_tag = soup.find('div', class_='pi-item pi-data pi-item-spacing pi-border-color')
    if title_tag:
        scraped_data['Title'] = title_tag.find('span').get_text(strip=True)
    else:
        scraped_data['Title'] = "Unknown"

    # Scrape the abilities (find section with id="Abilities")
    abilities_section = soup.find('span', {'id': 'Abilities'})
    if abilities_section:
        abilities_content = abilities_section.find_parent().find_next_sibling('div')
        if abilities_content:
            scraped_data['Abilities'] = abilities_content.get_text(strip=True)
        else:
            scraped_data['Abilities'] = "Not available"
    else:
        scraped_data['Abilities'] = "Not available"

    # Add a static value for category
    scraped_data['Category'] = "LoL_gameplay_character"

    # Add the URL for reference
    scraped_data['URL'] = url

    return scraped_data

# Function to scrape multiple pages and save to a JSON file
def scrape_and_save_to_json(urls, output_file='scraped_lol_data.json'):
    all_data = []
    for url in urls:
        print(f"Scraping: {url}")
        scraped_data = scrape_fandom_lol_page(url)
        all_data.append(scraped_data)

        # Add a 10-15 second delay between each request
        time.sleep(10 + 5 * random.random())  # Sleep for 10 to 15 seconds

    # Save to JSON file
    pd.DataFrame(all_data).to_json(output_file, orient='records')
    print(f"Data saved to {output_file}")

# List of URLs to scrape
urls = [
    "https://leagueoflegends.fandom.com/wiki/Caitlyn/LoL",
    "https://leagueoflegends.fandom.com/wiki/Ekko/LoL",
    "https://leagueoflegends.fandom.com/wiki/Jinx/LoL",
    "https://leagueoflegends.fandom.com/wiki/Jayce/LoL",
    "https://leagueoflegends.fandom.com/wiki/Heimerdinger/LoL",
    "https://leagueoflegends.fandom.com/wiki/Singed/LoL",
    "https://leagueoflegends.fandom.com/wiki/Vi/LoL",
    "https://leagueoflegends.fandom.com/wiki/Viktor/LoL",
    "https://leagueoflegends.fandom.com/wiki/Warwick/LoL",
    "https://leagueoflegends.fandom.com/wiki/Lux/LoL",
    "https://leagueoflegends.fandom.com/wiki/Akali/LoL", 
    "https://leagueoflegends.fandom.com/wiki/Ahri/LoL", 
    "https://leagueoflegends.fandom.com/wiki/Miss_Fortune/LoL",
    "https://leagueoflegends.fandom.com/wiki/Katarina/LoL",
    "https://leagueoflegends.fandom.com/wiki/Irelia/LoL",
    "https://leagueoflegends.fandom.com/wiki/Sylas/LoL"
]

# Scrape the pages and save the data into a JSON file
scrape_and_save_to_json(urls, output_file='lol_champion_data.json')
