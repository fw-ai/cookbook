import requests
from bs4 import BeautifulSoup
import pandas as pd
import time  # Import time for adding delay
import random  # Import random for adding a randomized delay

# Function to scrape the specific fields from a Fandom page
def scrape_fandom_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Dictionary to store the scraped fields
    scraped_data = {}

    # Helper function to scrape by section ID
    def scrape_section(soup, section_id):
        section = soup.find('span', {'id': section_id})
        if section:
            # Find the parent element containing the content
            content = section.find_parent().find_next_sibling('p')
            return content.get_text(strip=True) if content else "Not available"
        return "Not available"

    # Extract name from the 'data-source' attribute
    name_tag = soup.find(attrs={"data-source": "disp_name"})
    scraped_data['Name'] = name_tag.get_text(strip=True) if name_tag else "Unknown"

    # Scrape the relevant sections
    scraped_data['Background'] = scrape_section(soup, 'Background')
    scraped_data['Appearance'] = scrape_section(soup, 'Appearance')
    scraped_data['Personality'] = scrape_section(soup, 'Personality')
    scraped_data['Abilities'] = scrape_section(soup, 'Abilities')
    scraped_data['Relations'] = scrape_section(soup, 'Relations')

    # Add the URL for reference
    scraped_data['URL'] = url

    # Add static value for Category
    scraped_data['Category'] = "Arcane_characters"

    return scraped_data

# Function to scrape multiple pages and save to a JSON file
def scrape_and_save_to_json(urls, output_file='scraped_data.json'):
    all_data = []
    for url in urls:
        print(f"Scraping: {url}")
        scraped_data = scrape_fandom_page(url)
        all_data.append(scraped_data)

        # Add a 10-15 second delay between each request
        time.sleep(10 + 5 * random.random())  # Sleep for 10 to 15 seconds

    # Save to JSON file
    pd.DataFrame(all_data).to_json(output_file, orient='records')
    print(f"Data saved to {output_file}")

# List of URLs to scrape
urls = [
    "https://leagueoflegends.fandom.com/wiki/Amara",
    "https://leagueoflegends.fandom.com/wiki/Ambessa", 
    "https://leagueoflegends.fandom.com/wiki/Bolbok",
    "https://leagueoflegends.fandom.com/wiki/Caitlyn/Arcane",
    "https://leagueoflegends.fandom.com/wiki/Cassandra",
    "https://leagueoflegends.fandom.com/wiki/Chross",
    "https://leagueoflegends.fandom.com/wiki/Ekko/Arcane",
    "https://leagueoflegends.fandom.com/wiki/Elora",
    "https://leagueoflegends.fandom.com/wiki/Finn",
    "https://leagueoflegends.fandom.com/wiki/Heimerdinger/Arcane",
    "https://leagueoflegends.fandom.com/wiki/Jinx/Arcane",
    "https://leagueoflegends.fandom.com/wiki/Marcus",
    "https://leagueoflegends.fandom.com/wiki/Margot",
    "https://leagueoflegends.fandom.com/wiki/Mel",
    "https://leagueoflegends.fandom.com/wiki/Mylo",
    "https://leagueoflegends.fandom.com/wiki/Renni",
    "https://leagueoflegends.fandom.com/wiki/Salo",
    "https://leagueoflegends.fandom.com/wiki/Sevika",
    "https://leagueoflegends.fandom.com/wiki/Silco",
    "https://leagueoflegends.fandom.com/wiki/Singed/Arcane",
    "https://leagueoflegends.fandom.com/wiki/Smeech",
    "https://leagueoflegends.fandom.com/wiki/Tobias",
    "https://leagueoflegends.fandom.com/wiki/Vander",
    "https://leagueoflegends.fandom.com/wiki/Vi/Arcane",
    "https://leagueoflegends.fandom.com/wiki/Viktor/Arcane"
]

# Scrape the pages and save the data into a JSON file
scrape_and_save_to_json(urls, output_file='arcane_characters_data.json')
