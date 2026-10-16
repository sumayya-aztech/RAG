import os
import json
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from .scraper import fetch_website_links, fetch_website_contents
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key.startswith('sk-proj-') and len(api_key)>10:
    print("API key looks good so far")
else:
    print("There might be a problem with your API key? Please visit the troubleshooting notebook!")
    
MODEL = 'gpt-4.1-mini'
openai = OpenAI()

link_system_prompt = """
    You are provided with a list of links found on a webpage.
    You are able to decide which of the links would be most relevant to include in a brochure about the company,
    such as links to an About page, or a Company page, or Careers/Jobs pages.
    You should respond in JSON as in this example:

    {
        "links": [
            {"type": "about page", "url": "https://full.url/goes/here/about"},
            {"type": "careers page", "url": "https://another.full.url/careers"}
        ]
    }
    """

def get_links_user_prompt(url):
    user_prompt = f"""
        Here is the list of links on the website {url} -
        Please decide which of these are relevant web links for a brochure about the company, 
        respond with the full https URL in JSON format.
        Do not include Terms of Service, Privacy, email links.

        Links (some might be relative links):

        """
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)
    return user_prompt

def select_relevant_links(url):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    links = json.loads(result)
    return links

    
def fetch_page_and_all_relevant_links(url):
        contents = fetch_website_contents(url)
        relevant_links = select_relevant_links(url)
        result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
        for link in relevant_links['links']:
            result += f"\n\n### Link: {link['type']}\n"
            result += fetch_website_contents(link["url"])
        return result


brochure_system_prompt = """
You analyze relevant pages from a company's website and create
a short brochure for prospective customers, investors, and recruits.

Write in clear, professional Markdown without code blocks.
Include company culture, customers, products, services, and careers
when the information is available.
Use only facts supported by the website content.
"""


def get_brochure_user_prompt(company_name, url):
    website_content = fetch_page_and_all_relevant_links(url)

    user_prompt = f"""
        Company name: {company_name}

        Create a short company brochure using the website content below.
        Write the brochure in the language requested by the system instructions.

        Website content:
        {website_content}
        """
    return user_prompt[:5000]


def generate_brochure(company_name, url, language="en"):
    if language == "en":
        system_prompt = brochure_system_prompt
    elif language == "ar":
        system_prompt = """
        You are a professional Arabic company brochure writer.

        Write the entire brochure in fluent, professional Arabic
        using Markdown.
        Preserve company names and factual information.
        Do not invent information.
        """
    else:
        raise ValueError(
            "Unsupported language. Choose 'en' or 'ar'."
        )

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": get_brochure_user_prompt(
                    company_name, url
                ),
            },
        ],
    )

    return response.choices[0].message.content or ""
