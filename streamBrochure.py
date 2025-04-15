# imports
import os
import requests
from bs4 import BeautifulSoup
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
import anthropic
import gradio as gr
import re
import json

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-key-if-not-using-env')

openai = OpenAI()
claude = anthropic.Anthropic()
genai.configure()

# Default system message (can be overridden)
DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant that responds in markdown"
BROCHURE_SYSTEM_MESSAGE = "You are an assistant that analyzes the contents of a company website landing page \
and creates a short brochure about the company for prospective customers, investors and recruits. Do not use any logos. Respond in markdown."

# --- Modified Streaming Functions to accept system_message ---

def stream_gpt(prompt, system_message_content): # Added system_message_content argument
    messages = [
        {"role": "system", "content": system_message_content}, # Use passed argument
        {"role": "user", "content": prompt}
      ]
    stream = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        # Add check for None content, although API error usually catches it first
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
             result += chunk.choices[0].delta.content
        yield result

def stream_claude(prompt, system_message_content): # Added system_message_content argument
    result = claude.messages.stream(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0.7,
        system=system_message_content, # Use passed argument
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response = ""
    with result as stream:
        for text in stream.text_stream:
            response += text or ""
            yield response

def stream_gemini(prompt, system_message_content): # Added system_message_content argument
    gemini = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    safety_settings=None,
    system_instruction=system_message_content # Use passed argument
    )

    response = gemini.generate_content(prompt, safety_settings=[
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}], stream=True)

    result = ""
    for chunk in response:
        # Gemini might have empty chunks, check text attribute
        if hasattr(chunk, 'text') and chunk.text:
             result += chunk.text
        yield result

# --- Modified stream_model ---

def stream_model(prompt, model):
    print(model) #Shows what model is being used

    # Start with the default system message
    current_system_message = DEFAULT_SYSTEM_MESSAGE
    final_prompt = prompt # Use a new variable for the potentially modified prompt

    # Check for brochure keywords *before* potentially modifying the prompt
    if 'http' in prompt and 'brochure' in prompt.lower():
        print("Brochure mode activated.")
        # Set the specific system message for brochure creation
        current_system_message = BROCHURE_SYSTEM_MESSAGE
        try:
            # Call create_brochure and store its result
            final_prompt = create_brochure(prompt) # Pass the original prompt
            # Check if create_brochure failed (e.g., returned None or the original prompt on error)
            if final_prompt is None or final_prompt == prompt:
                 # Handle cases where brochure creation didn't produce a new prompt
                 # (e.g., no URL found, or an error occurred inside create_brochure)
                 print("Warning: create_brochure did not return a modified prompt. Using original prompt.")
                 final_prompt = prompt # Ensure final_prompt is not None
                 # Optionally, revert system message if brochure creation failed?
                 # current_system_message = DEFAULT_SYSTEM_MESSAGE
        except Exception as e:
            print(f"Error during create_brochure execution: {e}")
            yield f"Error processing brochure request: {e}" # Send error to UI
            return # Stop processing

    # Ensure final_prompt is always a string before passing to models
    if not isinstance(final_prompt, str):
        print(f"Error: Prompt became non-string type ({type(final_prompt)}). Reverting to original.")
        final_prompt = prompt # Fallback to original prompt
        current_system_message = DEFAULT_SYSTEM_MESSAGE # Revert system message too

    print(f"Using System Message: {current_system_message[:100]}...") # Log which system message is used
    print(f"Using Prompt: {final_prompt[:100]}...") # Log the start of the final prompt

    try:
        if model=="GPT":
            result = stream_gpt(final_prompt, current_system_message) # Pass both
        elif model=="Claude":
            result = stream_claude(final_prompt, current_system_message) # Pass both
        elif model=="Gemini":
            result = stream_gemini(final_prompt, current_system_message) # Pass both
        else:
            raise ValueError("Unknown model")
        yield from result
    except Exception as e:
        # Catch potential errors during streaming (e.g., API errors not caught earlier)
        print(f"Error during model streaming ({model}): {e}")
        yield f"Error communicating with {model}: {e}" # Send error to UI


# --- Modified create_brochure ---

def create_brochure(original_prompt):
    # --- Extract the URL using regex ---
    # This pattern looks for http:// or https:// followed by non-whitespace characters
    url_match = re.search(r"https?://\S+", original_prompt)
    print(f"URL Match Result: {url_match}") # Log the match object

    if not url_match:
        # Handle case where no URL is found
        print("Warning: No URL found in the prompt for brochure creation.")
        # Return the original prompt - stream_model will handle this
        return original_prompt

    extracted_url = url_match.group(0) # Get the matched URL string
    print(f"Extracted URL: {extracted_url}")
    # --- End URL Extraction ---

    # --- Determine Company Name (simple example: use domain) ---
    try:
        # Basic extraction, might need refinement for complex URLs
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", extracted_url)
        company_name = domain_match.group(1) if domain_match else extracted_url
    except Exception:
        company_name = extracted_url # Fallback
    print(f"Determined Company Name: {company_name}")
    # ---

    response_tone = "professional"

    # Now construct the new prompt using the extracted URL
    # Use a clear variable name for the new prompt being built
    brochure_request_prompt = f"Please generate a {response_tone} company brochure for the company at {company_name}. Use the following information from their landing page ({extracted_url}):\n"

    try:
        # Use the extracted URL string here
        print(f"Attempting to fetch content from: {extracted_url}")
        website_content = Website(extracted_url).get_contents()
        brochure_request_prompt += website_content
        print("Successfully fetched and added website content.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {extracted_url}: {e}")
        brochure_request_prompt += f"\n[Error: Could not retrieve content from {extracted_url}. Reason: {e}]"
    except Exception as e: # Catch other potential errors (e.g., parsing)
        print(f"Error processing website {extracted_url}: {e}")
        brochure_request_prompt += f"\n[Error: Could not process content from {extracted_url}. Reason: {e}]"

    # --- ADDED RETURN STATEMENT ---
    return brochure_request_prompt # Return the fully constructed prompt string

# --- Website Class (Unchanged) ---
class Website:
    url: str
    title: str
    text: str

    def __init__(self, url):
        self.url = url
        # Add headers to mimic a browser, some sites block default requests user-agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        # Add a timeout to prevent hanging indefinitely
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        # Try to find the main content area if possible (highly site-specific)
        main_content = soup.find('main') or soup.find('article') or soup.body
        if main_content is None: # Fallback if body is somehow None
             main_content = soup
        # Remove irrelevant tags from the main content area or body
        for irrelevant in main_content(["script", "style", "img", "input", "nav", "footer", "header", "aside", "form"]):
            irrelevant.decompose()
        self.text = main_content.get_text(separator="\n", strip=True)

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\n\nWebpage Contents:\n{self.text}\n\n"

# --- Tool Definition ---
def format_as_html_table(headers: List[str], rows: List[List[str]]) -> str:
    """
    Formats the given headers and rows into an HTML table string.

    Args:
        headers: A list of strings for the table header.
        rows: A list of lists, where each inner list represents a row's cell data.

    Returns:
        A string containing the HTML table. Returns an error message if input is invalid.
    """
    if not isinstance(headers, list) or not all(isinstance(h, str) for h in headers):
        return "Error: 'headers' must be a list of strings."
    if not isinstance(rows, list) or not all(isinstance(row, list) for row in rows):
         return "Error: 'rows' must be a list of lists."
    if rows and len(headers) != len(rows[0]):
        # Basic check, assumes all rows have the same length as the first
        return f"Error: Number of headers ({len(headers)}) does not match number of columns in rows ({len(rows[0])})."

    print(f"Tool 'format_as_html_table' called with headers: {headers}, rows: {rows}") # Log tool usage

    # Start table
    html = "<table border='1' style='border-collapse: collapse; width: 100%;'>\n"

    # Add header row
    html += "  <thead>\n    <tr>\n"
    for header in headers:
        html += f"      <th style='background-color: #f2f2f2; padding: 8px; text-align: left;'>{header}</th>\n"
    html += "    </tr>\n  </thead>\n"

    # Add data rows
    html += "  <tbody>\n"
    for row in rows:
        html += "    <tr>\n"
        # Ensure row has correct number of cells, pad if necessary (or error)
        for i in range(len(headers)):
            cell_data = row[i] if i < len(row) else "" # Handle rows shorter than headers
            html += f"      <td style='padding: 8px; border: 1px solid #ddd;'>{cell_data}</td>\n"
        html += "    </tr>\n"
    html += "  </tbody>\n"

    # End table
    html += "</table>"
    return html

# Dictionary to map tool names to functions
AVAILABLE_TOOLS = {
    "format_as_html_table": format_as_html_table,
}

# --- Gradio UI (Unchanged) ---
view = gr.Interface(
    fn=stream_model,
    inputs=[gr.Textbox(label="Your message:", lines=3), gr.Dropdown(["GPT", "Claude", "Gemini"], label="Select model", value="GPT")], # Added lines and default value
    outputs=[gr.Markdown(label="Response:")],
    title="Multi-Model Assistant with Brochure Feature", # Added title
    description="Enter a prompt. If you include 'brochure' and a URL (e.g., 'Create a brochure for http://example.com'), it will fetch the website content first.", # Added description
    flagging_mode="never"
)
view.launch(inbrowser=True, share=True, debug=True) # Share=True might require login if not on Hugging Face spaces

#include a Gradio UI, streaming, use of the system prompt to add expertise, and the ability to switch between models. Demonstrate use of a tool!
#add audio input so you can talk to it, and have it respond with audio.
#add a screenshot of the website to the output.
#a language tutor, to a company onboarding solution, to a companion AI
#to a social media manager, to a research assistant, to a personal assistant.
