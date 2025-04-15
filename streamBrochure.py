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

# --- Modified Streaming Functions to accept system_message AND HANDLE TOOLS ---

def stream_gpt(prompt, system_message_content):
    messages = [
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": prompt}
    ]
    result = ""
    print("--- Calling OpenAI ---")

    # Initial call with tools enabled
    stream = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        tools=openai_tools,
        tool_choice="auto", # Let the model decide
        stream=True
    )

    # --- Refined Tool Call Accumulation ---
    accumulated_tool_calls = [] # Use a more descriptive name

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta:
            continue

        # Accumulate content
        if delta.content:
            result += delta.content
            yield result # Yield intermediate text results

        # Accumulate tool calls more carefully
        if delta.tool_calls:
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index
                # Ensure list is long enough
                while len(accumulated_tool_calls) <= idx:
                    accumulated_tool_calls.append({}) # Initialize placeholder dict

                current_call = accumulated_tool_calls[idx]

                # Accumulate ID, Type, Function Name (only if present)
                if tc_chunk.id:
                    current_call['id'] = tc_chunk.id
                if tc_chunk.type:
                    current_call['type'] = tc_chunk.type
                if tc_chunk.function:
                    # Initialize function dict if needed
                    if 'function' not in current_call:
                        current_call['function'] = {}
                    # Set name ONLY if provided and non-empty
                    if tc_chunk.function.name: # Check for truthiness (non-empty)
                        current_call['function']['name'] = tc_chunk.function.name
                    # Initialize/append arguments
                    if 'arguments' not in current_call['function']:
                        current_call['function']['arguments'] = ""
                    if tc_chunk.function.arguments:
                        current_call['function']['arguments'] += tc_chunk.function.arguments

        finish_reason = chunk.choices[0].finish_reason
        # --- Tool Call Handling ---
        if finish_reason == "tool_calls":
            # --- Validation Step ---
            # Filter out any accumulated calls that ended up without a valid name
            valid_tool_calls = []
            for i, call in enumerate(accumulated_tool_calls):
                func = call.get('function')
                # Check if function exists and its name is non-empty
                if func and func.get('name'):
                    valid_tool_calls.append(call)
                else:
                    print(f"Warning: Skipping tool call at index {i} due to missing or empty function name: {call}")

            # Check if any valid calls remain
            if not valid_tool_calls:
                print("Error: Assistant requested tool calls, but no valid function names were found after accumulation.")
                yield result + "\n[Error: Assistant attempted to use a tool but failed to specify the function name correctly.]"
                return # Stop processing this request

            print(f"--- OpenAI wants to call tools (validated): {valid_tool_calls} ---")

            # Append the assistant's message with *VALIDATED* tool calls to history
            # Ensure the structure matches what OpenAI expects for the assistant message
            assistant_message_tool_calls = []
            for call in valid_tool_calls:
                 assistant_message_tool_calls.append({
                     "id": call.get("id"),
                     "type": "function", # Should always be 'function' for tool calls
                     "function": {
                         "name": call.get("function", {}).get("name"),
                         "arguments": call.get("function", {}).get("arguments", "")
                     }
                 })

            messages.append({"role": "assistant", "tool_calls": assistant_message_tool_calls})


            # Execute tools and gather results using the validated list
            tool_results_messages = []
            for tool_call in valid_tool_calls: # Iterate over the validated list
                function_name = tool_call['function']['name'] # Safe now due to validation
                tool_function = AVAILABLE_TOOLS.get(function_name)
                tool_call_id = tool_call['id']

                if tool_function:
                    try:
                        # Arguments should be complete now
                        function_args_str = tool_call['function'].get('arguments', '{}')
                        function_args = json.loads(function_args_str)
                        function_response = tool_function(**function_args)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON arguments for {function_name}: {function_args_str}")
                        function_response = f"Error: Invalid JSON arguments provided for {function_name}."
                    except Exception as e:
                        print(f"Error executing tool {function_name}: {e}")
                        import traceback
                        traceback.print_exc() # Print full traceback for tool errors
                        function_response = f"Error executing tool {function_name}: {e}"

                    tool_results_messages.append({
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_response), # Ensure content is string
                    })
                else:
                     print(f"Warning: Tool '{function_name}' requested by LLM but not found.")
                     tool_results_messages.append({
                         "tool_call_id": tool_call_id,
                         "role": "tool",
                         "name": function_name,
                         "content": f"Error: Tool '{function_name}' is not available.",
                     })

            # Append tool results to messages
            messages.extend(tool_results_messages)

            print("--- Calling OpenAI again with tool results ---")
            # Make the second call with tool results
            stream = openai.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
                stream=True
                # No tools needed here unless you want multi-turn tool use
            )
            # Stream the final response
            for chunk in stream:
                 if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    result += chunk.choices[0].delta.content
                    yield result
            return # Exit after handling tool call and streaming final response

    # If loop finishes without tool calls, the accumulated result is final
    print("--- OpenAI finished without tool calls ---")
    # No need for a final yield here, it happens inside the loop


def stream_claude(prompt, system_message_content):
    messages = [{"role": "user", "content": prompt}]
    response = ""
    print("--- Calling Claude ---")

    # Use a loop to handle potential tool calls
    while True:
        print(f"Claude call with messages: {messages}")
        stream = claude.messages.stream(
            model="claude-3-haiku-20240307", # Or other models supporting tools
            max_tokens=1500, # Increased tokens slightly
            temperature=0.7,
            system=system_message_content,
            messages=messages,
            tools=anthropic_tools,
            tool_choice={"type": "auto"} # Let Claude decide
        )

        assistant_response_content = [] # Accumulate content parts (text or tool_use)
        current_tool_use = None # Track the current tool being parsed

        with stream as s:
            for event in s:
                # print(f"Claude Event: {event.type}") # Debugging
                if event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        response += delta.text
                        yield response # Yield intermediate text
                        # Add text to assistant's response history
                        if not assistant_response_content or assistant_response_content[-1]['type'] != 'text':
                            assistant_response_content.append({"type": "text", "text": delta.text})
                        else:
                            assistant_response_content[-1]['text'] += delta.text

                    elif delta.type == "input_json_delta":
                         # This indicates the start or continuation of tool arguments
                         if current_tool_use:
                             current_tool_use['input_json'] += delta.partial_json
                         else:
                             print("Warning: Received input_json_delta without active tool_use block")


                elif event.type == "content_block_start":
                     if event.content_block.type == "tool_use":
                         print(f"Claude starting tool use: {event.content_block.name}")
                         current_tool_use = {
                             "type": "tool_use",
                             "id": event.content_block.id,
                             "name": event.content_block.name,
                             "input_json": "" # Initialize input accumulator
                         }
                         # Add the start of the tool use block to the assistant message
                         assistant_response_content.append({
                             "type": "tool_use",
                             "id": event.content_block.id,
                             "name": event.content_block.name,
                             "input": {} # Placeholder for parsed input later
                         })


                elif event.type == "content_block_stop":
                    if current_tool_use:
                        print(f"Claude finished tool use block: {current_tool_use['name']}")
                        # Parse the accumulated JSON input
                        try:
                            parsed_input = json.loads(current_tool_use['input_json'])
                            # Find the corresponding block in assistant_response_content and update input
                            for block in assistant_response_content:
                                if block.get("id") == current_tool_use["id"]:
                                    block["input"] = parsed_input
                                    break
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON for tool {current_tool_use['name']}: {current_tool_use['input_json']}")
                            # How to handle this? Maybe add an error block? For now, log it.
                        current_tool_use = None # Reset for next potential tool

                elif event.type == "message_stop":
                    # The LLM has finished its turn. Check if it ended with a tool request.
                    final_assistant_message = event.message # Get the complete message object
                    print(f"Claude message stop. Final role: {final_assistant_message.role}")

                    # Check if the last content block was a tool_use request
                    tool_calls_requested = [
                        block for block in final_assistant_message.content if block.type == 'tool_use'
                    ]

                    if tool_calls_requested:
                        print(f"--- Claude wants to call tools: {[tc.name for tc in tool_calls_requested]} ---")
                        # Append the assistant's full response (including tool requests)
                        messages.append({"role": "assistant", "content": final_assistant_message.content})

                        # Prepare the next user message containing tool results
                        tool_results_content = []
                        for tool_call in tool_calls_requested:
                            function_name = tool_call.name
                            tool_function = AVAILABLE_TOOLS.get(function_name)
                            tool_use_id = tool_call.id
                            function_args = tool_call.input # Already parsed dict

                            if tool_function:
                                try:
                                    function_response = tool_function(**function_args)
                                except Exception as e:
                                    print(f"Error executing tool {function_name}: {e}")
                                    function_response = f"Error executing tool {function_name}: {e}"
                                tool_results_content.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": str(function_response), # Ensure content is string
                                    # Can also add "is_error": True if needed
                                })
                            else:
                                print(f"Warning: Tool '{function_name}' requested by Claude but not found.")
                                tool_results_content.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": f"Error: Tool '{function_name}' is not available.",
                                    "is_error": True
                                })

                        # Add the user message with tool results for the next turn
                        messages.append({"role": "user", "content": tool_results_content})
                        # Continue the loop to call Claude again with the results
                        break # Exit the inner stream processing, outer loop will call again
                    else:
                        # No tool calls, streaming finished
                        print("--- Claude finished without tool calls ---")
                        return # Exit the function

        # If the stream finishes without message_stop (e.g., error), exit
        print("--- Claude stream ended unexpectedly ---")
        return


def stream_gemini(prompt, system_message_content):
    gemini = genai.GenerativeModel(
        model_name='gemini-1.5-flash', # Ensure model supports function calling
        safety_settings=None,
        system_instruction=system_message_content,
        tools=gemini_tools # Pass the tool definition
    )

    chat = gemini.start_chat(enable_automatic_function_calling=False) # Manual control
    result = ""
    print("--- Calling Gemini ---")

    # Use a loop for potential tool call cycles
    current_prompt = prompt
    while True:
        print(f"Gemini call with prompt/parts: {current_prompt}")
        response_stream = chat.send_message(current_prompt, stream=True)
        # response_stream = gemini.generate_content(current_prompt, stream=True, tools=gemini_tools) # Alternative if not using chat history

        function_call_part = None
        for chunk in response_stream:
            # Check for function call
            if chunk.parts:
                 # Check if any part is a function call
                 fc_parts = [part for part in chunk.parts if part.function_call]
                 if fc_parts:
                     # Assuming only one function call per turn for simplicity here
                     function_call_part = fc_parts[0]
                     print(f"--- Gemini wants to call tool: {function_call_part.function_call.name} ---")
                     break # Stop processing chunks for this turn, handle the tool call

            # Otherwise, process text
            if hasattr(chunk, 'text') and chunk.text:
                result += chunk.text
                yield result # Yield intermediate text

        # --- Tool Call Handling ---
        if function_call_part:
            fc = function_call_part.function_call
            function_name = fc.name
            tool_function = AVAILABLE_TOOLS.get(function_name)

            if tool_function:
                try:
                    # Args are already a dict-like structure in Gemini
                    function_args = dict(fc.args)
                    print(f"Executing Gemini tool {function_name} with args: {function_args}")
                    function_response = tool_function(**function_args)
                except Exception as e:
                    print(f"Error executing tool {function_name}: {e}")
                    function_response = f"Error executing tool {function_name}: {e}"

                # Prepare the FunctionResponse to send back
                tool_response_part = genai.types.Part.from_function_response(
                    name=function_name,
                    response={
                        "content": function_response, # The tool's output
                    }
                )
                # Set the prompt for the *next* iteration to be the tool response
                current_prompt = [tool_response_part]
                # Continue the loop to send the tool result back to Gemini
                continue # Go to the next iteration of the while loop

            else:
                print(f"Warning: Tool '{function_name}' requested by Gemini but not found.")
                # How to signal error back? Send an error message as the next prompt?
                # For now, break the loop. A more robust solution might send an error response part.
                yield result + f"\n[Error: Tool '{function_name}' not found]"
                break # Exit loop if tool is unknown

        else:
            # No function call detected in this turn, streaming is complete
            print("--- Gemini finished without tool calls ---")
            break # Exit the while loop

    # Final yield in case the last chunk didn't trigger yield
    yield result

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
            if final_prompt is None or final_prompt == prompt:
                 print("Warning: create_brochure did not return a modified prompt. Using original prompt.")
                 final_prompt = prompt
        except Exception as e:
            print(f"Error during create_brochure execution: {e}")
            yield f"Error processing brochure request: {e}" # Send error to UI
            return

    if not isinstance(final_prompt, str):
        print(f"Error: Prompt became non-string type ({type(final_prompt)}). Reverting to original.")
        final_prompt = prompt
        current_system_message = DEFAULT_SYSTEM_MESSAGE

    print(f"Using System Message: {current_system_message[:100]}...")
    print(f"Using Prompt: {final_prompt[:100]}...")

    try:
        # These functions now handle tool calls internally
        if model=="GPT":
            result = stream_gpt(final_prompt, current_system_message)
        elif model=="Claude":
            result = stream_claude(final_prompt, current_system_message)
        elif model=="Gemini":
            result = stream_gemini(final_prompt, current_system_message)
        else:
            raise ValueError("Unknown model")
        yield from result
    except Exception as e:
        # Catch potential errors during streaming (e.g., API errors, tool execution errors not caught inside)
        print(f"Error during model streaming ({model}): {e}")
        # Add traceback for debugging
        import traceback
        traceback.print_exc()
        yield f"\n\n[STREAMING ERROR ({model}): {e}]" # Send error to UI


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

# --- Tool Schemas for LLMs ---

# OpenAI Tool Schema
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": "format_as_html_table",
            "description": "Formats structured data (headers and rows) into an HTML table. Use this when the user asks to display data in a table format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "headers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of strings for the table header row.",
                    },
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "description": "List of lists, where each inner list is a row containing cell data as strings.",
                    },
                },
                "required": ["headers", "rows"],
            },
        },
    }
]

# Anthropic Tool Schema
anthropic_tools = [
    {
        "name": "format_as_html_table",
        "description": "Formats structured data (headers and rows) into an HTML table. Use this when the user asks to display data in a table format.",
        "input_schema": {
            "type": "object",
            "properties": {
                "headers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of strings for the table header row.",
                },
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "description": "List of lists, where each inner list is a row containing cell data as strings.",
                },
            },
            "required": ["headers", "rows"],
        }
    }
]

# Gemini Tool Schema (using FunctionDeclaration)
gemini_table_tool_func = genai.types.FunctionDeclaration(
    name="format_as_html_table",
    description="Formats structured data (headers and rows) into an HTML table. Use this when the user asks to display data in a table format.",
    # CORRECTED AGAIN: Use string literals for types, not the Type enum
    parameters={
        'type': "object", # Use string literal "object"
        'properties': {
            'headers': {
                'type': "array", # Use string literal "array"
                'items': {'type': "string"}, # Use string literal "string"
                'description': "List of strings for the table header row."
            },
            'rows': {
                'type': "array", # Use string literal "array"
                'items': {
                    'type': "array", # Use string literal "array"
                    'items': {'type': "string"} # Use string literal "string"
                },
                'description': "List of lists, where each inner list is a row containing cell data as strings."
            }
        },
        'required': ['headers', 'rows']
    }
)
# Wrap the FunctionDeclaration in a Tool object (this part remains the same)
gemini_tools = [genai.types.Tool(function_declarations=[gemini_table_tool_func])]

# --- Gradio UI (Unchanged) ---
view = gr.Interface(
    fn=stream_model,
    inputs=[gr.Textbox(label="Your message:", lines=3), gr.Dropdown(["GPT", "Claude", "Gemini"], label="Select model", value="GPT")], # Added lines and default value
    # Use gr.HTML or gr.Markdown. Markdown usually renders basic HTML tables.
    # Use gr.HTML if you need more control or face rendering issues with Markdown.
    outputs=[gr.Markdown(label="Response:")], # Changed to HTML for reliable table rendering
    title="Multi-Model Assistant with Brochure & Table Tool", # Updated title
    description="Enter a prompt. Try asking for data in a table (e.g., 'List the planets and their order from the sun in a table'). Also supports 'brochure' + URL.", # Updated description
    flagging_mode="never"
)
view.launch(inbrowser=True, share=True, debug=True) # Share=True might require login if not on Hugging Face spaces

#include a Gradio UI, streaming, use of the system prompt to add expertise, and the ability to switch between models. Demonstrate use of a tool!
#add audio input so you can talk to it, and have it respond with audio.
#add a screenshot of the website to the output.
#a language tutor, to a company onboarding solution, to a companion AI
#to a social media manager, to a research assistant, to a personal assistant.
