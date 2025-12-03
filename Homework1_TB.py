"""
OpenAI Function Calling with Tavily Web Search
===============================================

Tento skript demonstruje použití OpenAI function calling s Tavily web search API.

CO SKRIPT DĚLÁ:
--------------
1. Zavolá OpenAI GPT-4o API s uživatelskou otázkou
2. LLM rozhodne, zda potřebuje použít nástroj get_news (Tavily search)
3. Pokud ano, skript spustí Tavily vyhledávání a získá aktuální články
4. Výsledky se vrátí zpět do LLM
5. LLM vytvoří finální odpověď v přirozeném jazyce
"""

import os
import json
from tavily import TavilyClient
from openai import OpenAI
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# Function Implementations
def get_news(query: str, max_results: int = 5):
    """Search for recent news using Tavily."""
    try:
        response = tavily_client.search(
            query=query,  
            max_results=max_results,
            search_depth="advanced"
        )
        
        # Formátuj výsledky
        results = []
        for result in response.get('results', []):
            results.append({
                "title": result.get('title', 'N/A'),
                "url": result.get('url', 'N/A'),
                "content": result.get('content', 'N/A'),
                "score": result.get('score', 0)
            })
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to fetch aromatherapy news"
        }


# Define custom tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Search the web for current information using Tavily",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for aromatherapy information"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]

available_functions = {
    "get_news": get_news
}

# Function to process messages and handle function calls
def get_completion_from_messages(messages, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,  # Custom tools
        tool_choice="auto"  # Allow AI to decide if a tool should be called
    )

    response_message = response.choices[0].message

    print("First response:", response_message)

    if response_message.tool_calls:
        # Find the tool call content
        tool_call = response_message.tool_calls[0]

        # Extract tool name and arguments
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments) 
        tool_id = tool_call.id
        
        # Call the function
        function_to_call = available_functions[function_name]
        function_response = function_to_call(**function_args)

        print(function_response)

        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_id,  
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(function_args),
                    }
                }
            ]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,  
            "name": function_name,
            "content": json.dumps(function_response),
        })

        # Second call to get final response based on function output
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,  
            tool_choice="auto"  
        )
        final_answer = second_response.choices[0].message

        print("Second response:", final_answer)
        return final_answer

    return "No relevant function call found."

# Example usage
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What are the recent discoveries and news in aromatherapy?"},
]

response = get_completion_from_messages(messages)
print("--- Full response: ---")
pprint(response)
print("--- Response text: ---")
print(response.content)
