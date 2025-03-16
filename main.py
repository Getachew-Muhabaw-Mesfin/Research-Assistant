from dotenv import load_dotenv
import json
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import (
    search_tool, 
    wiki_tool, 
    save_tool,
    pdf_report_tool,
    citation_tool,
    data_extraction_tool,
    arxiv_tool,
    pubmed_tool,
    data_analysis_tool,
    rag_query_tool
)
import os
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str = Field(..., description="Research topic")
    summary: str = Field(..., description="Detailed summary")
    sources: list[str] = Field(..., description="List of sources")
    tools_used: list[str] = Field(..., description="Tools used in research")
    citations: list[str] = Field(default=[], description="Formatted citations")
    charts: list[str] = Field(default=[], description="Generated chart filenames")
    analysis: str = Field(default="", description="Statistical analysis")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an advanced AI research assistant with these capabilities:
1. Conduct comprehensive literature reviews using academic databases
2. Perform web scraping and data extraction
3. Generate visualizations and statistical analysis
4. Manage citations and references
5. Create professional reports in multiple formats
6. Maintain a knowledge base of previous research

Use available tools and structure output as:\n{format_instructions}"""
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [
    search_tool, 
    wiki_tool, 
    save_tool,
    pdf_report_tool,
    citation_tool,
    data_extraction_tool,
    arxiv_tool,
    pubmed_tool,
    data_analysis_tool,
    rag_query_tool
]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


print("\n" + "=" * 50)
print(" 🤖 AI Research Assistant Agent ")
print("=" * 50)

while True:
    try:
        query = input("\n🔍 What can I help you research? (type 'exit' to quit) ").strip()

        # Exit condition
        if query.lower() == "exit":
            print("\n👋 Goodbye! Closing AI Research Assistant.")
            break  # Exit loop

        # Invoke AI research assistant
        raw_response = agent_executor.invoke({"query": query})

        # Extract output text
        output_text = raw_response.get("output", "")

        # Remove markdown code block markers (```json ... ```)
        if output_text.startswith("```json"):
            output_text = output_text.replace("```json", "").replace("```", "").strip()

        # Parse JSON response
        structured_response = json.loads(output_text)  # Fix: Proper JSON parsing

        # Display research results
        print("\n📌 **Research Results**")
        print("-" * 50)
        print(f"📖 **Topic:** {structured_response.get('topic', 'Unknown')}")
        print(f"📄 **Summary:**\n{structured_response.get('summary', 'No summary available.')}")
        print(f"📚 **Sources:** {', '.join(structured_response.get('sources', [])) or 'No sources found'}")
        print(f"🛠 **Tools Used:** {', '.join(structured_response.get('tools_used', [])) or 'No tools used'}")

    except json.JSONDecodeError:
        print("\n❌ **Error:** Invalid JSON format received.")
        print("⚠️ **Raw Response:**", raw_response)

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Closing AI Research Assistant.")
        break  # Exit loop on Ctrl+C

    except Exception as e:
        print("\n❌ **Unexpected Error:**", e)
        print("⚠️ **Raw Response:**", raw_response)
