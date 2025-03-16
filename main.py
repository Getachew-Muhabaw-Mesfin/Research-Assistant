import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
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

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str = Field(..., description="Research topic")
    summary: str = Field(..., description="Detailed summary")
    sources: list[str] = Field(..., description="List of sources")
    tools_used: list[str] = Field(..., description="Tools used in research")
    citations: list[str] = Field(default=[], description="Formatted citations")
    charts: list[str] = Field(default=[], description="Generated chart filenames")
    analysis: str = Field(default="", description="Statistical analysis")

model_choices = {
    "1": "gemini-1.5-flash",
    "2": "gemini-2.0-flash",
    "3": "gemini-1.5-pro"
}

while True:
    model_choice = input("\nChoose the AI model:\n1. gemini-1.5-flash\n2. gemini-2.0-flash\n3. gemini-1.5-pro\nEnter the number (1-3): ").strip()
    if model_choice in model_choices:
        model_name = model_choices[model_choice]
        break
    else:
        print("\n❌ Please select the correct model (1-3) and try again.")

llm = ChatGoogleGenerativeAI(
    model=model_name, 
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an advanced AI research assistant with these capabilities:
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
]).partial(format_instructions=parser.get_format_instructions())

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
print(" 🤖 AI Research Assistant ")
print("=" * 50)

while True:
    try:
        query = input("\n🔍 What can I help you research? (type 'exit' to quit) ").strip()

        if query.lower() == "exit":
            print("\n👋 Goodbye! Closing AI Research Assistant.")
            break 

        raw_response = agent_executor.invoke({"query": query})

        output_text = raw_response.get("output", "")

        if output_text.startswith("```json"):
            output_text = output_text.replace("```json", "").replace("```", "").strip()

        try:
            structured_response = parser.parse(output_text)  
        except Exception as e:
            print("\n❌ **Error:** Failed to parse AI response.")
            print("⚠️ **Raw Response:**", raw_response)
            print("🔍 **Error Details:**", e)
            continue

        print("\n📌 **Research Results**")
        print("-" * 50)
        print(f"📖 **Topic:** {structured_response.topic}")
        print(f"📄 **Summary:**\n{structured_response.summary}")
        print(f"📚 **Sources:** {', '.join(structured_response.sources) or 'No sources found'}")
        print(f"🛠 **Tools Used:** {', '.join(structured_response.tools_used) or 'No tools used'}")

        if structured_response.citations:
            print(f"📑 **Citations:** {', '.join(structured_response.citations)}")

        if structured_response.charts:
            print(f"📊 **Generated Charts:** {', '.join(structured_response.charts)}")

        if structured_response.analysis:
            print(f"📈 **Analysis:**\n{structured_response.analysis}")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Closing AI Research Assistant.")
        break  

    except Exception as e:
        print("\n❌ **Unexpected Error:**", e)
        print("⚠️ **Raw Response:**", raw_response)
