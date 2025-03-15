from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, PubMedAPIWrapper
from langchain.tools import tool
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from pybtex.database import parse_string
import arxiv
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Existing tools
def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves research data to text file",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="web_search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# New Features
@tool
def pdf_report_tool(data: dict, filename: str = "research_report.pdf"):
    """Generates a professional PDF research report"""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title = Paragraph(data["topic"], styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))
    
    for section in ["summary", "analysis"]:
        if data.get(section):
            p = Paragraph(f"<b>{section.capitalize()}:</b><br/>{data[section]}", styles["Normal"])
            story.append(p)
            story.append(Spacer(1, 12))
    
    if data.get("charts"):
        story.append(Paragraph("<b>Visualizations:</b>", styles["Normal"]))
        for chart in data["charts"]:
            story.append(Paragraph(f"<i>{chart}</i>", styles["Normal"]))
    
    doc.build(story)
    return f"PDF report generated at {filename}"

@tool
def citation_tool(bibtex_entry: str):
    """Formats citations using BibTeX entries"""
    try:
        bib_data = parse_string(bibtex_entry, "bibtex")
        formatted = bib_data.to_string("bibtex")
        return formatted
    except Exception as e:
        return f"Citation error: {str(e)}"

@tool
def data_extraction_tool(url: str):
    """Extracts structured data from web pages"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        return {
            "title": soup.title.string if soup.title else "",
            "headers": [h.get_text() for h in soup.find_all(["h1", "h2", "h3"])],
            "paragraphs": [p.get_text() for p in soup.find_all("p")],
            "links": [a["href"] for a in soup.find_all("a", href=True)]
        }
    except Exception as e:
        return f"Extraction error: {str(e)}"

@tool
def arxiv_tool(query: str, max_results: int = 5):
    """Searches academic papers on arXiv"""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for result in search.results():
        results.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "summary": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "pdf_url": result.pdf_url
        })
    return results

@tool
def pubmed_tool(query: str, max_results: int = 5):
    """Searches biomedical literature on PubMed"""
    pubmed = PubMedAPIWrapper()
    return pubmed.run(f"{query} retmax={max_results}")

@tool
def data_analysis_tool(data: dict, chart_type: str = "bar"):
    """Analyzes data and generates visualizations"""
    try:
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        
        if chart_type == "bar":
            df.plot.bar()
        elif chart_type == "line":
            df.plot.line()
        elif chart_type == "hist":
            df.hist()
            
        filename = f"chart_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
        plt.savefig(filename)
        plt.close()
        return {"chart": filename, "stats": df.describe().to_dict()}
    except Exception as e:
        return f"Analysis error: {str(e)}"

# RAG System
model = SentenceTransformer("all-MiniLM-L6-v2")
vector_store = FAISS.from_texts([], model)

@tool
def rag_query_tool(query: str):
    """Queries the research knowledge base"""
    docs = vector_store.similarity_search(query, k=3)
    return [doc.page_content for doc in docs]

def add_to_knowledge_base(text: str):
    """Adds documents to the research knowledge base"""
    embedding = model.encode(text)
    doc = Document(page_content=text)
    vector_store.add_embeddings([embedding], [doc])