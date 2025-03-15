# AI Research Assistant 🤖📚

![Research Assistant Demo](https://via.placeholder.com/800x400.png?text=Research+Assistant+Demo)  
*Automated literature reviews, data analysis, and report generation powered by Gemini AI*

## Table of Contents
1. [Features](#features-)
2. [Installation](#installation-)
3. [Configuration](#configuration-)
4. [Usage](#usage-)
5. [Troubleshooting](#troubleshooting-)
6. [License](#license-)

## Features ✨
- **📄 Academic Research**  
  Search arXiv/PubMed papers & generate summaries
- **📊 Data Analysis**  
  Automated statistics & visualization (bar/line/hist charts)
- **📑 Report Generation**  
  Create PDF/TXT/DOCX reports with citations
- **🌐 Web Integration**  
  Web scraping & DuckDuckGo search
- **🧠 Knowledge Base**  
  RAG system with FAISS vector storage

## Installation 💻

### 1. Prerequisites
- Python 3.10+ ([Download](https://www.python.org/downloads/))
- Google Gemini API Key ([Get Free Key](https://ai.google.dev/))

### 2. One-Time Setup
```bash
# Clone repository
git clone https://github.com/yourusername/ai-research-assistant.git
cd ai-research-assistant

# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/MacOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Create configuration file
echo "GEMINI_API_KEY=your_key_here" > .env