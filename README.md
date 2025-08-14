# Understanding LangGraph by building RAG_Agent, Drafter application

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started (zsh/Mac)](#getting-started-zshmac)
  - [Using pyenv and uv](#using-pyenv-and-uv)
- [Usage](#usage)
- [Exercises](#exercises)
- [Requirements](#requirements)

---

## Overview

LangGraph is a Python framework for designing and managing the flow of tasks in your application using graph structures. This course demonstrates LangGraph concepts through  agent(node) implementations, and Jupyter notebooks.

This project uses RAG_Agent.py to showcase the practical application of vector databases. By processing and querying the Stock_Market_Performance_2024.pdf file, it enables you to ask natural language questions and retrieve relevant information from the document. This provides a hands-on way to understand how vector databases can be used for semantic search and question-answering tasks.

Drafter.py, is designed to automate content generation and management. By binding a collection of specialized tools (update, save) using langChain's tools, it provides a seamless workflow for creating, updating, and saving data.

---

## Repository Structure

```
LangGraph-Course/
├── Agents/            # Python agents for various tasks (e.g., RAG_Agent, Drafter)
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

**Notable Directories:**
- **Agents/**: Python scripts for agents such as Retrieval-Augmented Generation (RAG) and document drafting.

---`

#### Install Dependencies

```zsh
pip install -r requirements.txt
```

## Usage

- Run agent scripts in `Agents/` for more advanced experiments.
- All code is designed to work in a local.

## Requirements

Core dependencies (see `requirements.txt` for full list):

- langgraph
- langchain
- ipython
- langchain_openai
- langchain_community
- dotenv
- typing
- chromadb
- langchain_chroma


