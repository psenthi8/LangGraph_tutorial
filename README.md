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


