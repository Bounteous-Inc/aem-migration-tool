
# AEM Migration Tool

This project processes a CSV or Excel file containing URLs and their content types, groups and enriches the data, crawls live HTML, performs semantic grouping and SEO analysis using OpenAI, and generates reports and visualizations.

## Folders
- `input/`   : Place your input CSV/XLSX files here
- `output/`  : All results, enriched data, and charts will be saved here
- `scripts/` : Main code and modules

## Quick Start
1. Place your input file in the `input/` folder (must have columns for Address/URL and Content Type).
2. Set your OpenAI API key in your terminal:
	```sh
	export OPENAI_API_KEY=sk-<your-key-here>
	```
3. (Optional but recommended) Create and activate a Python virtual environment:
	```sh
	python3 -m venv .venv
	source .venv/bin/activate
	```
4. Install dependencies:
	```sh
	pip install -r requirements.txt
	```
5. Run the main enrichment script:
	```sh
	python scripts/semantic_enrich.py
	```
6. Generate a semantic group report PNG:
	```sh
	python scripts/semantic_group_report.py
	```
7. Find results in the `output/` folder.

## Requirements
- Python 3.8+
- pandas
- openpyxl
- requests
- beautifulsoup4
- matplotlib
- openai
- scikit-learn
- numpy

## Features
- Filters and groups URLs by structure and semantic similarity (OpenAI embeddings)
- Crawls live HTML and extracts metadata
- Performs SEO analysis and summarization using OpenAI GPT
- Exports enriched data to CSV and XLSX
- Visualizes URL groups as a bar chart (PNG)
- Generates a summary report with grouping accuracy

## Virtual Environment
Using a Python virtual environment is recommended to avoid dependency conflicts, but not strictly required. You may use your system Python if you prefer.
