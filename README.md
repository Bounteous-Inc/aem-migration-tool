# AEM Migration Tool

This project processes a CSV or Excel file containing URLs and their content types, groups and enriches the data using only existing metadata (no live crawling), performs semantic grouping using OpenAI, and generates reports and visualizations.

## Input Format
- Place your input file in the `input/` folder. The file must have at least these columns:
  - **Address**: contains URLs
  - **Content Type**: specifies the MIME type of the content
  - (Optional) **Page Title**, **Meta Description**, **Main Text**: if available, these will be used for better grouping.

## Steps Performed
1. **Data Filtering & Grouping**
   - Filter rows where `Content Type` = `text/html`.
   - Categorize each URL by its structure: use the first two path segments (e.g., `/personal/loans`).
   - Assign a unique grouping code to each category (e.g., G1, G2, etc.).
2. **Semantic Grouping (No Crawling)**
   - Use only existing metadata (title, meta, main text, URL structure) for grouping.
   - Generate OpenAI embeddings for each row and cluster using HDBSCAN for best accuracy.
3. **Export Results**
   - Export enriched and grouped data to both CSV (semicolon-separated) and XLSX formats in the `output/` folder.
4. **Visualization & Reporting**
   - Visualize all URL groups using a bar chart (no limit to top 10).
   - Generate a summary report including:
     - Total URLs processed
     - Grouping accuracy (% of URLs in top 10 groups)
     - Explanation of grouping logic

## Quick Start
1. Place your input file in the `input/` folder.
2. Set your OpenAI API key in your terminal:
   ```sh
   export OPENAI_API_KEY=sk-<your-key-here>
   ```
3. (Optional) Create and activate a Python virtual environment:
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
6. Find results in the `output/` folder.

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
- Uses only existing metadata (no live crawling)
- Exports enriched data to CSV and XLSX
- Visualizes URL groups as a bar chart (PNG)
- Generates a summary report with grouping accuracy

## Notes
- Crawling is skipped for maximum speed. Only metadata present in the input file is used for grouping and enrichment.
- For best results, provide as much metadata as possible in your input file.
