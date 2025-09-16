# AEM Migration Tool

This project processes all CSV and Excel files in the `input/` folder, groups and enriches the data using OpenAI embeddings and fast KMeans clustering, and generates reports and visualizations. Output is saved in the same format as the input (CSV or XLSX) with group labels.

## Input Format
- Place one or more input files in the `input/` folder. Each file must have at least these columns:
  - **Address**: contains URLs
  - **Content Type**: specifies the MIME type of the content
  - (Optional) **Page Title**, **Meta Description**, **Main Text**: if available, these will be used for better grouping.

## Steps Performed
1. **Batch Data Filtering & Grouping**
   - For each file, filter rows where `Content Type` = `text/html`.
   - Combine available metadata for each row.
   - Use OpenAI embeddings (batched) to represent each row semantically.
   - Use KMeans clustering for fast, scalable grouping.
   - Assign a semantic group label to each row.
2. **Export Results**
   - Export enriched and grouped data to the `output/` folder, in the same format as the input (CSV or XLSX).
3. **Visualization & Reporting**
   - Visualize all semantic groups using a bar chart (one per input file).
   - The report includes:
     - Total URLs processed
     - Group sizes by code
     - Explanation: Grouping is based on OpenAI semantic similarity of available metadata.

## Quick Start
1. Place your input files in the `input/` folder.
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
6. Find results in the `output/` folder, with one enriched file and one bar chart per input file.

## Requirements
- Python 3.8+
- pandas
- openpyxl
- requests
- matplotlib
- numpy
- scikit-learn
- openai

## Features
- Processes all CSV and XLSX files in the input folder
- Groups URLs by semantic similarity using OpenAI embeddings + KMeans
- Exports grouped data in the same format as input
- Visualizes group sizes as a bar chart (PNG)
- Fast, scalable, and works with large datasets
