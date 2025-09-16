import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
from openai_utils import get_embedding, get_gpt_completion
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_DIR = os.path.join(os.path.dirname(__file__), '../input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../output')

SUPPORTED_EXTENSIONS = ['.csv', '.xlsx']

def find_input_file():
    for fname in os.listdir(INPUT_DIR):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return os.path.join(INPUT_DIR, fname)
    print('No input file found in input/. Supported: .csv, .xlsx')
    sys.exit(1)

def load_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath, sep=None, engine='python')
    elif ext == '.xlsx':
        df = pd.read_excel(filepath)
    else:
        raise ValueError('Unsupported file type')
    return df

def filter_html(df):
    return df[df['Content Type'].str.contains('text/html', case=False, na=False)].copy()

def fetch_html(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return ''

def extract_features(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.title.string.strip() if soup.title and soup.title.string else ''
    meta_desc = ''
    meta = soup.find('meta', attrs={'name': 'description'})
    if meta and meta.get('content'):
        meta_desc = meta['content'].strip()
    # Remove scripts/styles and get main text
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=' ', strip=True)
    words = text.split()
    main_text = ' '.join(words[:1000])
    # First two path segments
    parsed = urlparse(url)
    segments = [seg for seg in parsed.path.split('/') if seg]
    struct = '/'.join(segments[:2]) if segments else ''
    # Topic label via GPT
    prompt = f"Classify the following page into a concise topic label (2-4 words):\nTitle: {title}\nMeta: {meta_desc}\nText: {main_text[:300]}"
    try:
        topic_label = get_gpt_completion(prompt, max_tokens=8)
    except Exception:
        topic_label = ''
    return title, meta_desc, main_text, struct, topic_label

def preprocess_text(*args):
    # Lowercase, remove extra whitespace, join
    return ' '.join([str(a).lower().strip().replace('\n', ' ') for a in args if a]).replace('  ', ' ')

def main():
    input_file = find_input_file()
    df = load_data(input_file)
    # Normalize column names
    def normalize_col(col):
        return str(col).strip().replace('"', '').replace("'", '').replace('\ufeff', '')
    df.columns = [normalize_col(c) for c in df.columns]
    address_col = next((c for c in df.columns if c.lower().replace(' ', '') == 'address'), None)
    content_type_col = next((c for c in df.columns if c.lower().replace(' ', '').startswith('contenttype')), None)
    if not address_col or not content_type_col:
        print(f"Missing required columns. Found columns: {df.columns.tolist()}")
        sys.exit(1)
    df = df.rename(columns={address_col: 'Address', content_type_col: 'Content Type'})
    df = filter_html(df)
    if df.empty:
        print('No HTML rows found after filtering.')
        sys.exit(1)
    # Feature extraction (parallelized HTML fetch)
    urls = df['Address'].tolist()
    htmls = [''] * len(urls)
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_idx = {executor.submit(fetch_html, url): i for i, url in enumerate(urls)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                htmls[idx] = future.result()
            except Exception:
                htmls[idx] = ''
    # Feature extraction (parallelized OpenAI calls)
    def extract_all_features(args):
        html, url = args
        return extract_features(html, url)
    features = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_idx = {executor.submit(extract_all_features, (htmls[i], urls[i])): i for i in range(len(urls))}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                features[idx] = future.result()
            except Exception:
                features[idx] = ('', '', '', '', '')
    titles, metas, main_texts, url_structs, topic_labels = zip(*features)
    df['Page Title'] = titles
    df['Meta Description'] = metas
    df['Main Text'] = main_texts
    df['URL Structure'] = url_structs
    df['Topic Label'] = topic_labels
    # Prepare text for embedding
    texts = [preprocess_text(t, m, mt, us, tl) for t, m, mt, us, tl in zip(titles, metas, main_texts, url_structs, topic_labels)]
    # Embedding (parallelized)
    def embed_text(text):
        return get_embedding(text, model="text-embedding-3-large")
    embeddings = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_idx = {executor.submit(embed_text, text): i for i, text in enumerate(texts)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                embeddings[idx] = future.result()
            except Exception:
                embeddings[idx] = np.zeros(1536)  # fallback
    X = np.array(embeddings)
    # HDBSCAN clustering with parameter tuning
    best_score = -1
    best_labels = None
    best_params = None
    for min_cluster_size in range(2, max(3, len(df)//5)):
        for min_samples in [1, 2, 3]:
            clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = clusterer.fit_predict(X)
            # Ignore if all in one cluster or all outliers
            if len(set(labels)) <= 1 or (labels == -1).all():
                continue
            # Compute silhouette score (ignore outliers)
            mask = labels != -1
            if mask.sum() < 2:
                continue
            score = silhouette_score(X[mask], labels[mask], metric='euclidean')
            if score > best_score:
                best_score = score
                best_labels = labels
                best_params = (min_cluster_size, min_samples)
    # Use best clustering
    if best_labels is None:
        print('Clustering failed. All points are outliers or in one group.')
        df['Semantic Group'] = 'SG1'
    else:
        # Assign outliers to nearest cluster
        labels = best_labels.copy()
        mask = labels == -1
        if mask.any():
            from scipy.spatial.distance import cdist
            centers = np.array([X[best_labels == l].mean(axis=0) for l in set(best_labels) if l != -1])
            dists = cdist(X[mask], centers)
            nearest = dists.argmin(axis=1)
            valid_labels = [l for l in set(best_labels) if l != -1]
            for i, idx in enumerate(np.where(mask)[0]):
                labels[idx] = valid_labels[nearest[i]]
        df['Semantic Group'] = [f"SG{l+1}" for l in labels]
    # Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, 'enriched_semantic.csv'), sep=';', index=False)
    # Accuracy and evaluation
    group_counts = df['Semantic Group'].value_counts()
    top_group_pct = 100 * group_counts.iloc[0] / len(df)
    print(f"Semantic grouping accuracy (largest group %): {top_group_pct:.2f}%")
    if best_score > -1:
        print(f"Best silhouette score: {best_score:.3f} (params: min_cluster_size={best_params[0]}, min_samples={best_params[1]})")
    print('Output saved to output/enriched_semantic.csv')

    # Bar chart report
    num_groups = group_counts.shape[0]
    total_urls = len(df)
    plt.figure(figsize=(10, 6))
    group_counts.plot(kind='bar', color='skyblue')
    plt.title(f'URL Count per Semantic Group (Total URLs: {total_urls}, Groups: {num_groups})')
    plt.xlabel('Semantic Group')
    plt.ylabel('Number of URLs')
    plt.tight_layout()
    report_png = os.path.join(OUTPUT_DIR, 'semantic_group_report.png')
    plt.savefig(report_png)
    print(f'Report saved as {report_png}')

if __name__ == '__main__':
    main()
