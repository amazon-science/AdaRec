#!/usr/bin/env python3
"""
Dataset Preparation Script for UCI Online Retail Dataset

Script to create the example dataset by downloading product categories from GitHub
and processing the UCI Online Retail dataset.

This script will:
1. Download product categories from GitHub if not present locally
2. Fetch and clean the UCI Online Retail dataset
3. Engineer customer features (RFM analysis, category preferences)
4. Generate product description embeddings
5. Create the final dataset files

Usage:
    python script/prepare_example_dataset.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import polars as pl
from tqdm.auto import tqdm
from ucimlrepo import fetch_ucirepo
from sentence_transformers import SentenceTransformer
import urllib.request
import urllib.error

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))
from counterfactual.similarity_searcher import SimilaritySearcher

def string_to_yearmon(date):
    """Convert date string to year-month format."""
    date = date.split()
    date = date[0].split('/') + date[1].split(':')
    date = date[2] + '-' + date[0].zfill(2)
    return date

def download_product_categories(data_dir):
    """Download product description category file from GitHub."""
    url = "https://raw.githubusercontent.com/cstorm125/cstorm125.github.io/refs/heads/main/data/sales_prediction/product_description_category.csv"
    product_category_file = data_dir / 'product_description_category.csv'
    
    print(f"Downloading product categories from: {url}")
    
    try:
        # Download with progress indication
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                print(f"\rDownloading... {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(url, product_category_file, reporthook=show_progress)
        print(f"\nProduct categories downloaded successfully to: {product_category_file}")
        return True
        
    except urllib.error.URLError as e:
        print(f"\nError downloading product categories: {e}")
        print(f"Please check your internet connection and try again.")
        return False
    except Exception as e:
        print(f"\nUnexpected error downloading product categories: {e}")
        return False

def main():
    """Main function to prepare the example dataset."""
    print("Starting UCI Online Retail dataset preparation...")
    
    # Set up paths
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Load and clean data
    print("Loading UCI Online Retail dataset...")
    online_retail = fetch_ucirepo(id=352) 
    transaction_df = online_retail['data']['original']
    original_nb = transaction_df.shape[0]
    print(f"Original dataset size: {original_nb:,} transactions")

    # Create yearmon for train-valid split
    transaction_df['yearmon'] = transaction_df.InvoiceDate.map(string_to_yearmon)

    # Get rid of transactions without customer ID
    transaction_df = transaction_df[~transaction_df.CustomerID.isna()].reset_index(drop=True)
    has_cid_nb = transaction_df.shape[0]
    print(f"After removing missing CustomerID: {has_cid_nb:,} transactions")

    # Fill in unknown descriptions
    transaction_df['Description'] = transaction_df.Description.fillna('UNKNOWN')

    # Convert customer ID to string
    transaction_df['CustomerID'] = transaction_df['CustomerID'].map(lambda x: str(int(x)))

    # Filter out non-product stock codes
    transaction_df = transaction_df[transaction_df.StockCode.map(
        lambda x: x not in ['BANK CHARGES','C2','DOT','M','PADS','POST']
    )]

    # Simplify by filtering unit price and quantity to be non-zero
    transaction_df = transaction_df[
        (transaction_df.UnitPrice > 0) & (transaction_df.Quantity > 0)
    ].reset_index(drop=True)
    has_sales_nb = transaction_df.shape[0]
    print(f"After filtering valid transactions: {has_sales_nb:,} transactions")

    # Add sales
    transaction_df['Sales'] = transaction_df.UnitPrice * transaction_df.Quantity

    # Download and load product categories
    product_category_file = data_dir / 'product_description_category.csv'
    
    # Check if file exists, if not download it
    if not product_category_file.exists():
        print("Product categories file not found locally. Downloading...")
        if not download_product_categories(data_dir):
            print("Failed to download product categories. Exiting.")
            return
    else:
        print(f"Using existing product categories file: {product_category_file}")
    
    print("Loading product categories...")
    product_description_category = pd.read_csv(product_category_file)
    
    # Merge with product categories
    transaction_df = transaction_df.merge(
        product_description_category[['StockCode','category']],
        how='left',
        on='StockCode'
    )

    # Print category distribution
    print("\nProduct category distribution:")
    print(product_description_category.category.value_counts(normalize=True))

    # Define periods
    feature_period = {'start': '2011-01', 'end': '2011-09'}
    outcome_period = {'start': '2011-10', 'end': '2011-12'}

    feature_transaction = transaction_df[
        (transaction_df.yearmon >= feature_period['start']) &
        (transaction_df.yearmon <= feature_period['end'])
    ]
    outcome_transaction = transaction_df[
        (transaction_df.yearmon >= outcome_period['start']) &
        (transaction_df.yearmon <= outcome_period['end'])
    ]

    print(f"\nFeature period transactions: {len(feature_transaction):,}")
    print(f"Outcome period transactions: {len(outcome_transaction):,}")

    # Aggregate sales during outcome period
    outcome_sales = outcome_transaction.groupby('CustomerID').Sales.sum().reset_index()

    # Aggregate sales during feature period
    feature_sales = feature_transaction.groupby('CustomerID').Sales.sum().reset_index()

    # Aggregate items during outcome period
    outcome_items = outcome_transaction.groupby('CustomerID').Description.apply(
        lambda x: '|'.join(x.unique())
    )

    # Aggregate categories during feature period
    feature_categories = feature_transaction.groupby('CustomerID').category.apply(
        lambda x: '|'.join(x.unique())
    )

    # Aggregate categories during outcome period
    outcome_categories = outcome_transaction.groupby('CustomerID').category.apply(
        lambda x: '|'.join(x.unique())
    )

    # Aggregate items during feature period
    feature_items = feature_transaction.groupby('CustomerID').Description.apply(
        lambda x: '|'.join(x.unique())
    )

    # Merge to get TargetSales including those who spent during feature period but not during outcome (zeroes)
    outcome_df = (feature_sales[['CustomerID']]
                 .merge(outcome_sales, on='CustomerID', how='left')
                 .merge(outcome_items, on='CustomerID', how='left')
                 .merge(outcome_categories, on='CustomerID', how='left')
                 .merge(feature_items, on='CustomerID', how='left')
                 .merge(feature_categories, on='CustomerID', how='left'))

    outcome_df.columns = [
        'CustomerID',
        'TargetSales','TargetDescriptions','TargetCategories',
        'bought_descriptions','bought_categories',
    ]
    outcome_df['TargetSales'] = outcome_df['TargetSales'].fillna(0)
    outcome_df['TargetDescriptions'] = outcome_df['TargetDescriptions'].fillna('')
    outcome_df['TargetCategories'] = outcome_df['TargetCategories'].fillna('')

    print(f"Outcome dataframe shape: {outcome_df.shape}")

    # Convert invoice date to datetime
    feature_transaction['InvoiceDate'] = pd.to_datetime(feature_transaction['InvoiceDate'])

    # Last date in feature set
    current_date = feature_transaction['InvoiceDate'].max()

    # RFM features
    print("Engineering customer features...")
    customer_features = feature_transaction.groupby('CustomerID').agg({
        'InvoiceDate': [
            ('recency', lambda x: (current_date - x.max()).days),
            ('first_purchase_date', 'min'),
            ('purchase_day', 'nunique'),
        ],
        'InvoiceNo': [('nb_invoice', 'nunique')],
        'Sales': [('total_sales', 'sum')],
        'StockCode': [('nb_product', 'nunique')],
        'category': [('nb_category', 'nunique')]
    }).reset_index()

    # Flatten column names
    customer_features.columns = [
        'CustomerID',
        'recency',
        'first_purchase_date',
        'purchase_day',
        'nb_invoice',
        'total_sales',
        'nb_product',
        'nb_category'
    ]

    customer_features['customer_lifetime'] = (current_date - customer_features['first_purchase_date']).dt.days
    customer_features['avg_purchase_frequency'] = customer_features['customer_lifetime'] / customer_features['purchase_day']
    customer_features['avg_purchase_value'] = customer_features['total_sales'] / customer_features['purchase_day']

    # Category preference
    category_sales = feature_transaction.pivot_table(
        values='Sales', 
        index='CustomerID', 
        columns='category', 
        aggfunc='sum', 
        fill_value=0
    )
    category_sales.columns = [i.lower().replace(' ','_') for i in category_sales.columns]
    customer_features = customer_features.merge(category_sales, on='CustomerID', how='left')

    total_sales = customer_features['total_sales']
    for col in category_sales.columns:
        percentage_col = f'per_{col}'
        customer_features[percentage_col] = customer_features[col] / total_sales

    # Select features (matching the notebook)
    selected_features = [
        'recency',
        'purchase_day',
        'total_sales',
        'nb_product',
        'nb_category',
        'customer_lifetime',
        'avg_purchase_frequency',
        'avg_purchase_value',
    ]
    
    # Add percentage features that exist
    percentage_features = [col for col in customer_features.columns if col.startswith('per_')]
    selected_features.extend(percentage_features)

    customer_features = customer_features[['CustomerID'] + selected_features]

    # Merge final dataset
    df = outcome_df.merge(customer_features, on='CustomerID')
    
    # Save final dataset
    output_file = data_dir / 'uci_online_retail.csv'
    df.to_csv(output_file, index=False)
    print(f"\nFinal dataset saved to: {output_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Generate product embeddings if requested
    print("\nGenerating product description embeddings...")
    
    # Load product description 
    product_df = pl.read_csv(product_category_file)
    print(f"Product descriptions: {product_df.shape}")

    # Load embedding model
    model = SentenceTransformer('NovaSearch/stella_en_1.5B_v5')

    # Generate embeddings
    embeddings = model.encode(product_df['Description'], show_progress_bar=True)

    embedding_cols = {f'emb_{i}': embeddings[:, i] for i in range(embeddings.shape[1])}
    product_df = product_df.hstack(pl.DataFrame(embedding_cols))
    
    # Save embeddings
    embedding_file = data_dir / 'product_description_category_emb.csv'
    product_df.write_csv(embedding_file)
    print(f"Product embeddings saved to: {embedding_file}")

    # Initialize SimilaritySearcher
    print("\nTesting similarity searcher...")
    similarity_features = [f'emb_{i}' for i in range(embeddings.shape[1])]
    searcher = SimilaritySearcher(product_df, similarity_features=similarity_features)

    # Example query
    example_product = product_df[0].to_dicts()[0]
    print(f"Example product: {example_product['Description']}")
    
    similar_products = searcher.search(example_product, top_k=5)
    print("Top 5 similar products:")
    for i, product in enumerate(similar_products, 1):
        score = product.get('_similarity_score', 'N/A')
        print(f"  {i}. {product['Description']} (similarity: {score:.3f})")

    print("\nDataset preparation completed successfully!")
    print(f"Generated files:")
    print(f"  - {output_file}")
    print(f"  - {embedding_file}")

if __name__ == '__main__':
    main()
