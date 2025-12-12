# Dataset Preparation Script

This directory contains scripts to prepare the UCI Online Retail dataset for use with AdaRec.

## Quick Start

To create the example dataset, simply run:

```bash
cd AdaRec
python script/prepare_example_dataset.py
```

This will:
1. Download the UCI Online Retail dataset
2. Clean and preprocess the transaction data
3. Use existing product categories from `data/product_description_category.csv`
4. Engineer customer features (RFM analysis, category preferences, etc.)
5. Generate product description embeddings
6. Create the final dataset files

## Requirements

The script requires the existing product categories file to be present:
- `data/product_description_category.csv`

This file should already exist in your data directory from the original notebook execution.

## Output Files

The script generates:
- `data/uci_online_retail.csv` - Main customer dataset with features and targets
- `data/product_description_category_emb.csv` - Product descriptions with embeddings

## Dataset Structure

The final dataset (`uci_online_retail.csv`) contains:

### Target Variables
- `TargetSales` - Sales amount during outcome period (2011-10 to 2011-12)
- `TargetDescriptions` - Products purchased during outcome period
- `TargetCategories` - Product categories purchased during outcome period

### Customer Features (from 2011-01 to 2011-09)
- `recency` - Days since last purchase
- `purchase_day` - Number of unique purchase days
- `total_sales` - Total sales amount
- `nb_product` - Number of unique products purchased
- `nb_category` - Number of unique categories purchased
- `customer_lifetime` - Days from first to last purchase
- `avg_purchase_frequency` - Average days between purchases
- `avg_purchase_value` - Average purchase amount per day
- `per_*` - Percentage of spending in each product category

### Historical Data
- `bought_descriptions` - Products purchased during feature period
- `bought_categories` - Categories purchased during feature period

## Advanced Usage

For more control over the dataset preparation process, use the full script:

```bash
python script/prepare_dataset.py --help
```

This provides options for:
- Custom date ranges
- Product categorization with Claude AI
- Different embedding models
- Force regeneration of existing files
