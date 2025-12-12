# AdaRec: Adaptive Generative Recommendations with Large Language Models

AdaRec is a counterfactual inference framework for customer outcomes using few-shot in-context learning. Built with CatBoost, Polars, and DSPy, it generates natural language customer profiles and predicts counterfactual outcomes using Large Language Models with AWS Bedrock.

For more information, refer to our FMSD @ ICML 2025 paper: [AdaRec: Adaptive Generative Recommendations with Large Language Models](https://openreview.net/forum?id=OCGldvUR9C).

## What AdaRec Does

AdaRec transforms tabular customer data into rich narrative profiles and uses them to predict counterfactual outcomes:

1. **Profile Generation**: Converts customer features into natural language narratives
2. **Counterfactual Prediction**: Predicts customer behavior under different treatments/promotions
3. **Few-Shot Learning**: Uses similar customers as examples for better predictions
4. **Product Recommendations**: Generates ideal product descriptions and matches them to catalog items

## Installation

### Requirements
- Python 3.12
- AWS Account with Bedrock access
- UV package manager (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/amazon-science/AdaRec
cd AdaRec
```

2. **Install dependencies**
```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -e .
```

3. **Configure AWS Bedrock**
```bash
# Set up AWS credentials
aws configure
# Ensure you have access to Claude models in your region
```

4. **Prepare example dataset**
```bash
python script/prepare_example_dataset.py
```

## Quick Start

### 1. Run the Getting Started Notebook

The easiest way to understand AdaRec is through our comprehensive example:

```bash
jupyter notebook notebook/getting_started.ipynb
```

This notebook demonstrates the complete workflow using the UCI Online Retail dataset:
- Data preprocessing with Polars
- Profile generation (template → narrative)
- Counterfactual prediction with few-shot learning
- Comparison with gradient boosting baseline
- Product recommendation via semantic similarity

### 2. Basic Usage Example

```python
import polars as pl
from src.counterfactual import BaseProfileGenerator, BatchProfileGenerator, BatchPredictor

# Load your customer data
df = pl.read_csv('your_customer_data.csv')

# Generate template profiles
generator = BaseProfileGenerator(
    df=df,
    feature_config='config/customer_feature_config.yaml',
    treatment_config='config/treatment_config.yaml',
    outcome_config='config/outcome_config.yaml'
)

# Generate narrative profiles using LLMs
batch_generator = BatchProfileGenerator(
    df=df,
    model_config='config/model_config.yaml',
    feature_config='config/customer_feature_config.yaml',
    treatment_config='config/treatment_config.yaml',
    outcome_config='config/outcome_config.yaml'
)
narrative_profiles = batch_generator.generate_batch_narrative_profiles()

# Do the same for reference dataset to get few-shot examples from
ref_batch_generator = BatchProfileGenerator(
    df=ref_df,
    model_config='config/model_config.yaml',
    feature_config='config/customer_feature_config.yaml',
    treatment_config='config/treatment_config.yaml',
    outcome_config='config/outcome_config.yaml'
)
ref_narrative_profiles = ref_batch_generator.generate_batch_narrative_profiles()

# Make counterfactual predictions
predictor = BatchPredictor(
    generator=batch_generator,
    customer_profile_column='narrative_profile',
    outcome_config='config/outcome_config.yaml',
    model_config='config/model_config.yaml',
    reference_generator=ref_batch_generator,  # Historical data for few-shot examples
    similarity_features=['recency', 'total_sales', 'purchase_frequency'],
    top_k_examples=3
)

predictions = predictor.generate_predictions()
```

## Core Components

### Profile Generation
- **BaseProfileGenerator**: Converts tabular features to template text profiles
- **BatchProfileGenerator**: Uses LLMs to create natural narrative profiles from templates

### Counterfactual Prediction
- **BatchPredictor**: Generates counterfactual predictions using few-shot learning
- **SimilaritySearcher**: Finds similar customers for few-shot examples

### Data Processing
- **transform.py**: Polars-based data preprocessing utilities
  - Feature engineering and imputation
  - Stratified sampling and train/test splits
  - Outcome resampling for imbalanced datasets

## Configuration

AdaRec uses YAML configuration files for flexibility:

### Model Configuration (`config/model_config.yaml`)
- AWS Bedrock model settings
- Batch processing parameters
- System prompts for profile generation and prediction

### Feature Configuration (`config/customer_feature_config.yaml`)
- Maps tabular columns to natural language descriptions
- Organizes features by domain (RFM, preferences, etc.)

### Treatment Configuration (`config/treatment_config.yaml`)
- Defines promotional treatments/interventions
- Treatment descriptions for profile generation

### Outcome Configuration (`config/outcome_config.yaml`)
- Target variables and their descriptions
- Outcome distributions for context

## Project Structure

```
AdaRec/
├── config/                     # YAML configuration files
├── data/                       # Example datasets
├── notebook/                   # Jupyter notebooks and examples
│   ├── getting_started.ipynb   # Main tutorial
│   └── src/                    # Notebook-specific source code
├── script/                     # Data preparation scripts
├── src/                        # Main source code
│   ├── counterfactual/         # Core AdaRec components
│   ├── data_processing/        # Polars utilities
│   └── evaluation/             # Metrics and evaluation
└── research/                   # Research pipeline for the original paper
```

## Usage Examples

### Predicting Sales

Predict customer sales behavior under normal conditions without promotions:

1. **Prepare organic sales data**:
   - `customer_id`: Unique customer identifier
   - Customer behavior features (recency, frequency, monetary value, category preferences)
   - Historical organic sales data (no promotional periods)

2. **Configure for organic prediction**:
   - Map customer features in `customer_feature_config.yaml` (RFM metrics, purchase patterns)
   - Set treatment to "No Promotion" in `treatment_config.yaml`
   - Specify sales outcomes in `outcome_config.yaml` (baseline sales amount, purchase likelihood)

3. **Run organic sales prediction**:
```python
# Load customer data from non-promotional periods
organic_df = pl.read_csv('organic_customer_data.csv')

# Generate profiles for baseline customer behavior
organic_batch_generator = BatchProfileGenerator(
    df=organic_df,
    model_config='config/model_config.yaml',
    feature_config='config/customer_feature_config.yaml',
    treatment_config='config/treatment_config.yaml',  # Set to "No Promotion"
    outcome_config='config/outcome_config.yaml'       # Baseline sales targets
)
narrative_profiles = organic_batch_generator.generate_batch_narrative_profiles()

# Predict organic sales behavior
predictor = BatchPredictor(
    generator=organic_batch_generator,
    customer_profile_column='narrative_profile',
    outcome_config='config/outcome_config.yaml',
    model_config='config/model_config.yaml',
    similarity_features=['recency', 'total_sales', 'purchase_frequency'],
    top_k_examples=3
)

# Generate baseline organic predictions
organic_predictions = predictor.generate_predictions()
```

### Predicting Promotional Response

Predict how customers respond to specific promotional campaigns:

1. **Prepare promotional campaign data**:
   - `customer_id`: Unique customer identifier
   - Customer behavior features from pre-promotion period
   - Promotional treatment details (discount type, offer mechanics)
   - Historical promotional response data

2. **Configure for promotional prediction**:
   - Map customer features in `customer_feature_config.yaml` (same RFM metrics)
   - Define specific promotions in `treatment_config.yaml` ("Buy 1 Get 1 Free", "20% Discount", etc.)
   - Specify promotional outcomes in `outcome_config.yaml` (uplift in sales, response rate)

3. **Run promotional response prediction**:
```python
# Load customer data for promotional campaign
promo_df = pl.read_csv('promotional_campaign_data.csv')

# Generate profiles for customers under promotional treatment
promo_batch_generator = BatchProfileGenerator(
    df=promo_df,
    model_config='config/model_config.yaml',
    feature_config='config/customer_feature_config.yaml',
    treatment_config='config/treatment_config.yaml',  # Specific promotional treatments
    outcome_config='config/outcome_config.yaml'       # Promotional response targets
)

# Predict promotional response using organic behavior as reference
promo_predictor = BatchPredictor(
    generator=promo_batch_generator,
    customer_profile_column='narrative_profile',
    outcome_config='config/outcome_config.yaml',
    model_config='config/model_config.yaml',
    reference_generator=organic_reference_generator,  # Use organic data as baseline
    similarity_features=['recency', 'total_sales', 'purchase_frequency'],
    top_k_examples=3
)

# Generate promotional response predictions
promotional_predictions = promo_predictor.generate_predictions()
```

### Product Recommendations

Generate ideal product descriptions and match to catalog:

```python
from sentence_transformers import SentenceTransformer
from src.counterfactual.similarity_searcher import SimilaritySearcher

# Load embedding model
model = SentenceTransformer('NovaSearch/stella_en_1.5B_v5')

# Search product catalog
searcher = SimilaritySearcher(product_df, similarity_features=embedding_cols)
recommendations = searcher.search(customer_embedding)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use AdaRec in your research, please cite our paper:

```bibtex
@article{wang2025adarec,
  title={AdaRec: Adaptive Recommendation with LLMs via Narrative Profiling and Dual-Channel Reasoning},
  author={Wang, Meiyun and Polpanumas, Charin},
  journal={arXiv preprint arXiv:2511.07166},
  year={2025}
}
```

## Appendix

### Example Template Profile

```
[Membership Domain] Number of days since first purchase: 147.
[Purchase History Domain] Product descriptions of previously purchased items: 12
  PENCILS TALL TUBE RED RETROSPOT|RED RETROSPOT MINI CASES|ENAMEL FLOWER JUG
  CREAM|WHITE HANGING HEART T-LIGHT HOLDER|DOORMAT UNION FLAG|CREAM SWEETHEART
  MINI CHEST|WOOD BLACK BOARD ANT WHITE FINISH|PICNIC BASKET WICKER LARGE|PIGGY
  BANK RETROSPOT |GINGHAM HEART DECORATION|STRAWBERRY CERAMIC TRINKET BOX|HEART
  OF WICKER SMALL|ROLL WRAP 50'S CHRISTMAS|ROLL WRAP 50'S RED CHRISTMAS |ROLL
  WRAP VINTAGE CHRISTMAS|ROLL WRAP VINTAGE SPOT |RED SPOT PAPER GIFT BAG| RED
  SPOT GIFT BAG LARGE|15CM CHRISTMAS GLASS BALL 20 LIGHTS| 50'S CHRISTMAS GIFT
  BAG LARGE|50'S CHRISTMAS PAPER GIFT BAG|6 GIFT TAGS 50'S CHRISTMAS |HANGING
  METAL HEART LANTERN|LANTERN CREAM GAZEBO |PLEASE ONE PERSON METAL SIGN|WOODEN
  HEART CHRISTMAS SCANDINAVIAN|HOME SWEET HOME HANGING HEART|HOME SWEET HOME
  CUSHION COVER |ENAMEL BREAD BIN CREAM|PARISIENNE JEWELLERY DRAWER |LARGE IVORY
  HEART WALL ORGANISER|ZINC METAL HEART DECORATION|CHRISTMAS HANGING
  SNOWFLAKE|GREEN CHRISTMAS TREE CARD HOLDER.
[Recency Domain] Number of days since customer's last purchase: 4.
[Frequency Domain] Number of purchase days: 2. Average days between purchases:
  73.50.
[Monetary Domain] Total sales: 592.84. Average basket size: 296.42.
[Diversity Domain] Number of distinct products purchased: 34. Number of distinct
  product categories purchased: 5.
[Category Preference Domain] Percentage of purchases in Fashion Accessories
  category: 5.2%. Percentage of purchases in Home Decor category: 50.6%.
  Percentage of purchases in Kitchen and Dining category: 5.5%. Percentage of
  purchases in Other categories: 0.0%. Percentage of purchases in Outdoor and
  Garden category: 0.0%. Percentage of purchases in Personal Care and Wellness
  category: 0.0%. Percentage of purchases in Seasonal and Holiday category:
  12.5%. Percentage of purchases in Stationary and Gifts category: 26.2%.
  Percentage of purchases in Toys and Games category: 0.0%.
```

### Example Narrative Profile

```
Customer 14691 is a highly engaged customer with 142 days of membership (30th
  percentile) and four purchase occasions. Their most recent purchase was just 1
  day ago, placing them at the 10th percentile for recency and indicating peak
  current engagement. With total sales of £1,769.07 (80th percentile) and an
  average basket size of £442.27 (50th percentile), they demonstrate strong
  monetary value. They've purchased 50 distinct products across 8 categories -
  maximum category breadth - showing exceptional product exploration. Their
  category preferences are remarkably balanced: Home Decor (37.7%), Kitchen and
  Dining (10.7%), Stationary and Gifts (10.7%), Fashion Accessories (10.3%),
  Personal Care (10.4%), Seasonal/Holiday (9.8%), Outdoor and Garden (9.5%), and
  Toys and Games (0.8%). The purchase history reveals interest in doormats, herb
  markers, garden items, lunch bags, hot water bottles, and decorative signs
  with humorous or inspirational messages. Their purchase frequency of 35.5 days
  is significantly faster than the median. This customer is currently in an
  active buying phase with diverse interests across nearly all product
  categories. They represent a high-value, engaged customer who would respond
  well to cross-category promotions, new product launches, or loyalty incentives
  to maintain their momentum and encourage continued frequent purchasing.
[Membership Domain] Number of days since first purchase: 142.
```

### Example Counterfactual Prediction

```
{'customer_id': '14691',
 'template_profile': "[Membership Domain] Number of days since first purchase: 142.\n[Purchase History Domain] Product descriptions of previously purchased items: DOORMAT KEEP CALM AND COME IN|DOORMAT NEW ENGLAND|DOORMAT SPOTTY HOME SWEET HOME|HERB MARKER MINT|HERB MARKER CHIVES |HERB MARKER ROSEMARY|HERB MARKER BASIL|HERB MARKER PARSLEY|HERB MARKER THYME|BLUE GIANT GARDEN THERMOMETER|GREEN GIANT GARDEN THERMOMETER|LE GRAND TRAY CHIC SET|JUMBO BAG ALPHABET|JUMBO BAG DOILEY PATTERNS|LUNCH BAG ALPHABET DESIGN|LUNCH BAG DOILEY PATTERN |JUMBO SHOPPER VINTAGE RED PAISLEY|SET 12 KIDS COLOUR  CHALK STICKS|LOVE HOT WATER BOTTLE|HOT WATER BOTTLE KEEP CALM|HOT WATER BOTTLE TEA AND SYMPATHY|HOT WATER BOTTLE I AM SO POORLY|LUNCH BAG VINTAGE DOILY |JUMBO BAG APPLES|LUNCH BAG APPLE DESIGN|PLEASE ONE PERSON METAL SIGN|GIN + TONIC DIET METAL SIGN|COOK WITH WINE METAL SIGN |DOORMAT ENGLISH ROSE |ZINC HERB GARDEN CONTAINER|VINTAGE  2 METER FOLDING RULER|CREAM SWEETHEART MINI CHEST|PETIT TRAY CHIC|CREAM HEART CARD HOLDER|SWEETHEART RECIPE BOOK STAND|BATH BUILDING BLOCK WORD|WELCOME  WOODEN BLOCK LETTERS|LOVE BUILDING BLOCK WORD|PLAYING CARDS KEEP CALM & CARRY ON|GARDENERS KNEELING PAD KEEP CALM |GARDENERS KNEELING PAD CUP OF TEA |WHITE SKULL HOT WATER BOTTLE |CHOCOLATE HOT WATER BOTTLE|METAL SIGN DROP YOUR PANTS|PACK OF SIX LED TEA LIGHTS|JINGLE BELL HEART ANTIQUE GOLD|SET OF 6 T-LIGHTS SANTA|CARDHOLDER GINGHAM CHRISTMAS TREE|RED STAR CARD HOLDER|DOORMAT CHRISTMAS VILLAGE|DOORMAT MERRY CHRISTMAS RED .\n[Recency Domain] Number of days since customer's last purchase: 1.\n[Frequency Domain] Number of purchase days: 4. Average days between purchases: 35.50.\n[Monetary Domain] Total sales: 1769.07. Average basket size: 442.27.\n[Diversity Domain] Number of distinct products purchased: 50. Number of distinct product categories purchased: 8.\n[Category Preference Domain] Percentage of purchases in Fashion Accessories category: 10.3%. Percentage of purchases in Home Decor category: 37.7%. Percentage of purchases in Kitchen and Dining category: 10.7%. Percentage of purchases in Other categories: 0.0%. Percentage of purchases in Outdoor and Garden category: 9.5%. Percentage of purchases in Personal Care and Wellness category: 10.4%. Percentage of purchases in Seasonal and Holiday category: 9.8%. Percentage of purchases in Stationary and Gifts category: 10.7%. Percentage of purchases in Toys and Games category: 0.8%.",
 'treatment': ['No Promotion'],
 'template_treatment': 'Buy X units to get Y% discount (BxGy) promotion: No Promotion.',
 'outcome': ['345.26', 'Dummy Description'],
 'template_outcome': 'Sales during period of interest: 345.26. Product description of the ideal product for this customer: Dummy Description.',
 'narrative_profile': "Customer 14691 is a highly engaged customer with 142 days of membership (30th percentile) and four purchase occasions. Their most recent purchase was just 1 day ago, placing them at the 10th percentile for recency and indicating peak current engagement. With total sales of £1,769.07 (80th percentile) and an average basket size of £442.27 (50th percentile), they demonstrate strong monetary value. They've purchased 50 distinct products across 8 categories - maximum category breadth - showing exceptional product exploration. Their category preferences are remarkably balanced: Home Decor (37.7%), Kitchen and Dining (10.7%), Stationary and Gifts (10.7%), Fashion Accessories (10.3%), Personal Care (10.4%), Seasonal/Holiday (9.8%), Outdoor and Garden (9.5%), and Toys and Games (0.8%). The purchase history reveals interest in doormats, herb markers, garden items, lunch bags, hot water bottles, and decorative signs with humorous or inspirational messages. Their purchase frequency of 35.5 days is significantly faster than the median. This customer is currently in an active buying phase with diverse interests across nearly all product categories. They represent a high-value, engaged customer who would respond well to cross-category promotions, new product launches, or loyalty incentives to maintain their momentum and encourage continued frequent purchasing.\n[Membership Domain] Number of days since first purchase: 142.\n[Purchase History Domain] Product descriptions of previously purchased items: DOORMAT KEEP CALM AND COME IN|DOORMAT NEW ENGLAND|DOORMAT SPOTTY HOME SWEET HOME|HERB MARKER MINT|HERB MARKER CHIVES |HERB MARKER ROSEMARY|HERB MARKER BASIL|HERB MARKER PARSLEY|HERB MARKER THYME|BLUE GIANT GARDEN THERMOMETER|GREEN GIANT GARDEN THERMOMETER|LE GRAND TRAY CHIC SET|JUMBO BAG ALPHABET|JUMBO BAG DOILEY PATTERNS|LUNCH BAG ALPHABET DESIGN|LUNCH BAG DOILEY PATTERN |JUMBO SHOPPER VINTAGE RED PAISLEY|SET 12 KIDS COLOUR  CHALK STICKS|LOVE HOT WATER BOTTLE|HOT WATER BOTTLE KEEP CALM|HOT WATER BOTTLE TEA AND SYMPATHY|HOT WATER BOTTLE I AM SO POORLY|LUNCH BAG VINTAGE DOILY |JUMBO BAG APPLES|LUNCH BAG APPLE DESIGN|PLEASE ONE PERSON METAL SIGN|GIN + TONIC DIET METAL SIGN|COOK WITH WINE METAL SIGN |DOORMAT ENGLISH ROSE |ZINC HERB GARDEN CONTAINER|VINTAGE  2 METER FOLDING RULER|CREAM SWEETHEART MINI CHEST|PETIT TRAY CHIC|CREAM HEART CARD HOLDER|SWEETHEART RECIPE BOOK STAND|BATH BUILDING BLOCK WORD|WELCOME  WOODEN BLOCK LETTERS|LOVE BUILDING BLOCK WORD|PLAYING CARDS KEEP CALM & CARRY ON|GARDENERS KNEELING PAD KEEP CALM |GARDENERS KNEELING PAD CUP OF TEA |WHITE SKULL HOT WATER BOTTLE |CHOCOLATE HOT WATER BOTTLE|METAL SIGN DROP YOUR PANTS|PACK OF SIX LED TEA LIGHTS|JINGLE BELL HEART ANTIQUE GOLD|SET OF 6 T-LIGHTS SANTA|CARDHOLDER GINGHAM CHRISTMAS TREE|RED STAR CARD HOLDER|DOORMAT CHRISTMAS VILLAGE|DOORMAT MERRY CHRISTMAS RED .\n[Recency Domain] Number of days since customer's last purchase: 1.\n[Frequency Domain] Number of purchase days: 4. Average days between purchases: 35.50.\n[Monetary Domain] Total sales: 1769.07. Average basket size: 442.27.\n[Diversity Domain] Number of distinct products purchased: 50. Number of distinct product categories purchased: 8.\n[Category Preference Domain] Percentage of purchases in Fashion Accessories category: 10.3%. Percentage of purchases in Home Decor category: 37.7%. Percentage of purchases in Kitchen and Dining category: 10.7%. Percentage of purchases in Other categories: 0.0%. Percentage of purchases in Outdoor and Garden category: 9.5%. Percentage of purchases in Personal Care and Wellness category: 10.4%. Percentage of purchases in Seasonal and Holiday category: 9.8%. Percentage of purchases in Stationary and Gifts category: 10.7%. Percentage of purchases in Toys and Games category: 0.8%.",
 'pred_TargetSales': 0.0,
 'pred_TargetDescription': 'Home decor and organizational items (doormats, garden accessories, hot water bottles, decorative signs, building block words)',
 'pred_confidence': 0.62,
 'pred_reasoning': "Customer 14691 is a highly engaged, high-value customer (£1,769.07, 80th percentile) with maximum category breadth (8 categories) and very recent activity (1 day ago). Similar Example 1 (similarity 0.792) showed no return, while Examples 2 and 3 returned with £679.50 and £713.66 respectively. The customer's 35.5-day average purchase frequency is very fast, but they just made a purchase YESTERDAY (1 day ago). The baseline shows 52% make no purchase. Despite their exceptional engagement profile and diverse interests, the immediate recency (just purchased 1 day ago) makes another purchase in the near-term period highly unlikely based on their 35.5-day rhythm. Their next natural purchase window would be ~30-35 days away. Without promotional incentive and given the immediate post-purchase timing, I predict zero sales with 62% confidence for the period of interest. This customer will almost certainly return in their next cycle, but not immediately after just completing a transaction."}
```

### Example Feature Distribution

```
[Membership Domain]
Distribution of Number of days since first purchase:
- Range: 0 to 269
- Percent of non-zero values: 99.60%
- Mean: 168
- Percentiles:
  10th: 37
  20th: 94
  30th: 133
  40th: 168
  50th: 186
  60th: 206
  70th: 225
  80th: 245
  90th: 260
[Recency Domain]
Distribution of Number of days since customer's last purchase:
- Range: 0 to 268
- Percent of non-zero values: 97.40%
- Mean: 79
- Percentiles:
  10th: 4
  20th: 10
  30th: 19
  40th: 35
  50th: 58
  60th: 85
  70th: 115
  80th: 148
  90th: 194
[Frequency Domain]
Distribution of Number of purchase days:
- Range: 1 to 119
- Percent of non-zero values: 100.00%
- Mean: 3
- Percentiles:
  10th: 1
  20th: 1
  30th: 1
  40th: 1
  50th: 2
  60th: 3
  70th: 3
  80th: 4
  90th: 7
Distribution of Average days between purchases:
- Range: 0.00 to 268.00
- Percent of non-zero values: 99.60%
- Mean: 82.68
- Percentiles:
  10th: 16.55
  20th: 30.25
  30th: 40.83
  40th: 51.50
  50th: 63.67
  60th: 80.00
  70th: 101.00
  80th: 131.00
  90th: 187.00
[Monetary Domain]
Distribution of Total sales:
- Range: 3.75 to 159154.14
- Percent of non-zero values: 100.00%
- Mean: 1821.97
- Percentiles:
  10th: 151.98
  20th: 235.23
  30th: 318.76
  40th: 419.91
  50th: 583.36
  60th: 783.90
  70th: 1120.81
  80th: 1714.27
  90th: 2791.62
Distribution of Average basket size:
- Range: 3.75 to 9341.26
- Percent of non-zero values: 100.00%
- Mean: 377.23
- Percentiles:
  10th: 120.97
  20th: 156.90
  30th: 201.16
  40th: 245.83
  50th: 295.67
  60th: 332.41
  70th: 400.77
  80th: 494.51
  90th: 677.10
[Diversity Domain]
Distribution of Number of distinct products purchased:
- Range: 1 to 1,416
- Percent of non-zero values: 100.00%
- Mean: 51
- Percentiles:
  10th: 7
  20th: 11
  30th: 17
  40th: 23
  50th: 32
  60th: 42
  70th: 54
  80th: 73
  90th: 113
Distribution of Number of distinct product categories purchased:
- Range: 1 to 9
- Percent of non-zero values: 100.00%
- Mean: 5
- Percentiles:
  10th: 2
  20th: 3
  30th: 4
  40th: 5
  50th: 6
  60th: 6
  70th: 7
  80th: 8
  90th: 8
[Category Preference Domain]
Distribution of Percentage of purchases in Fashion Accessories category:
- Range: 0.0% to 100.0%
- Percent of non-zero values: 65.60%
- Mean: 10.6%
- Percentiles:
  10th: 0.0%
  20th: 0.0%
  30th: 0.0%
  40th: 1.1%
  50th: 3.7%
  60th: 6.7%
  70th: 10.8%
  80th: 17.1%
  90th: 30.7%
Distribution of Percentage of purchases in Home Decor category:
- Range: 0.0% to 100.0%
- Percent of non-zero values: 92.00%
- Mean: 36.2%
- Percentiles:
  10th: 3.9%
  20th: 13.5%
  30th: 20.1%
  40th: 27.0%
  50th: 31.9%
  60th: 39.0%
  70th: 46.6%
  80th: 57.5%
  90th: 72.7%
Distribution of Percentage of purchases in Kitchen and Dining category:
- Range: 0.0% to 100.0%
- Percent of non-zero values: 84.90%
- Mean: 22.8%
- Percentiles:
  10th: 0.0%
  20th: 4.2%
  30th: 8.9%
  40th: 12.9%
  50th: 17.5%
  60th: 24.1%
  70th: 30.4%
  80th: 38.4%
  90th: 49.7%
Distribution of Percentage of purchases in Other categories:
- Range: 0.0% to 30.7%
- Percent of non-zero values: 8.50%
- Mean: 0.3%
- Percentiles:
  10th: 0.0%
  20th: 0.0%
  30th: 0.0%
  40th: 0.0%
  50th: 0.0%
  60th: 0.0%
  70th: 0.0%
  80th: 0.0%
  90th: 0.0%
Distribution of Percentage of purchases in Outdoor and Garden category:
- Range: 0.0% to 100.0%
- Percent of non-zero values: 47.80%
- Mean: 4.1%
- Percentiles:
  10th: 0.0%
  20th: 0.0%
  30th: 0.0%
  40th: 0.0%
  50th: 0.0%
  60th: 1.5%
  70th: 3.3%
  80th: 5.7%
  90th: 10.0%
Distribution of Percentage of purchases in Personal Care and Wellness category:
- Range: 0.0% to 100.0%
- Percent of non-zero values: 45.40%
- Mean: 3.6%
- Percentiles:
  10th: 0.0%
  20th: 0.0%
  30th: 0.0%
  40th: 0.0%
  50th: 0.0%
  60th: 1.3%
  70th: 2.9%
  80th: 5.3%
  90th: 10.1%
Distribution of Percentage of purchases in Seasonal and Holiday category:
- Range: 0.0% to 100.0%
- Percent of non-zero values: 67.00%
- Mean: 8.5%
- Percentiles:
  10th: 0.0%
  20th: 0.0%
  30th: 0.0%
  40th: 2.1%
  50th: 4.3%
  60th: 6.3%
  70th: 9.9%
  80th: 14.7%
  90th: 22.3%
Distribution of Percentage of purchases in Stationary and Gifts category:
- Range: 0.0% to 100.0%
- Percent of non-zero values: 74.40%
- Mean: 9.1%
- Percentiles:
  10th: 0.0%
  20th: 0.0%
  30th: 1.5%
  40th: 3.8%
  50th: 5.7%
  60th: 7.8%
  70th: 11.2%
  80th: 15.4%
  90th: 21.7%
Distribution of Percentage of purchases in Toys and Games category:
- Range: 0.0% to 100.0%
- Percent of non-zero values: 54.10%
- Mean: 4.9%
- Percentiles:
  10th: 0.0%
  20th: 0.0%
  30th: 0.0%
  40th: 0.0%
  50th: 0.8%
  60th: 2.4%
  70th: 4.7%
  80th: 7.6%
  90th: 14.2%
```
