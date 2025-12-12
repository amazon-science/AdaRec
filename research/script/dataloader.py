import json
from copy import deepcopy
import re
import pandas as pd
import random
import pickle as pkl
from tqdm import tqdm
import pickle
import os
from typing import Dict, Any, Union
from script.utils import *


def value_mapping():
    values_dict = {
        "gl_14_order_days_360d": "Books",
        "gl_21_order_days_360d": "Toys",
        "gl_121_order_days_360d": "Drug Store",
        "gl_23_order_days_360d": "Electronics",
        "gl_194_order_days_360d": "Beauty",
        "gl_201_order_days_360d": "Home",
        "gl_79_order_days_360d": "Kitchen",
        "gl_107_order_days_360d": "Wireless",
        "gl_193_order_days_360d": "Apparel",
        "gl_200_order_days_360d": "Sports",
        "gl_325_order_days_360d": "Grocery",
        "gl_60_order_days_360d": "Home Improvement",
        "gl_405_order_days_360d": "Mobile Apps",
        "gl_63_order_days_360d": "Video Games",
        "gl_147_order_days_360d": "PC",
        "gl_263_order_days_360d": "Automotive",
        "gl_309_order_days_360d": "Shoes",
        "gl_351_order_days_360d": "E-books",
        "gl_15_order_days_360d": "Music",
        "gl_229_order_days_360d": "Office Products",
        "gl_others": "Others",
        "gl_14_session_30d": "Books",
        "gl_21_session_30d": "Toys",
        "gl_23_session_30d": "Electronics",
        "gl_351_session_30d": "E-books",
        "gl_79_session_30d": "Kitchen",
        "gl_121_session_30d": "Drug Store",
        "gl_60_session_30d": "Home Improvement",
        "gl_201_session_30d": "Home",
        "gl_194_session_30d": "Beauty",
        "gl_193_session_30d": "Apparel",
        "gl_147_session_30d": "PC",
        "gl_107_session_30d": "Wireless",
        "gl_325_session_30d": "Grocery",
        "gl_200_session_30d": "Sports",
        "gl_63_session_30d": "Video Games",
        "gl_15_session_30d": "Music",
        "gl_309_session_30d": "Shoes",
        "gl_263_session_30d": "Automotive",
        "gl_74_session_30d": "DVD",
        "gl_others": "Others",
        "prime_video_days_360d": "Prime Video",
        "day1_ship_days_360d": "Prime 1-day Shipping",
        "cloud_drive_app_days_360d": "Cloud Drive App",
        "hawkfire_days_360d": "Prime Music Unlimited",
        "prime_music_days_360d": "Prime Music",
        "day2_ship_days_360d": "Prime 2-day Shipping",
        "day3_5_ship_days_360d": "Prime 3-to-5-day Shipping",
        "cloud_drive_days_360d": "Cloud Drive",
        "non_physical_days_360d": "Digital Goods",
        "amazon_channels_days_360d": "Amazon Channel",
        "prime_reading_days_360d": "Prime Reading",
        "sns_orders_days_360d": "Subscribe and Save",
    }

    return values_dict


def round_if_not_str(x: Union[int, float]) -> str:
    """Rounds a number to the nearest integer if it's not already a string. Otherwise, returns the string as-is.

    Args:
        x: The value to round or return (int or float).

    Returns:
        A string representation of the rounded value or the original string.
    """
    value_dict = value_mapping()

    x = str("{:,}".format(round(x))) if type(x) != str else x
    x = value_dict[x] if x in value_dict else x
    return x


def generate_customer_text(
    customer_row: pd.Series,
    feature_dict: Dict[str, Dict[str, Any]],
    shuffle_data_cols: bool = False,
) -> str:
    """Generates a customer description text based on features and their corresponding values.

    Args:
        customer_row: A pandas Series containing customer features.
        feature_dict: A dictionary defining feature groups, check columns, and data columns. Refer to the function docstring for details on the dictionary format.
        shuffle_data_cols (bool, optional): If True, shuffles the order of data columns within each feature group before using them. Defaults to False.

    Returns:
        A string describing the customer based on the provided features and values.
    """

    customer_text = feature_dict["prefix"]
    for col_group in list(feature_dict.keys())[1:]:
        if feature_dict[col_group]["check_col"]:
            check_col, not_found_text = feature_dict[col_group]["check_col"]
            if customer_row[check_col] == 0:
                customer_text += not_found_text
                continue
            
        data_col_dict = feature_dict[col_group]["data_cols"]
        data_cols = list(data_col_dict.keys()) 
        
        if shuffle_data_cols:
            random.shuffle(data_cols)

        for data_col in data_cols:
            if customer_row[data_col] == 0:
                customer_text += data_col_dict[data_col][-1].replace(
                    "[VALUE]", round_if_not_str(customer_row[data_col])
                )
            else:
                customer_text += data_col_dict[data_col][0].replace(
                    "[VALUE]", round_if_not_str(customer_row[data_col])
                )  

    return customer_text


def human_template():
    snapshot_features = {
        "prefix": "The customer currently ",
        "population": {
            "check_col": [],
            "data_cols": {
                "total_amazon_tenure": [
                    "has been with Amazon for [VALUE] days, ",
                    "has just joined Amazon today, ",
                ],
                "current_points_balance": [
                    "has [VALUE] Amazon Points, ",
                    "has no Amazon Points, ",
                ],
                "is_cbcc": [
                    "has Amazon credit card, ",
                    "does not own any Amazon credit card, ",
                ],
            },
        },
    }
    agg_features = {
        "prefix": "In the last 360 days, the customer ",
        "visit": {
            "check_col": ["num_days_visited_360d", "did not visit Amazon at all, "],
            "data_cols": {
                "num_days_visited_360d": ["visited Amazon [VALUE] days, "] * 2,
                "days_since_last_visit": [
                    "last visited Amazon [VALUE] days ago, ",
                    "last visited Amazon today, ",
                ],
                "total_mobile_app_visits_360d": [
                    "visited [VALUE] times using mobile app, "
                ]
                * 2,
                "total_desktop_visits_360d": ["visited [VALUE] times using desktop, "]
                * 2,
                "nb_gl_session_30d": [
                    "viewed products from [VALUE] different categories, "
                ]
                * 2,
                "top20_frequent_gl_session": [
                    "viewed products from [VALUE] most frequently, "
                ]
                * 2,
            },
        },
        "purchase": {
            "check_col": ["total_ops_360d", "did not purchase from Amazon at all, "],
            "data_cols": {
                "days_since_last_order": [
                    "last purchased from Amazon [VALUE] days ago, ",
                    "last purchased from Amazon today, ",
                ],
                "total_order_days_360d": ["purchased from Amazon for [VALUE] days, "]
                * 2,
                "total_ops_360d": [
                    "spent [VALUE] JPY on Amazon, ",
                    "did not shop on Amazon, ",
                ],
                "total_promotion_amt_360d": ["received [VALUE] JPY in discount, "] * 2,
                "total_ops_per_day_360d": [
                    "spent on average [VALUE] JPY per day on days when they purchase, "
                ]
                * 2,
                "nb_gl_order_360d": ["purchased from [VALUE] different categories, "]
                * 2,
                "top20_frequent_gl_order_days": [
                    "purchased from [VALUE] most frequently, "
                ]
                * 2,
                "points_redeemed_360d": [
                    "used [VALUE] Amazon Points to shop, ",
                    "did not use any Amazon Points to shop, ",
                ],
            },
        },
        "xde": {
            "check_col": [],
            "data_cols": {
                "nb_xde_sign_flag_360d": [
                    "opted in to [VALUE] deal events, ",
                    "did not opt in to any deal event, ",
                ],
                "nb_xde_fulfill_flag_360d": [
                    "completed [VALUE] deal events, ",
                    "did not complete any deal event, ",
                ],
                "hve_total_ops_360d": [
                    "spent [VALUE] JPY during deal events, ",
                    "did not spent anything during deal events, ",
                ],
            },
        },
        "bbr_point": {
            "check_col": [
                "points_issued_bbr_360d",
                "did not receive any Amazon Points via completing marketing offers, ",
            ],
            "data_cols": {
                "days_points_issued_bbr_360d": [
                    "received Amazon Points for [VALUE] days via completing marketing offers, "
                ]
                * 2,
                "days_since_last_points_issued_bbr": [
                    "last received Amazon Points via completing marketing offers [VALUE] days ago, ",
                    "last received Amazon Points via completing marketing offers today, ",
                ],
                "points_issued_bbr_360d": [
                    "received [VALUE] Amazon Points via completing marketing offers, "
                ]
                * 2,
                "points_per_fulfilled_360d": [
                    "received on average [VALUE] Amazon Points per marketing offer completed, "
                ]
                * 2,
            },
        },
        "benefit": {
            "check_col": ["nb_benefit_360d", "did not use any Prime benefits, "],
            "data_cols": {
                "nb_benefit_360d": ["used [VALUE] Prime benefits, "] * 2,
                "most_frequent_benefit": [
                    "used [VALUE] the most out of all Prime benefits, "
                ]
                * 2,
            },
        },
        "wishlist": {
            "check_col": ["wishlist_ops_360d", "did not add any product to wishlist, "],
            "data_cols": {
                "days_since_last_wishlist": [
                    "last added products to wishlist [VALUE] days ago, ",
                    "last added products to wishlist today, ",
                ],
                "wishlist_ops_360d": [
                    "added [VALUE] JPY worth of products to wishlist, "
                ]
                * 2,
                "wishlist_asin_count_360d": ["added [VALUE] products to wishlist, "]
                * 2,
            },
        },
    }
    return snapshot_features, agg_features


def describe_numeric_distribution(df_numeric):
    description = ""
    feature_dict = feature_name_mapping()
    columns = list(df_numeric.columns)
    random.shuffle(columns)
    
    for column in columns:
        if (
            column in feature_dict
            and column != "outcome_fulfill_flag"
            and column != "click_any"
        ):
            column_desc = feature_dict[column]
            col_data = df_numeric[column]
            desc = col_data.describe()
            desc = col_data.describe()
            distribution = (
                f"'{column_desc}' has a mean value of {int(desc['mean']) if float(desc['mean']).is_integer() else desc['mean']:.1f} with a standard deviation of {int(desc['std']) if float(desc['std']).is_integer() else desc['std']:.1f}. "
                f"The minimum observed value is {int(desc['min']) if float(desc['min']).is_integer() else desc['min']:.1f}, while the maximum is {int(desc['max']) if float(desc['max']).is_integer() else desc['max']:.1f}. "
                f"Approximately 25% of the values are below {int(desc['25%']) if float(desc['25%']).is_integer() else desc['25%']:.1f}, the median (50th percentile) is {int(desc['50%']) if float(desc['50%']).is_integer() else desc['50%']:.1f}, "
                f"and 75% of the values fall below {int(desc['75%']) if float(desc['75%']).is_integer() else desc['75%']:.1f}. "
            )
            description += distribution
    return description


def feature_name_mapping():

    feature_name = {
        "outcome_fulfill_flag": "Flag indicating if customer completed the campaign.",
        "click_any": "Flag indicating if customer clicked any of the recommended brands.",
        "total_amazon_tenure": "Total tenure with Amazon.",
        "is_current_prime": "Indicator if the customer is currently a Prime member.",
        "is_cbcc": "Indicator if the customer has a co-branded credit card.",
        "most_frequent_benefit": "Most frequently used benefit by the customer.",
        "per_paid_session_30d": "Percentage of paid sessions in the last 30 days.",
        "nb_gl_order_360d": "Number of category purchased in the last 360 days.",
        "nb_gl_session_30d": "Number of category viewed in the last 30 days.",
        "nb_benefit_360d": "Number of benefits used in the last 360 days.",
        "total_ops_per_day_30d": "Total order product sales per day in the last 30 days.",
        "total_ops_per_day_360d": "Total order product sales per day in the last 360 days.",
        "days_since_last_order": "Days since the last order.",
        "total_order_days_30d": "Total order days in the last 30 days.",
        "total_order_days_360d": "Total order days in the last 360 days.",
        "total_ops_30d": "Total order product sales in the last 30 days.",
        "total_ops_360d": "Total order product sales in the last 360 days.",
        "per_redeem_30d": "Percentage of Points redeemed in the last 30 days.",
        "per_redeem_360d": "Percentage of Points redeemed in the last 360 days.",
        "points_per_signed_30d": "Points earned per opt-in to promotion in the last 30 days.",
        "points_per_signed_360d": "Points earned per opt-in to promotion in the last 360 days.",
        "points_per_fulfilled_30d": "Points earned per completion of promotion in the last 30 days.",
        "points_per_fulfilled_360d": "Points earned per completion of promotion in the last 360 days.",
        "current_points_balance": "Current balance of points for the customer.",
        "avg_current_points_balance_360d": "Average points balance over the last 360 days.",
        "days_since_last_points_redeemed": "Days since the last points were redeemed.",
        "days_since_last_points_issued": "Days since the last points were issued.",
        "days_since_last_points_issued_bbr": "Days since the last behavior based rewards points were issued.",
        "days_points_redeemed_30d": "Days points were redeemed in the last 30 days.",
        "days_points_redeemed_360d": "Days points were redeemed in the last 360 days.",
        "days_points_issued_30d": "Days points were issued in the last 30 days.",
        "days_points_issued_360d": "Days points were issued in the last 360 days.",
        "days_points_issued_bbr_30d": "Days behavior based rewards points were issued in the last 30 days.",
        "days_points_issued_bbr_360d": "Days behavior based rewards points were issued in the last 360 days.",
        "points_redeemed_30d": "Points redeemed in the last 30 days.",
        "points_redeemed_360d": "Points redeemed in the last 360 days.",
        "days_since_last_email_sent_count": "Days since the last email was sent.",
        "days_since_last_email_opened_count": "Days since the last email was opened.",
        "days_since_last_email_clicked_count": "Days since the last email was clicked.",
        "email_sent_count_30d": "Number of emails sent in the last 30 days.",
        "email_sent_count_360d": "Number of emails sent in the last 360 days.",
        "email_opened_count_30d": "Number of emails opened in the last 30 days.",
        "email_opened_count_360d": "Number of emails opened in the last 360 days.",
        "email_clicked_count_30d": "Number of emails clicked in the last 30 days.",
        "email_clicked_count_360d": "Number of emails clicked in the last 360 days.",
        "days_since_last_total_view_count": "Days since the last total view count.",
        "days_since_last_smartphone_view_count": "Days since the last smartphone view count.",
        "days_since_last_desktop_view_count": "Days since the last desktop view count.",
        "total_view_count_30d": "Total views in the last 30 days.",
        "total_view_count_360d": "Total views in the last 360 days.",
        "smartphone_view_count_30d": "Smartphone views in the last 30 days.",
        "smartphone_view_count_360d": "Smartphone views in the last 360 days.",
        "desktop_view_count_30d": "Desktop views in the last 30 days.",
        "desktop_view_count_360d": "Desktop views in the last 360 days.",
        "per_promotion_30d": "Percentage of promotions/discounts of total order product sales in the last 30 days.",
        "per_promotion_360d": "Percentage of promotions/discounts of total order product sales in the last 360 days.",
        "per_fulfill_30d": "Percentage of promotions completed over opted in in the last 30 days.",
        "per_fulfill_360d": "Percentage of promotions completed over opted in in the last 360 days.",
        "nb_sign_flag_30d": "Number of promotions opted in in the last 30 days.",
        "nb_sign_flag_360d": "Number of promotions opted in in the last 360 days.",
        "nb_fulfill_flag_30d": "Number of promotions completed in the last 30 days.",
        "nb_fulfill_flag_360d": "Number of promotions completed in the last 360 days.",
        "nb_xde_sign_flag_360d": "Number of opt-ins to future dated event or monthly deal event in the last 360 days.",
        "nb_xde_fulfill_flag_360d": "Number of completions to future dated event or monthly deal event in the last 360 days.",
        "days_since_last_asins_added_to_cart": "Days since the last amazon standard identification numbers were added to the cart.",
        "has_asins_in_cart": "Indicator if there are amazon standard identification numbers in the cart.",
        "asins_ordered_added_ratio_30d": "Ratio of amazon standard identification numbers ordered to added in the last 30 days.",
        "asins_ordered_added_ratio_360d": "Ratio of amazon standard identification numbers ordered to added in the last 360 days.",
        "days_since_last_wishlist": "Days since the last wishlist interaction.",
        "wishlist_asin_count_30d": "Number of amazon standard identification numbers in the wishlist in the last 30 days.",
        "wishlist_asin_count_360d": "Number of amazon standard identification numbers in the wishlist in the last 360 days.",
        "wishlist_ops_30d": "Wishlist operations in the last 30 days.",
        "wishlist_ops_360d": "Wishlist operations in the last 360 days.",
        "cnt_payment_methods": "Count of payment methods used.",
        "cnt_credit_cards": "Count of credit cards used.",
        "days_since_last_visit": "Days since the last visit.",
        "num_days_visited_30d": "Number of days visited in the last 30 days.",
        "num_days_visited_360d": "Number of days visited in the last 360 days.",
        "total_visits_30d": "Total visits in the last 30 days.",
        "total_visits_360d": "Total visits in the last 360 days.",
        "total_desktop_visits_30d": "Total desktop visits in the last 30 days.",
        "total_desktop_visits_360d": "Total desktop visits in the last 360 days.",
        "total_mobile_app_visits_30d": "Total mobile app visits in the last 30 days.",
        "total_mobile_app_visits_360d": "Total mobile app visits in the last 360 days.",
        "total_mobile_web_visits_30d": "Total mobile web visits in the last 30 days.",
        "total_mobile_web_visits_360d": "Total mobile web visits in the last 360 days.",
        "gl_14_order_days_360d": "Previous order of Books in the last 360 days.",
        "gl_21_order_days_360d": "Previous order of Toys in the last 360 days.",
        "gl_121_order_days_360d": "Previous order of Drug Store items in the last 360 days.",
        "gl_23_order_days_360d": "Previous order of Electronics in the last 360 days.",
        "gl_194_order_days_360d": "Previous order of Beauty products in the last 360 days.",
        "gl_201_order_days_360d": "Previous order of Home items in the last 360 days.",
        "gl_79_order_days_360d": "Previous order of Kitchen items in the last 360 days.",
        "gl_107_order_days_360d": "Previous order of Wireless products in the last 360 days.",
        "gl_193_order_days_360d": "Previous order of Apparel in the last 360 days.",
        "gl_200_order_days_360d": "Previous order of Sports items in the last 360 days.",
        "gl_325_order_days_360d": "Previous order of Grocery items in the last 360 days.",
        "gl_60_order_days_360d": "Previous order of Home Improvement items in the last 360 days.",
        "gl_405_order_days_360d": "Previous order of Mobile Apps in the last 360 days.",
        "gl_63_order_days_360d": "Previous order of Video Games in the last 360 days.",
        "gl_147_order_days_360d": "Previous order of PC products in the last 360 days.",
        "gl_263_order_days_360d": "Previous order of Automotive products in the last 360 days.",
        "gl_309_order_days_360d": "Previous order of Shoes in the last 360 days.",
        "gl_351_order_days_360d": "Previous order of E-books in the last 360 days.",
        "gl_15_order_days_360d": "Previous order of Music in the last 360 days.",
        "gl_229_order_days_360d": "Previous order of Office Products in the last 360 days.",
        "gl_others": "Previous view of Others in the last 360 days.",
        "gl_14_session_30d": "Previous view of Books in the last 30 days.",
        "gl_21_session_30d": "Previous view of Toys in the last 30 days.",
        "gl_23_session_30d": "Previous view of Electronics in the last 30 days.",
        "gl_351_session_30d": "Previous view of E-books in the last 30 days.",
        "gl_79_session_30d": "Previous view of Kitchen items in the last 30 days.",
        "gl_121_session_30d": "Previous view of Drug Store items in the last 30 days.",
        "gl_60_session_30d": "Previous view of Home Improvement items in the last 30 days.",
        "gl_201_session_30d": "Previous view of Home items in the last 30 days.",
        "gl_194_session_30d": "Previous view of Beauty products in the last 30 days.",
        "gl_193_session_30d": "Previous view of Apparel in the last 30 days.",
        "gl_147_session_30d": "Previous view of PC products in the last 30 days.",
        "gl_107_session_30d": "Previous view of Wireless products in the last 30 days.",
        "gl_325_session_30d": "Previous view of Grocery items in the last 30 days.",
        "gl_200_session_30d": "Previous view of Sports items in the last 30 days.",
        "gl_63_session_30d": "Previous view of Video Games in the last 30 days.",
        "gl_15_session_30d": "Previous view of Music in the last 30 days.",
        "gl_309_session_30d": "Previous view of Shoes in the last 30 days.",
        "gl_263_session_30d": "Previous view of Automotive products in the last 30 days.",
        "gl_74_session_30d": "Previous view of DVD items in the last 30 days.",
        "prime_video_days_360d": "Previous usage of Prime Video in the last 360 days.",
        "day1_ship_days_360d": "Previous usage of Prime 1-day Shipping in the last 360 days.",
        "cloud_drive_app_days_360d": "Previous usage of Cloud Drive App in the last 360 days.",
        "hawkfire_days_360d": "Previous usage of Prime Music Unlimited in the last 360 days.",
        "prime_music_days_360d": "Previous usage of Prime Music in the last 360 days.",
        "day2_ship_days_360d": "Previous usage of Prime 2-day Shipping in the last 360 days.",
        "day3_5_ship_days_360d": "Previous usage of Prime 3-to-5-day Shipping in the last 360 days.",
        "cloud_drive_days_360d": "Previous usage of Cloud Drive in the last 360 days.",
        "non_physical_days_360d": "Previous usage of Digital Goods in the last 360 days.",
        "amazon_channels_days_360d": "Previous usage of Amazon Channel in the last 360 days.",
        "prime_reading_days_360d": "Previous usage of Prime Reading in the last 360 days.",
        "sns_orders_days_360d": "Previous usage of Subscribe and Save in the last 360 days.",
        "view_history": "View History",
    }

    return feature_name


def get_features(df, mode="cls"):
    snapshot_features, agg_features = human_template()

    column_names = list(df.columns)
    features = [f for f in column_names if f in feature_name_mapping()]

    df_numerical = df[features].select_dtypes(include=[np.number])
    textual_distribution = describe_numeric_distribution(
        df_numerical
    )
    ds = []
    for i, customer_row in tqdm(df.iterrows()):
        snapshot_text = generate_customer_text(
            customer_row, snapshot_features, shuffle_data_cols=False
        )
        agg_text = generate_customer_text(
            customer_row, agg_features, shuffle_data_cols=False
        )

        customer_value = [customer_row[features[k]] for k in range(len(features))]
        customer_value = [
            (
                f"{int(float(k))}"
                if isinstance(k, (int, float)) and float(k).is_integer()
                else f"{float(k):.2f}" if isinstance(k, (int, float)) else str(k)
            )
            for k in customer_value
        ]
        numerical_description = (
            ". ".join(
                [
                    str(feature_name_mapping()[features[k]]).replace(".", "")
                    + " is "
                    + customer_value[k]
                    for k in range(len(features))
                ]
            )
            + ". "
        )

        if mode == "cls":
            d = {
                "id": int(i),
                "human_template": snapshot_text + agg_text,
                "numerical_template": numerical_description,
                "label": customer_row["outcome_fulfill_flag"],
                "textual_distribution": textual_distribution,
            }
            ds.append(d)
        elif mode == "rec":
            brands = customer_row["pred_called"].split("|")
            d = {
                "id": int(i),
                "human_template": snapshot_text + agg_text,
                "numerical_template": numerical_description,
                "label": ", ".join(brands) + ".",
                "textual_distribution": textual_distribution,
                "click": int(customer_row["click_any"]),
            }
            ds.append(d)
    return ds


class LLMRec_Dataloader:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.train_path = args.train_path
        self.valid_path = args.valid_path
        self.test_path = args.test_path
        self.train_size = args.train_size
        self.valid_size = args.valid_size
        self.test_size = args.test_size
        self.task_name = args.task_name
        self.method = args.method
        self.ablation = args.ablation
        self.strategy = args.strategy

    def load_data(self, args):
        print("*" * 50)
        self.shuffler = random.Random(args.seed)

        if self.method == "ICL":
            if os.path.exists(f"{args.pred_path}{args.file_path}.pkl"):
                test_path = f"{args.pred_path}{args.file_path}.pkl"
                test = load_pkl(test_path)
                print(
                    f"loading test data from predictions: {args.pred_path}{args.file_path}.pkl"
                )
            else:
                test_path = f"{self.data_dir}{self.task_name}_test.pkl"
                if os.path.exists(test_path):
                    test = load_pkl(test_path)
                    print(f"loading pkl of test data: {test_path}")
                else:
                    print(f"creating {self.task_name} pkl of test data...")
                    test = self.get_data(self.test_path, test_path)
                    print(f"pkl file of test data created: {test_path}")

            train_path = f"{self.data_dir}{self.task_name}_train.pkl"
            if os.path.exists(train_path):
                train = load_pkl(train_path)
                print(f"loading pkl of train data: {train_path}")
            else:
                print(f"creating {self.task_name} pkl of train data...")
                train = self.get_data(self.train_path, train_path)
                print(f"pkl file of train data created: {train_path}")
            return test, test_path, train, train_path

        elif self.method == "FT":
            train_path = f"{self.data_dir}{self.task_name}_train.pkl"
            if os.path.exists(train_path):
                train = load_pkl(train_path)
                print(f"loading pkl of train data: {train_path}")
            else:
                print(f"creating {self.task_name} pkl of train data...")
                train = self.get_data(self.train_path, train_path)
                print(f"pkl file of train data created: {train_path}")

            valid_path = f"{self.data_dir}{self.task_name}_valid.pkl"
            if os.path.exists(valid_path):
                valid = load_pkl(valid_path)
                print(f"loading pkl of valid data: {valid_path}")
            else:
                print(f"creating {self.task_name} pkl of valid data...")
                valid = self.get_data(self.valid_path, valid_path)
                print(f"pkl file of valid data created: {valid_path}")
            return train, valid, train_path, valid_path
        print("*" * 50)

    def get_FT_data(self, data_path, mode):
        data = load_pkl(data_path)
        # for cls we use normal FT dataset, for rec we use the KTO dataset
        if "cls" in self.task_name:
            system_text = cls_template()["start_template"]
            new_data = []
            for i in range(len(data)):
                label = label2text(data[i]["label"])
                if self.strategy == "human":
                    text = data[i]["human_template"]
                else:
                    text = data[i]["adaptive_template"]

                new_dict = {"instruction": system_text, "input": text, "output": label}
                new_data.append(new_dict)

        elif "rec" in self.task_name:
            system_text = rec_template()["start_template"]
            new_data = []
            print(data[0])
            for i in range(len(data)):
                label = data[i]["label"]
                click = click2bool(
                    data[i]["click"]
                )  # Note: label needs to be of type bool, not str.
                if self.strategy == "human":
                    text = data[i]["human_template"]
                else:
                    text = data[i]["adaptive_template"]

                new_dict = {
                    "system": system_text,
                    "query": text,
                    "response": label,
                    "label": click,
                }
                new_data.append(new_dict)

        save_path = f"{self.data_dir}{self.task_name}_FT_{mode}.json"
        save_json(new_data, save_path)
        return new_data

    def get_data(self, data_path, save_path):
        df = pd.read_feather(data_path)
        data = get_features(df, mode=self.task_name)
        self.shuffler.shuffle(data)
        save_pkl(data, save_path)
        return data
