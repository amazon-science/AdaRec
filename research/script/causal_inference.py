import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz
from contextlib import contextmanager
import signal
import warnings
import time
from script.utils import *


def feature_name_mapping():

    feature_name = {
        "outcome_fulfill_flag": "Flag indicating if customer completed the campaign",
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


def get_brand_description():
    brand_dict = {
        "amon": "Automotive supplies",
        "benq": "PC",
        "cocacola": "Beverages",
        "combi": "Baby products",
        "hikoki": "Home improvement tools",
        "hisense": "Home entertainment electronics",
        "kao": "Health and personal care products",
        "lego": "Toys",
        "loreal": "Beauty products",
        "nestlepurinapetcare": "Pet supplies",
        "philips": "Smart home lighting",
        "regza": "Home entertainment electronics",
        "swans": "Sports equipment",
        "tcl": "Home entertainment electronics",
        "toshiba": "Kitchen appliances",
        "vixen": "Optical equipment",
        "yamazen": "Home furniture",
    }
    return brand_dict



@contextmanager
def timeout(seconds):
    def handler(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class causal_discovery:
    def __init__(
        self,
        args,
        n_neighbors_mi=2000,
        n_neighbors_fci=1000,
        top_k=15,
        max_features=30,
        timeout_seconds=10,
    ):
        self.seed = args.seed
        self.n_neighbors_mi = n_neighbors_mi
        self.n_neighbors_fci = n_neighbors_fci
        self.top_k = top_k
        self.max_features = max_features
        self.timeout_seconds = timeout_seconds
        self.n_shot = args.n_shot
        self.task_name = args.task_name

    def select_top_features(self, data, target, feature_cols, method="mutual_info"):
        """Select most relevant features using mutual information or correlation."""
        if method == "mutual_info":
            mi_scores = mutual_info_classif(data[feature_cols], data[target])
            selected_features = [
                x for _, x in sorted(zip(mi_scores, feature_cols), reverse=True)
            ][: self.max_features]
        else: 
            correlations = [abs(data[col].corr(data[target])) for col in feature_cols]
            selected_features = [
                x for _, x in sorted(zip(correlations, feature_cols), reverse=True)
            ][: self.max_features]
        return selected_features

    def preprocess_features(self, data):
        brand_category = get_brand_description()
        entries_dict = {}
        all_features = set()

        for entry in data:
            entry_id = entry["id"]
            pairs = [
                pair.strip()
                for pair in entry["numerical_template"].split(".")
                if pair.strip()
            ]
            entries_dict[entry_id] = {}

            for pair in pairs:
                pair = pair.rsplit(" is ", 1)
                if len(pair) == 2:
                    try: 
                        feature, value = pair
                        entries_dict[entry_id][feature] = int(value)
                        all_features.add(feature)
                    except:
                        continue
            if self.task_name == 'cls':
                entries_dict[entry_id]['label'] = cls_n2t(entry['label'])
            elif self.task_name == 'rec':
                label = entry['label'].replace(".", "").replace(" ", "").split(",")
                label = ", ".join([brand_category[k] for k in label])
                entries_dict[entry_id]['label'] = label

        df = pd.DataFrame.from_dict(entries_dict, orient="index")
        return df

    def find_nearest_neighbors_cosine(self, query_points, reference_points, k):
        start_time = time.time()
        query_norm = np.linalg.norm(query_points, axis=1)[:, np.newaxis]
        ref_norm = np.linalg.norm(reference_points, axis=1)[np.newaxis, :]

        query_norm = np.maximum(query_norm, 1e-10)
        ref_norm = np.maximum(ref_norm, 1e-10)

        similarities = np.dot(query_points, reference_points.T) / (
            query_norm * ref_norm
        )

        k = min(k, similarities.shape[1])
        indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
        similarities = np.take_along_axis(similarities, indices, axis=1)

        print(f"Neighbor finding took {time.time() - start_time:.2f} seconds")
        return similarities, indices

    def run_fci_with_timeout(self, X, feature_names):
        """Run FCI with timeout and error handling."""
        start_time = time.time()
        try:
            with timeout(self.timeout_seconds):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    G, edges = fci(X, fisherz, 0.1)
                print(f"FCI completed in {time.time() - start_time:.2f} seconds")
                return G, edges
        except (TimeoutError, Exception) as e:
            print(f"FCI failed after {time.time() - start_time:.2f} seconds: {str(e)}")
            return None, None

    def find_important_features_by_id(self, test_df, test, train_df, test_path, args):
        
        overall_start_time = time.time()
        print(f"Starting processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        ids_to_process = set()
        for i in range(len(test)):
            if "causal_text" not in test[i] or 'similar_text' not in test[i]:
                ids_to_process.add(test[i]["id"])
            elif len(test[i]['causal_text'].split('>')) < 11:
                ids_to_process.add(test[i]["id"])

        if not ids_to_process:
            print("*" * 50, "All records already have causal texts", "*" * 50)
            return test

        print(f"Total samples to process: {len(ids_to_process)}")

        test_df = test_df[test_df.index.isin(ids_to_process)]
        
        print("Starting data scaling...")
        scaling_start = time.time()
        numeric_columns = train_df.select_dtypes(include=['int64', 'float64']).columns
        brand_columns = [n_col for n_col in numeric_columns if 'Previous view of' in n_col or 'Previous order of' in n_col]
        numeric_columns = [n_col for n_col in numeric_columns if 'Previous view of' not in n_col and 'Previous order of' not in n_col]
        
        scaler = MinMaxScaler()
        train_df_scaled = pd.DataFrame(
            scaler.fit_transform(train_df[numeric_columns]),
            columns=numeric_columns,
            index=train_df.index,
        )
        test_df_scaled = pd.DataFrame(
            scaler.transform(test_df[numeric_columns]), columns=numeric_columns, index=test_df.index
        )
        print(f"Data scaling completed in {time.time() - scaling_start:.2f} seconds")

        train_array = train_df_scaled.values
        test_array = test_df_scaled.values

        important_features = {}
        similar_cases = {}
        target_col = (
            "Flag indicating if customer completed the campaign"
            if args.task_name == "cls"
            else "Flag indicating if customer clicked any of the recommended brands"
        )

        total_samples = len(test_df)
        for idx, current_id in enumerate(test_df.index):
            sample_start_time = time.time()
            print(f"\nProcessing sample {idx+1}/{total_samples} (ID: {current_id})")
            print(f"Time elapsed: {(time.time() - overall_start_time)/3600:.2f} hours")

            print("Finding neighbors for mutual info...")
            similarities_mi, indices_mi = self.find_nearest_neighbors_cosine(
                test_array[idx : idx + 1], train_array, self.n_neighbors_mi
            )

            similar_indices_mi = indices_mi[0]
            similar_df_mi = train_df[numeric_columns].iloc[similar_indices_mi]

            feature_cols = [col for col in similar_df_mi.columns if col != target_col]
            
            print("Selecting top features...")
            feature_start = time.time()
            selected_features = self.select_top_features(
                similar_df_mi, target_col, feature_cols, method="mutual_info"
            )
            print(f"Feature selection took {time.time() - feature_start:.2f} seconds")

            mi_ranked_features = selected_features.copy()

            print("Finding neighbors for FCI...")
            similarities_fci, indices_fci = self.find_nearest_neighbors_cosine(
                test_array[idx : idx + 1], train_array, self.n_neighbors_fci
            )

            similar_df_fci = train_df[numeric_columns].iloc[indices_fci[0]]

            selected_features_with_target = selected_features + [target_col]
            X = similar_df_fci[selected_features_with_target].values

            print("Running FCI...")
            G, edges = self.run_fci_with_timeout(X, selected_features_with_target)

            if edges is not None:
                target_idx = len(selected_features)
                feature_importance = {}

                for i, feature in enumerate(selected_features):
                    causal_importance = sum(
                        1
                        for edge in edges
                        if (
                            (i == edge.get_node1() and target_idx == edge.get_node2())
                            or (
                                i == edge.get_node2() and target_idx == edge.get_node1()
                            )
                        )
                        and (
                            edge.get_endpoint1() in [2, 3]
                            or edge.get_endpoint2() in [2, 3]
                        )
                    )
                    feature_importance[feature] = causal_importance

                sorted_features = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )

                causal_features = [
                    feature[0] for feature in sorted_features if feature[1] > 0
                ]

                if len(causal_features) < self.top_k:
                    remaining_features = [
                        f for f in mi_ranked_features if f not in causal_features
                    ]
                    needed_features = self.top_k - len(causal_features)
                    causal_features.extend(remaining_features[:needed_features])

                top_features = causal_features[: self.top_k]
            else:
                print("FCI failed, using mutual info ranking")
                top_features = mi_ranked_features[: self.top_k]

            
            
            n_example = int(args.n_shot.split("_")[0])

            similar_case = train_df.iloc[indices_fci[0]][:n_example] 
            
            case_descriptions = []
            for idx in range(n_example):
                case_answer = similar_case.iloc[idx]['label'] 
                

                row_descriptions = [
                    f"{feature} is {similar_case.iloc[idx][feature]}"
                    for feature in top_features
                ]
                if self.task_name == 'rec':
                    row_descriptions += [
                        f"{feature} is {similar_case.iloc[idx][feature]}" 
                        for feature in brand_columns 
                        if feature in similar_case.iloc[idx]  
                        and int(similar_case.iloc[idx][feature]) != 0  
                        ]
                    case_description = "Customer Profile: "+", ".join(row_descriptions) + "." + "Preferred Categories: " + case_answer + '.'
                else:
                    case_description = "Customer Profile: "+", ".join(row_descriptions) + "." + "Answer: " + case_answer + '.'
                case_descriptions.append(case_description)
                
            case_descriptions = '\n'.join(case_descriptions)
            similar_cases[current_id] = case_descriptions
            important_features[current_id] = " > ".join(top_features) + "."


            sample_time = time.time() - sample_start_time
            print(f"Sample completed in {sample_time:.2f} seconds")
            print(
                f"Estimated time remaining: {(sample_time * (total_samples-idx-1))/3600:.2f} hours"
            )
            print(f"Selected features: {important_features[current_id]}")
            print(f"Similar cases found: {similar_cases[current_id]}")

        for i in range(len(test)):
            customer_id = test[i]["id"]
            if customer_id in ids_to_process:
                test[i]["causal_text"] = important_features[customer_id]
                test[i]["similar_text"] = similar_cases[customer_id]

        total_time = time.time() - overall_start_time
        print(f"\nTotal processing time: {total_time/3600:.2f} hours")
        print(f"Average time per sample: {total_time/total_samples:.2f} seconds")

        save_pkl(test, test_path)
        print(f"save causal texts & similar texts to {test_path}")
        return test
