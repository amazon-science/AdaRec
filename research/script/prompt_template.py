import re


def get_brand_description():
    brand_dict = {
        "brand1": "Automotive supplies specializing in small car accessories",
        "brand2": "Electronics specializing in PC monitors",
        "brand3": "Grocery beverages offering tea and coffee products for this promotion",
        "brand4": "Baby products and accessories",
        "brand5": "Home improvement tools including power drills",
        "brand6": "Home entertainment electronics specializing in large TVs",
        "brand7": "Health and personal care products including Biore face creams",
        "brand8": "Toys specializing in LEGO building sets",
        "brand9": "Beauty products specializing in cosmetics",
        "brand10": "Pet supplies including food and accessories",
        "brand11": "Smart home lighting featuring Philips Hue products",
        "brand12": "Home entertainment electronics specializing in large TVs",
        "brand13": "Sports equipment specializing in swimming gear",
        "brand14": "Home entertainment electronics specializing in large TVs",
        "brand15": "Kitchen appliances including rice cookers",
        "brand16": "Optical equipment specializing in telescopes",
        "brand17": "Home furniture including bookshelves and cupboards",
    }
    return brand_dict


def cls_template():
    start_template = "As the Senior Marketing Manager at Amazon Japan, your primary responsibility is analyzing deal events - 3-day promotion periods that occur about once a month. These deals typically offer customers up to 10% back in Amazon Points when they meet spending conditions. Amazon Points are cash equivalents that customers can use for Amazon purchases at a rate of 1 Point = 1 JPY. Based on a customer profile, estimate whether the customer will accept or refuse this offer."
    end_template = "Based on the information above, please answer with either 'accept' or 'refuse', indicating whether they will accept or not. Additionally, provide your confidence score as a percentage. Return the answer, confidence, and reason in the following json format without extra text: {'answer': answer, 'confidence': confidence, 'reason': reason}: "
    return {"start_template": start_template, "end_template": end_template}


def rec_template():
    brand_dict = get_brand_description()
    brand_description = "\n".join(
        f"{key}: {value}" for key, value in brand_dict.items()
    )

    start_template = f"As the Senior Marketing Manager at Amazon Japan, your task is to recommend three brands for the promotional carousel. The promotion has two components: 1. Condition: Customers must purchase X units 2. Reward: Customers receive 10% in Amazon Points. \nAvailable brands: {brand_description}. Based on a customer profile, please recommend three brand names for the customer."
    end_template = "Based on the information above, please recommend three brand names for the promotional carousel. Additionally, provide your confidence score as a percentage. Return the brands, confidence score, and reason for your choices in the following JSON format, without any extra text: {'brand': 'brand1, brand2, brand3', 'confidence': confidence, 'reason': reason}: "

    return {"start_template": start_template, "end_template": end_template}
