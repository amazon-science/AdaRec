from tqdm import trange
from script.dataloader import *
from script.inference import *
import os
import random
from script.finetune_llama import *
from datetime import datetime
from script.causal_inference import *
from script.utils import *
import numpy as np
import re


def run_adaptive_prompting(data_size, data, model, save_path, args, mode):
    adaptive_files = []

    for i in trange(data_size, desc="[Inference] get adaptive templates: "):
        if "adaptive_template" in data[i].keys():
            if data[i]["adaptive_template"] != "":
                start_ = data[i]["adaptive_template"].find(":")
                data[i]["adaptive_template"] = data[i]["adaptive_template"][
                    start_ + 1 :
                ].replace("\n", " ")
                print("*" * 50)
                print(data[i]["adaptive_template"])
                print("*" * 50)
                print(f"{i} already has the adaptive_template")
                continue

        adaptive_template = (
            "You are a customer profile generator. Below is the data distribution for each feature: "
            + "\n"
            + data[i]["textual_distribution"]
            + "\nUsing this information, generate a clear and cohesive profile for the customer. For non-numerical features, emphasize the specific values. For numerical features, describe the customer's data relative to overall trends without using exact numbers. Present all available customer data as a single, fluid paragraph without any extra formatting. Customer Profile: "
        )

        phrase1 = r"Flag indicating if customer clicked any of the recommended brands is [^.]+"
        phrase2 = r"Flag indicating if customer completed the campaign \(purchased 10k JPY or more during day-of period\) is [^.]+"

        numerical_template = re.sub(phrase1 + r"\.", "", data[i]["numerical_template"])
        numerical_template = re.sub(phrase2 + r"\.", "", numerical_template)

        customer_profile = model.get_adaptive_template(
            adaptive_template, numerical_template
        ).replace("\n", "")
        start_index = (
            customer_profile.lower().find("profile:") + 8
            if customer_profile.lower().find("profile:") != -1
            else 0
        )
        if start_index == 0:
            start_index = (
                customer_profile.lower().find(":") + 1
                if customer_profile.lower().find(":") != -1
                else 0
            )
        else:
            customer_profile = customer_profile[start_index:]
        data[i]["adaptive_template"] = customer_profile
        save_pkl(data, save_path)
        print(f"save adaptive template to {save_path}")
        if i > 0 and i % 100 == 0:
            save_pkl(data, f"{args.data_dir}{args.task_name}_{mode}_{i}.pkl")
            print(
                f"save adaptive template to {args.data_dir}{args.task_name}_{mode}_{i}.pkl"
            )
    print("*" * 50, "Adaptive Prompting Done", "*" * 50)
    return data


def run(args, log):
    log.logger.info(f"run_{args.task_name}_{args.method}")
    dataloder = LLMRec_Dataloader(args)

    if args.method == "ICL":

        test, test_path, train, train_path = dataloder.load_data(args)
        args.test_size = min(args.test_size, len(test))
        args.train_size = min(args.train_size, len(train))
        test = test[: args.test_size]
        train = train[: args.train_size]

        model = LLM(args)
        model.init_bedrock()
        model.init_template(args)

        if args.strategy == "human":
            if "causal" in args.ablation:
                    dag = causal_discovery(args)
                    test_df = dag.preprocess_features(test)
                    train_df = dag.preprocess_features(train)
                    test = dag.find_important_features_by_id(
                        test_df, test, train_df, test_path, args
                    ) 

        elif args.strategy == "adapt":

            if len([d for d in test if "adaptive_template" in d]) < args.test_size:
                model.init_template_model(args)
                test = run_adaptive_prompting(
                    args.test_size, test, model, test_path, args, "test"
                ) 

            if "causal" in args.ablation:
                    dag = causal_discovery(args)
                    test_df = dag.preprocess_features(test)
                    train_df = dag.preprocess_features(train)
                    test = dag.find_important_features_by_id(
                        test_df, test, train_df, test_path, args
                    )  

        model.init_inference_model(args)
        if args.evaluate:
            model.load_inference_model(args.save_model)

        elif (
            "claude" not in args.inference_model
            and "meta.llama3" not in args.inference_model
        ):
            model.load_inference_model()

        for i in trange(args.test_size, desc="[Inference:] get LLM predictions"):
            if "prediction" in test[i].keys():
                if test[i]["prediction"] != "":
                    print(f"{i} already has predictions")
                    continue

            if args.evaluate:
                prediction, rationale, confidence, text_prompt = model.get_ft_results(
                    test[i], args
                )
            elif "claude" not in args.inference_model and "meta.llama3" not in args.inference_model:
                prediction, rationale, confidence, text_prompt = model.get_ft_results(
                    test[i], args
                )
            else:
                prediction, rationale, confidence, text_prompt = model.get_results(
                    test[i], args
                )
            test[i]["prediction"] = prediction
            test[i]["rationale"] = rationale
            test[i]["confidence"] = confidence
            test[i]["text_prompt"] = text_prompt
            save_pkl(test, f"{args.pred_path}{args.file_path}.pkl")
            print(f"save predictions to {args.pred_path}{args.file_path}.pkl")

    elif args.method == "FT":

        train, valid, train_path, valid_path = dataloder.load_data(args)
        args.valid_size = min(args.valid_size, len(valid))
        args.train_size = min(args.train_size, len(train))
        valid = valid[: args.valid_size]
        train = train[: args.train_size]
        model = Llama_FT(
            data_dir=args.data_dir,
            task_name=args.task_name,
            save_path=args.save_model,
            model_id=args.inference_model,
            train_path=train_path,
            valid_path=valid_path,
            max_output=args.max_output,
            temperature=args.temperature,
        )
        if args.strategy == "adapt":
            if (
                len([t for t in train if "adaptive_template" in t]) < args.train_size
                or len([v for v in valid if "adaptive_template" in v]) < args.valid_size
            ):
                print(len([t for t in train if "adaptive_template" in t]))
                print(len([v for v in valid if "adaptive_template" in v]))
                model.init_template_model(args)
                train = run_adaptive_prompting(
                    args.train_size, train, model, train_path, args, "train"
                )  
                valid = run_adaptive_prompting(
                    args.valid_size, valid, model, valid_path, args, "valid"
                )  

        train = dataloder.get_FT_data(train_path, "train")
        valid = dataloder.get_FT_data(valid_path, "valid")

        if args.task_name == "cls":
            model.load_model()
            model.run(args)
            model.save_model()
        else:
            model.run(args)
