import os
import argparse
import logging
import time
from script.run import run
import os
from script.utils import *
from script.evaluate import get_report
from sklearn.metrics import classification_report
from datetime import datetime
import os
import glob


class Logger(object):
    def __init__(self, filename, level="info"):
        level = logging.INFO if level == "info" else logging.DEBUG
        self.logger = logging.getLogger(filename)
        self.logger.propagate = False
        self.logger.setLevel(level)
        th = logging.FileHandler(filename, "w")
        self.logger.addHandler(th)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/", type=str, help="Path for data")
    parser.add_argument(
        "--train_path",
        default="t1_10_train.feather",
        type=str,
        help="File path for training data: mde6_fulfill_train.feather / t1_10_train.feather",
    )
    parser.add_argument(
        "--valid_path",
        default="t1_10_valid.feather",
        type=str,
        help="File path for valid data: mde6_fulfill_valid.feather / t1_10_valid.feather",
    )
    parser.add_argument(
        "--test_path",
        default="t1_10_test.feather",
        type=str,
        help="File path for test data: mde6_fulfill_test.feather / t1_10_test.feather",
    )
    parser.add_argument(
        "--train_size",
        default=5000,
        type=int,
        help="size of training set",
    )
    parser.add_argument(
        "--valid_size",
        default=600,
        type=int,
        help="size of valid set",
    )
    parser.add_argument(
        "--test_size",
        default=600,
        type=int,
        help="size of test set",
    )
    parser.add_argument(
        "--inference_model",
        default="qwen2_5_32b_instruct",
        type=str,
        help="You can select from 'anthropic.claude-3-5-sonnet-20240620-v1:0','llama3_1_70b_instruct_gptq_int4','qwen2_5_32b_instruct'",
    )
    parser.add_argument(
        "--template_model",
        default="qwen2_5_32b_instruct",
        type=str,
        help="You can select from 'anthropic.claude-3-5-sonnet-20240620-v1:0','llama3_1_70b_instruct_gptq_int4','qwen2_5_32b_instruct'",
    )
    parser.add_argument(
        "--embedding_model",
        default="Ceceliachenen/gte-large-en-v1.5",
        type=str,
        help="You can select from 'Ceceliachenen/gte-large-en-v1.5'",
    )
    parser.add_argument(
        "--evaluate",
        default=False,
        type=bool,
        help="Whether only evaluate FT model or not.",
    )
    parser.add_argument(
        "--task_name",
        default="rec",
        type=str,
        help="task to test: cls/rec/cross-cls/cross-rec",
    )
    parser.add_argument(
        "--method", 
        default="ICL", 
        type=str, 
        help="method: ICL/FT"
    )
    parser.add_argument(
        "--strategy",
        default="adapt",
        type=str,
        help="method: human/adapt/",
    )
    parser.add_argument(
        "--ablation",
        default="causalsimi",
        type=str,
        help="module to input in the inference: simi/causal",
    )
    parser.add_argument(
        "--n_shot",
        default="5_shot",
        type=str,
        help="n_shot: 0_shot/1_shot/3_shot/5_shot",
    )
    parser.add_argument(
        "--max_output", type=int, default=1000, help="Max output token of LLMs"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature of LLMs"
    )
    parser.add_argument(
        "--log_dir", default="log/", type=str, help="Path for Logging file"
    )
    parser.add_argument(
        "--save_dir", default="save_model/", type=str, help="Path for saving model"
    )

    parser.add_argument("--train_ratio", default=0.8, type=float, help="train ratio.")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size.")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--eval_step", type=int, default=100, help="Evaluation step")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")

    args = parser.parse_args()
    return args


def main():
    args = parse_argument()
    if 'cls' in args.task_name:
        args.train_path = 'mde6_fulfill_train.feather'
        args.valid_path = 'mde6_fulfill_valid.feather'
        args.test_path = 'mde6_fulfill_test.feather'
    elif 'rec' in args.task_name:
        args.train_path = 't1_10_train.feather'
        args.valid_path = 't1_10_valid.feather'
        args.test_path = 't1_10_test.feather'
        
    if args.ablation == 'none' or args.ablation == 'causal':
        args.n_shot = '0_shot'
        
        
    if args.task_name == 'cross-cls':
        args.task_name ='cls'
        args.ft_name = 'rec'
    elif args.task_name == 'cross-rec':
        args.task_name = 'rec'
        args.ft_name = 'cls'
    else:
        args.ft_name = args.task_name
        
    

    if "llama" in args.template_model.lower():
        args.data_dir = args.data_dir + "llama_template/"
    elif "claude" in args.template_model.lower():
        args.data_dir = args.data_dir + "claude_template/"
    elif "qwen" in args.template_model.lower():
        args.data_dir = args.data_dir + "qwen_template/"

    args.train_path = args.data_dir + args.train_path
    args.valid_path = args.data_dir + args.valid_path
    args.test_path = args.data_dir + args.test_path

    inference_model = args.inference_model.lower()
    if "claude" in inference_model:
        args.inference_model_name = "claude"
    elif "llama" in inference_model:
        args.inference_model_name = "llama"
    if "qwen" in inference_model:
        args.inference_model_name = "qwen"

    template_model = args.template_model.lower()
    if "claude" in template_model:
        args.template_model_name = "claude"
    elif "llama" in template_model:
        args.template_model_name = "llama"
    if "qwen" in template_model:
        args.template_model_name = "qwen"

    if args.strategy == "human":
        args.ablation == "NA"
    if args.method == "ICL" or args.evaluate:
        args.train_size == "NA"


    base_path = f"{args.save_dir}{args.ft_name}/{args.inference_model_name}/"
    os.makedirs(base_path, exist_ok=True)
    checkpoint_dirs = [
        d
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
        and d.startswith("checkpoint")
        and "merged" in d
    ]
    if checkpoint_dirs:
        checkpoint_dir = checkpoint_dirs[0]
        args.save_model = os.path.join(base_path, checkpoint_dir)
        print(f"checkpoint path: {args.save_model}")
    else:
        args.save_model = base_path
        print("No checkpoint directory found")
    
    if args.ft_name == args.task_name:
        args.log_path = (
            args.log_dir + args.task_name + "/" + args.template_model_name + "/"
        )
        args.save_path = (
            args.save_dir + args.task_name + "/" + args.template_model_name + "/"
        )
        args.pred_path = (
            "predictions/" + args.task_name + "/" + args.template_model_name + "/"
        )
    else:
        args.log_path = (
            args.log_dir +'cross-' +args.task_name + "/" + args.template_model_name + "/"
        )
        args.save_path = (
            args.save_dir + 'cross-' +args.task_name + "/" + args.template_model_name + "/"
        )
        args.pred_path = (
            "predictions/" + 'cross-' +args.task_name + "/" + args.template_model_name + "/"
        )  

    if args.evaluate:
        args.file_path = f"{args.inference_model_name}_{args.method}_{args.strategy}_{args.ablation}_{args.n_shot}_{args.evaluate}"
    else:
        args.file_path = f"{args.inference_model_name}_{args.method}_{args.strategy}_{args.ablation}_{args.n_shot}"

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pred_path):
        os.makedirs(args.pred_path)

    log = Logger(args.log_path + args.file_path + ".log")

    start = time.time()
    log.logger.info("************************Start Test**********************")
    log.logger.info(
        f"【task】: {args.task_name} 【inference_model】: {args.inference_model_name} 【template_model】: {args.template_model_name} 【method】: {args.method} 【strategy】: {args.strategy} 【ablation】: {args.ablation} 【n_shot】: {args.n_shot}"
    )
    log.logger.info(
        "\n"
        + f"【inference model】: {args.inference_model}  【template model】: {args.template_model}  【embedding model】: {args.embedding_model} "
    )
    log.logger.info(
        "\n"
        + f"【test size】: {args.test_size}  【train size】: {args.train_size}  【only evaluate】: {str(args.evaluate)} "
    )

    run(args, log)

    if args.method == "ICL":
        data = load_pkl(f"{args.pred_path}{args.file_path}.pkl")
        print("************************Loading Predictions**********************")
        print(f"load {args.pred_path}{args.file_path}.pkl")

        if args.task_name != "rec":
            preds = [
                int(i["prediction"])
                for i in data
                if "prediction" in i and type(i["prediction"]) != str
            ]
            answers = [
                int(i["label"])
                for i in data
                if "prediction" in i and type(i["prediction"]) != str
            ]
            report = classification_report(answers, preds, digits=4)
        elif "rec" in args.task_name:
            preds = [brand2list(i["prediction"]) for i in data if "prediction" in i]
            clicks = [i["click"] for i in data if "prediction" in i]
            answers = [brand2list(i["label"]) for i in data if "prediction" in i]
            report = get_report(answers, preds, clicks)

        log.logger.info("******************Classification Report*****************")
        log.logger.info("\n" + report)
        end = time.time()
        log.logger.info("************************End Test************************")
        log.logger.info("Processing time: {} mins".format((end - start) / 60))
        print(f"save log to {args.log_path}{args.file_path}.log")

if __name__ == "__main__":
    main()
