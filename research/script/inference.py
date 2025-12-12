import torch
from sentence_transformers import SentenceTransformer, util
import boto3
from script.prompt_template import *
import time
from script.utils import *
import torch
from script.prompt_template import *
from script.utils import *
from sentence_transformers import SentenceTransformer
from modelscope import Model
import torch
from modelscope import AutoTokenizer
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments
from modelscope import AutoTokenizer, AutoModelForCausalLM
from swift import Swift, LoraConfig
from datasets import load_dataset
from swift.llm import get_template, TemplateType, to_device, inference
from swift.tuners import Swift
from modelscope import snapshot_download
import os
from swift.llm import (
    ModelType,
    get_vllm_engine,
    get_default_template_type,
    get_template,
    inference_vllm,
    inference_stream_vllm,
)


def get_brand_description():
    brand_dict = {
        "brand1": "Automotive supplies",
        "brand2": "PC",
        "brand3": "Beverages",
        "brand4": "Baby products",
        "brand5": "Home improvement tools",
        "brand6": "Home entertainment electronics",
        "brand7": "Health and personal care products",
        "brand8": "Toys",
        "brand9": "Beauty products",
        "brand10": "Pet supplies",
        "brand11": "Smart home lighting",
        "brand12": "Home entertainment electronics",
        "brand13": "Sports equipment",
        "brand14": "Home entertainment electronics",
        "brand15": "Kitchen appliances",
        "brand16": "Optical equipment",
        "brand17": "Home furniture",
    }
    return brand_dict


def set_all_gpus():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()

        if num_gpus > 0:
            gpu_indices = ",".join(str(i) for i in range(num_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices
            return f"Set CUDA_VISIBLE_DEVICES to {gpu_indices}"
        else:
            return "No GPUs found"
    else:
        return "CUDA is not available"


def generate_message(
    bedrock_runtime, model_id, system_prompt, conversation, inference_config
):
    if "meta.llama3" in model_id:
        conversation[0]["content"][0][
            "text"
        ] = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{conversation[0]['content'][0]['text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    elif "anthropic.claude" in model_id:
        conversation[0]["content"][0][
            "text"
        ] = f"""{system_prompt}{conversation[0]['content'][0]['text']}"""
    elif "mistral" in model_id:
        conversation[0]["content"][0][
            "text"
        ] = f"""<s>[INST]{system_prompt}{conversation[0]['content'][0]['text']}[/INST]"""

    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig=inference_config,
        )

        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text

    except Exception as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        print("*" * 30, " retry ", "*" * 30)
        time.sleep(10)
        response = bedrock_runtime.converse(
            modelId=model_id,
            messages=conversation,
            inferenceConfig=inference_config,
        )
        response_text = response["output"]["message"]["content"][0]["text"]
        return response_text


class LLM:
    def __init__(self, args):
        set_all_gpus()
        self.lookup = {}
        self.task_name = args.task_name
        self.temperature = args.temperature
        self.max_output = args.max_output
        self.seed = args.seed

    def init_inference_model(self, args):
        self.inference_model = args.inference_model
        self.inference_model_name = args.inference_model.lower()

    def init_template_model(self, args):
        self.template_model = args.template_model
        self.template_model_name = args.template_model.lower()

        if (
            "claude" not in self.template_model_name
            and "meta.llama3" not in self.template_model_name
        ):
            if "qwen" in self.template_model_name:
                model_type = ModelType.qwen2_5_32b_instruct
                llm_engine = get_vllm_engine(
                    model_type,
                    model_id_or_path=None,
                    tensor_parallel_size=torch.cuda.device_count(),
                    # torch_dtype=torch.float16,
                    trust_remote_code=True,
                    # gpu_memory_utilization=0.90,
                    # max_model_len=32768,
                    do_sample=False,
                )
            elif "llama" in self.template_model_name:
                model_type = ModelType.llama3_1_70b_instruct_gptq_int4
                llm_engine = get_vllm_engine(
                    model_type,
                    model_id_or_path=None,
                    tensor_parallel_size=torch.cuda.device_count(),
                    # torch_dtype=torch.float16,
                    trust_remote_code=True,
                    # gpu_memory_utilization=0.90,
                    # max_model_len=113200,
                    do_sample=False,
                )

            llm_engine.generation_config.max_new_tokens = self.max_output

            self.template_engine = llm_engine
            self.template_model_type = model_type

    def init_embedding_model(self, args):

        model_dir = snapshot_download(args.embedding_model)
        self.embedding_model = SentenceTransformer(
            model_dir, trust_remote_code=True
        ).cuda()

    def init_bedrock(self):
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime", region_name="us-west-2"
        )

    def init_template(self, args):
        if args.task_name == "cls":
            s_template = cls_template()["start_template"]
            e_template = cls_template()["end_template"]
            self.template = {"start": s_template, "end": e_template}
        elif args.task_name == "rec":
            s_template = rec_template()["start_template"]
            e_template = rec_template()["end_template"]
            self.template = {"start": s_template, "end": e_template}

    def load_inference_model(self, save_path=None):

        if "qwen" in self.inference_model_name:
            model_type = ModelType.qwen2_5_32b_instruct
            llm_engine = get_vllm_engine(
                model_type,
                model_id_or_path=save_path,
                tensor_parallel_size=torch.cuda.device_count(),
                torch_dtype=torch.float16,
                trust_remote_code=True,
                # gpu_memory_utilization=0.90,
                max_model_len=32768,
                do_sample=False,
            )
        elif "llama" in self.inference_model_name:
            model_type = ModelType.llama3_1_70b_instruct_gptq_int4
            llm_engine = get_vllm_engine(
                model_type,
                model_id_or_path=save_path,
                tensor_parallel_size=torch.cuda.device_count(),
                # torch_dtype=torch.float16,
                trust_remote_code=True,
                # gpu_memory_utilization=0.90,
                # max_model_len=113200,
                do_sample=False,
            )

        llm_engine.generation_config.max_new_tokens = self.max_output

        self.inference_engine = llm_engine
        self.inference_model_type = model_type

    def get_adaptive_template(self, adaptive_template, prompt):
        if (
            "claude" in self.template_model_name
            or "meta.llama3" in self.template_model_name
        ):
            prompt = [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ]
            reply = generate_message(
                self.bedrock_runtime,
                self.template_model,
                system_prompt=adaptive_template,
                conversation=prompt,
                inference_config={
                    "maxTokens": self.max_output,
                    "temperature": self.temperature,
                },
            )
            print("*" * 50)
            print(prompt)
        else:
            template_type = get_default_template_type(self.template_model_type)
            template = get_template(template_type, self.template_engine.hf_tokenizer)

            request_list = [{"query": adaptive_template + " " + prompt}]
            response = inference_vllm(
                self.template_engine, template, request_list, generation_info={}
            )
            reply = response[0]["response"]
            print("*" * 50)
            print(adaptive_template + " " + prompt)

        print("*" * 50)
        print(reply)
        print("*" * 50)
        return reply

    def generate_data_embeddings(
        self, test_texts, train_texts, test_labels, train_labels, train_clicks=None
    ):
        test_embeddings = self.embedding_model.encode(
            test_texts, convert_to_tensor=True
        )
        self.test_lookup = {
            i: (test_embeddings[i], test_texts[i], test_labels[i])
            for i in range(len(test_texts))
        }
        train_embeddings = self.embedding_model.encode(
            train_texts, convert_to_tensor=True
        )
        if self.task_name == "rec":
            self.train_lookup = {
                i: (train_embeddings[i], train_texts[i], train_labels[i])
                for i in range(len(train_texts))
                if int(train_clicks[i]) == 1
            }
        else:
            self.train_lookup = {
                i: (train_embeddings[i], train_texts[i], train_labels[i])
                for i in range(len(train_texts))
            }
        print(f"[embedding length]: {len(self.train_lookup)}")
        del self.embedding_model
        torch.cuda.empty_cache()
        print(f"*************************Embeddings generated*************************")

    def add_simi_data(self, args, test):
        n_example = int(args.n_shot.split("_")[0])
        if n_example <= 0:
            return []

        brand_category = get_brand_description()

        for idx in range(len(test)):
            text_embedding = self.test_lookup[idx][0]
            filtered_lookup = self.train_lookup
            filtered_keys = list(filtered_lookup.keys())

            cos_scores = util.pytorch_cos_sim(
                text_embedding,
                torch.stack([filtered_lookup[i][0] for i in filtered_keys]),
            ).squeeze()

            sorted_indices = torch.argsort(cos_scores, descending=True)
            examples = []

            for index in sorted_indices[:n_example]:
                lookup_key = filtered_keys[index.item()]
                text = filtered_lookup[lookup_key][1]
                score = cos_scores[index].item()

                try:
                    label = int(filtered_lookup[lookup_key][2])
                    examples.append((text, label, score))
                except:
                    label = (
                        filtered_lookup[lookup_key][2]
                        .replace(".", "")
                        .replace(" ", "")
                        .split(",")
                    )
                    label = ", ".join([brand_category[k] for k in label])
                    examples.append((text, label, score))

            formatted_output = []
            confidence_values = [85, 80, 75, 70, 65]

            for key, item in enumerate(examples):
                text, label, score = item
                confidence = (
                    confidence_values[key] if key < len(confidence_values) else 80
                )

                if args.task_name == "cls":
                    formatted_text = (
                        f"{text}. Answer: {cls_n2t(label)}. Confidence: {confidence}."
                    )
                elif args.task_name == "rec":
                    formatted_text = f"{text}. Preferred categories: {label}."
                formatted_output.append(formatted_text)

            formatted_output = "\n".join(formatted_output)
            test[idx]["similar_text"] = formatted_output

        print("*" * 50, "Similar texts added", "*" * 50)
        return test

    def get_ft_results(self, entry, args):
        # torch.cuda.empty_cache()
        instruction, conversation = get_amz_template(args, entry, self.template)
        template_type = get_default_template_type(self.inference_model_type)
        template = get_template(template_type, self.inference_engine.hf_tokenizer)

        query = instruction + " " + conversation
        request_list = [{"query": query}]

        response = inference_vllm(
            self.inference_engine, template, request_list, generation_info={}
        )

        reply = response[0]["response"]
        print("*" * 50)
        print(f"query: {query}")
        print("*" * 50)
        print(f"response: {reply}")
        print("*" * 50)
        start_index = reply.find("{") if "{" in reply else 0
        reply = reply[start_index:]
        print(reply)
        print("*" * 50)

        try:
            result = eval(reply)
            if args.task_name == "cls":
                result["answer"] = cls_t2n(result["answer"])
            elif args.task_name == "rec":
                result["answer"] = rec_t2n(result["brand"])
        except:
            result = string2dict(reply, task=args.task_name)
        if args.task_name == "rec":
            pred = result["answer"]
        else:
            pred = result["answer"]
        rationale = result["reason"]
        confidence = round(float(convert_confidence(result["confidence"])), 2) * 0.01
        print("*" * 50, "complete", "*" * 50)
        print("[answer]", entry["label"])
        print("[prediction]", pred)
        return pred, rationale, confidence, query

    def get_results(self, entry, args):

        instruction, conversation = get_amz_template(args, entry, self.template)

        conversation = [
            {
                "role": "user",
                "content": [{"text": conversation}],
            }
        ]

        print("*" * 50)
        text_prompt = instruction + "\n" + conversation[0]["content"][0]["text"]
        print(text_prompt)
        print("*" * 50)

        reply = generate_message(
            self.bedrock_runtime,
            self.inference_model,
            system_prompt=instruction,
            conversation=conversation,
            inference_config={
                "maxTokens": self.max_output,
                "temperature": self.temperature,
            },
        )
        print("*" * 50)
        start_index = reply.find("{") if "{" in reply else 0
        reply = reply[start_index:]
        print(reply)
        print("*" * 50)

        try:
            result = eval(reply)
            if args.task_name == "cls":
                result["answer"] = cls_t2n(result["answer"])
            elif args.task_name == "rec":
                result["answer"] = rec_t2n(result["brand"])
        except:
            result = string2dict(reply, task=args.task_name)
        if args.task_name == "rec":
            pred = result["answer"]
        else:
            pred = int(result["answer"])
        rationale = result["reason"]
        confidence = round(float(convert_confidence(result["confidence"])), 2) * 0.01
        print("*" * 50, "complete", "*" * 50)
        print("[answer]", entry["label"])
        print("[prediction]", pred)

        return pred, rationale, confidence, text_prompt
