import torch
from script.prompt_template import *
from script.utils import *
from sentence_transformers import SentenceTransformer
from modelscope import Model
import torch
from modelscope import AutoTokenizer
from swift import Seq2SeqTrainer, Seq2SeqTrainingArguments
from modelscope import AutoTokenizer
from swift import Swift, LoraConfig
from datasets import load_dataset
import boto3
import time
from swift.llm import get_template, TemplateType, to_device, inference
from swift.tuners import Swift
from swift.llm import (
    ModelType,
    get_vllm_engine,
    get_default_template_type,
    get_template,
    inference_vllm,
    inference_stream_vllm,
)
import os
import os
import json
import torch
from swift.llm import ModelType, RLHFArguments, rlhf_main
from swift.utils import parse_args
from datasets import Dataset


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
    # put system prompt
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


class Llama_FT:

    def __init__(
        self,
        data_dir,
        task_name,
        save_path,
        model_id=None,
        train_path=None,
        valid_path=None,
        max_output=None,
        temperature=None,
    ):
        set_all_gpus()
        self.data_dir = data_dir
        self.model_id = model_id
        self.task_name = task_name
        self.train_path = train_path
        self.valid_path = valid_path
        self.save_path = save_path
        self.lookup = None
        self.max_output = max_output
        self.temperature = temperature
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime", region_name="us-west-2"
        )
        self.alpaca_prompt = """Below is an instruction that describes a task, pasired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

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
                    # max_seq_len_to_capture=32768
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
                self.template_engine,
                template,
                request_list,
                generation_info={},
                max_seq_len_to_capture=32768,
                max_model_len=32768,
            )
            reply = response[0]["response"]
            print("*" * 50)
            print(adaptive_template + " " + prompt)

        print("*" * 50)
        print(reply)
        print("*" * 50)
        return reply

    def load_model(self):
        torch.cuda.empty_cache()
        model = Model.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.max_new_tokens = self.max_ouput
        model.generation_config.temperature = self.temperature
        self.tokenizer = tokenizer
        self.model = model

    def run(self,args):
        self.template_model = args.template_model
        self.template_model_name = args.template_model.lower()
        print("*************************Start FT Llama*************************")
        torch.cuda.empty_cache()
        if self.task_name == "cls":
            train = load_pkl(self.train_path)
            valid = load_pkl(self.valid_path)
            print(train[0])
            print(f"length of train:valid: {len(train)}:{len(valid)}")
            lora_config = LoraConfig(target_modules=["q_proj", "k_proj", "v_proj"])
            model = Swift.prepare_model(self.model, lora_config)
            template = get_template(TemplateType.chatglm3, self.tokenizer)
            train_args = Seq2SeqTrainingArguments(
                output_dir="output",
                learning_rate=1e-4,
                num_train_epochs=2,
                eval_steps=500,
                save_steps=500,
                evaluation_strategy="steps",
                save_strategy="steps",
                dataloader_num_workers=4,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                logging_steps=10,
            )
            trainer = Seq2SeqTrainer(
                model=model,
                args=train_args,
                data_collator=template.data_collator,
                train_dataset=train,
                eval_dataset=valid,
                tokenizer=self.tokenizer,
            )
            trainer.train()
            self.model = model

        elif self.task_name == "rec":

            if "qwen" in self.template_model_name:
                model_type = ModelType.qwen2_5_32b_instruct

            elif "llama" in self.template_model_name:
                model_type = ModelType.llama3_1_70b_instruct_gptq_int4

            rlhf_args = RLHFArguments(
                model_type=model_type,
                rlhf_type="kto",
                beta=0.1,
                desirable_weight=1.0,
                undesirable_weight=1.0,
                sft_type="lora",
                num_train_epochs=2,
                lora_target_modules=["q_proj", "k_proj", "v_proj"],
                gradient_checkpointing=True,
                # dtype="fp16",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=16,
                max_grad_norm=1.0,
                # ddp_backend='nccl',
                # ddp_find_unused_parameters=False,
                # ddp_broadcast_buffers=False,
                learning_rate=5e-5,
                warmup_ratio=0.03,
                weight_decay=0.1,
                evaluation_strategy="steps",
                eval_steps=100,
                save_strategy="steps",
                save_steps=100,
                save_total_limit=2,
                dataloader_num_workers=0,
                dataloader_pin_memory=True,
                dataset=f"{self.data_dir}{self.task_name}_FT_train.json",
                val_dataset=f"{self.data_dir}{self.task_name}_FT_valid.json",
                output_dir="output_kto",
                logging_steps=5,
            )

            output = rlhf_main(rlhf_args)

    def save_model(self):
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        print("*************************Model Saved*************************")
        print(f"save the fine-tuned model to: {self.save_path}")
