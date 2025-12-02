import pandas as pd
import re
import gc
from datasets import Dataset
import torch
import huggingface_hub
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from transformers import (BitsAndBytesConfig,
                          AutoTokenizer,
                          AutoModelForCausalLM,
                          pipeline,
                        )

class CharacterChatbot():
    def __init__(self,
                 model_path,
                 data_path='../data/naruto.csv',
                 huggingface_token = None
        ):
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.model_path = model_path
        self.data_path = data_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
        if huggingface_token is not None:
            huggingface_hub.login(huggingface_token)

        if not huggingface_hub.repo_exists(self.model_path):
            print("Model not found in huggingface hub, we will train our own model")
            dataset = self.Load_data()
            self.train_model(dataset)

        self.model = self.load_model(self.model_path)

    def chat(self, message, history):
        messages = []
        # Add the system ptomp 
        messages.append({"role":"system","content":""""Your are Naruto from the anime "Naruto". Your responses should reflect his personality and speech patterns \n"""})

        for message_and_respnse in history:
            messages.append({"role":"user","content":message_and_respnse[0]})
            messages.append({"role":"assistant","content":message_and_respnse[1]})
        
        messages.append({"role":"user","content":message})

        terminator = [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output = self.model(
            messages,
            max_length=256,
            eos_token_id=terminator,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        output_message = output[0]['generated_text'][-1]
        return output_message
        

    def load_model(self, model_path):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.float16
        )
        model = pipeline(
            "text-generation",
            model=model_path,
            quantization_config=quant_config,
            device_map= self.device,
            torch_dtype=torch.float16,)
        
        return model

    
    def train_model(self, 
                    dataset,
                    output_dir = "./results",
                    per_device_train_batch_size = 1,
                    gradient_accumulation_steps = 1,
                    optim = "paged_adamw_32bit",
                    save_steps = 200,
                    logging_steps = 10,
                    learning_rate = 2e-4,
                    max_grad_norm = 0.3,
                    max_steps = 300,
                    warmup_ratio = 0.3,
                    lr_scheduler_type = "constant",
               ):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map= self.device,
            torch_dtype=torch.float16,
        )
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CASUAL_LM"
        )

        peft_model = PeftModel.from_pretrained(
            model,
            lora_config=lora_config,
        )

        sft_config = SFTConfig(
                output_dir=output_dir,
                per_device_train_batch_size = per_device_train_batch_size,
                gradient_accumulation_steps = gradient_accumulation_steps,
                optim = optim,
                save_steps = save_steps,
                logging_steps = logging_steps,
                learning_rate = learning_rate,
                fp16= True,
                max_grad_norm = max_grad_norm,
                max_steps = max_steps,
                warmup_ratio = warmup_ratio,
                group_by_length = True,
                lr_scheduler_type = lr_scheduler_type,
                report_to = "none"
        )
        max_seq_len = 512

        trainer = SFTTrainer(
            model=peft_model,
            train_dataset=dataset,
            peft_config=lora_config,
            dataset_text_field="prompt",
            tokenizer=tokenizer,
            args=sft_config,
            max_seq_length=max_seq_len,
        )

        trainer.train()
        
        # Save model 
        trainer.model.save_pretrained("final_ckpt")
        tokenizer.save_pretrained("final_ckpt")

        trainer.model.push_to_hub(self.model_path)
        tokenizer.push_to_hub(self.model_path)

        # Flush memory
        del trainer, model
        gc.collect()
   


    def load_data(self):
        df = pd.read_csv(self.data_path)
        df.dropna(inplace=True)
        def clean_text(text):
            return re.sub(r'\(.*?\)', '', text)
        def count_words(text):
            return len(text.strip().split(" "))
        
        df['line'] = df['line'].apply(clean_text)
        df['word_count'] = df['line'].apply(count_words)
        df['selected'] = 0
        df.loc[(df['name'] == "Naruto") & (df['word_count'] > 5), 'selected'] = 1
        selected_indx = df[(df['selected'] == 1) & (df.index>0)].index.tolist()
        system_prompt = "You are a Naruto from the anime Naruto. Answer the questions in the style of Naruto Uzumaki."
        prompts = []
        for i in selected_indx:
            prompt = system_prompt + "\n"
            prompt += df.iloc[i-1]['line'] + "\n"
            prompt += df.iloc[i]['line'] 
            prompts.append(prompt)

        df_prompt = pd.DataFrame({'prompt': prompts})
        dataset = Dataset.from_pandas(df_prompt)
        return dataset