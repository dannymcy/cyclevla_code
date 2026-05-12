import os
import time
import json
from PIL import Image
import torch, gc
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info


# Load environment variables from .env file
load_dotenv()

# huggingface-cli login: HUGGINGFACE_TOKEN
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)


# ==============================================================================================
# Local VLM inference
def clear_gpu_memory(model, trainer, tokenizer):
    # Step 1: Move any remaining models/tensors to CPU
    torch.cuda.empty_cache()

    # Step 2: Force garbage collection to clear up memory
    gc.collect()

    # Step 3: If using a model, explicitly move to CPU and delete
    # For example, if your model is still in memory:
    model.cpu()
    del model
    if trainer is not None: del trainer
    del tokenizer

    # Step 4: Clear the GPU cache
    torch.cuda.empty_cache()

    # Step 5: (Optional) Reset device states
    for device in range(torch.cuda.device_count()):
        with torch.cuda.device(device):
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()


def get_available_memory():
    free_memory = {}
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        buffer_memory = -1 * 1024 ** 3  # 0.1GB buffer
        available_memory = max(0, free - buffer_memory)
        free_memory[i] = f"{int(available_memory / (1024 ** 3))}GB"
        print(f"GPU {i} | free: {free/(1024**3):.2f} GB | total: {total/(1024**3):.2f} GB | available: {free_memory[i]}")
    return free_memory


def vlm_init(model_name="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    """
    Initialize the VLM model, tokenizer, and processor.
    This only needs to be run once.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        tuple: (model, tokenizer, processor)
    """
    if "Qwen" in model_name:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            use_auth_token=os.environ['HUGGINGFACE_TOKEN'],  
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16,
            device_map='balanced',
            # device_map='auto',
            # max_memory=get_available_memory(),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=os.environ['HUGGINGFACE_TOKEN'],  
            trust_remote_code=True,
            # torch_dtype=torch.bfloat16,
            device_map='balanced'
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=os.environ['HUGGINGFACE_TOKEN'],
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        model_name, 
        use_auth_token=os.environ['HUGGINGFACE_TOKEN']
    )
    
    return [model, tokenizer, processor]


def vlm_inference_mllama(model, tokenizer, processor, prompt, image=None, max_new_tokens=64, temperature=0.2):
        # Construct message for Llama-3 Vision
        messages = [{"role": "user", "content": []}]

        if image is None:
            messages[0]["content"].append({"type": "text", "text": prompt})
        else:
            messages[0]["content"].append({"type": "image", "image": image})
            messages[0]["content"].append({"type": "text", "text": prompt})

        # Prepare input prompt
        input_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Tokenize input
        inputs = tokenizer(input_prompt, return_tensors="pt", truncation=False).to(model.device)

        # Run inference
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
        
        # Decode and print response
        prompt_ids = inputs["input_ids"][0]
        full_output_ids = outputs[0]
        generated_ids = full_output_ids[len(prompt_ids):]

        # Decode and print response
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text


def vlm_inference_qwen(save_path, model, tokenizer, processor, prompt, images=None, max_new_tokens=64, temperature=0.2, image_size=(256, 256)):
    """
    Run inference with Qwen2.5-VL or similar VLM models supporting multiple images.
    
    Args:
        model: The vision-language model
        tokenizer: The tokenizer for the model
        processor: The image/text processor
        prompt: Text prompt
        images: Single image path/URL/base64 or list of images (up to 3)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        image_size: Tuple of (height, width) for image resizing
    
    Returns:
        str: Generated text response
    """
    # Process images parameter to ensure consistent format
    if images is None:
        image_list = []
    elif not isinstance(images, list):
        image_list = [images]  # Convert single image to list
    else:
        image_list = images  # Limit to 3 images if more are provided
    
    # Construct message for vision model
    messages = [{"role": "user", "content": []}]
    
    # Add images with specific size
    for img in image_list:
        messages[0]["content"].append({
            "type": "image",
            "image": img,
            "resized_height": image_size[0],
            "resized_width": image_size[1]
        })
    
    # Add text prompt
    messages[0]["content"].append({"type": "text", "text": prompt})
    
    # Process with qwen-vl-utils if available (for better handling of various image formats)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    # Move to device
    start = time.time()
    inputs = inputs.to(model.device)
    
    # Run inference
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
    
    # Decode response
    prompt_ids = inputs.input_ids[0]
    generated_ids = outputs[0][len(prompt_ids):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    end = time.time()
    used_time = end - start

    print(generated_text)
    print()

    json_data = {"used_time": used_time, "res": generated_text, "system": "You are a helpful assistant.", "user": prompt}
    
    with open(save_path, "w") as f:
        json.dump(json_data, f, indent=4)
    
    return json_data




# ==============================================================================================
# GPT inference
def wait_for_finetune_completion(job_id, model_path):
    """
    Poll the fine-tuning job status until it completes, then extract and save the fine-tuned model name.

    Args:
    - job_id (str): The ID of the fine-tuning job.
    - model_path (str): Path to save the fine-tuned model name.
    """
    while True:
        # Retrieve the current status of the fine-tuning job
        finetune_job = client.fine_tuning.jobs.retrieve(job_id)
        job_status = finetune_job.status

        print(f"Job Status: {job_status}")

        if job_status == "succeeded":
            # Extract the fine-tuned model name
            finetuned_model_name = finetune_job.fine_tuned_model
            print(f"Fine-tuned Model Name: {finetuned_model_name}")

            # Save the fine-tuned model name for future use
            with open(model_path, "w") as f:
                f.write(finetuned_model_name)

            return finetuned_model_name
        
        elif job_status == "failed":
            raise Exception(f"Fine-tuning job failed: {finetune_job.error}")
        
        # Wait for some time before polling again
        time.sleep(30)  # Poll every 30 seconds


def finetune(data, suffix, model_path, n_epochs=1, model="gpt-4o-mini-2024-07-18"):
    file_upload = client.files.create(
        file=open(data, "rb"),
        purpose="fine-tune"
    )
    file_id = file_upload.id

    hyperparameters = {
        "n_epochs": n_epochs
    }

    finetune_job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model,
        suffix=suffix,
        hyperparameters=hyperparameters,
        seed=42
    )
    finetuned_model_name = wait_for_finetune_completion(finetune_job.id, model_path)

    return finetuned_model_name


def load_finetuned_model_name(model_path):
    # Read the model name from the file
    with open(model_path, "r") as f:
        finetuned_model_name = f.read().strip()
    return finetuned_model_name


def create_data(prompts, responses, json_path=None, training=True):
    """
    Create structured data for prompt-response pairs for fine-tuning or inference.

    Args:
    - prompts (list of str): A list of user prompts.
    - responses (list of str): A list of assistant responses.
    - json_path (str): Path to save the generated JSONL file if in training mode.
    - training (bool): If True, save the data as JSONL. If False, return structured data for inference.

    Returns:
    - If training is False, returns the structured data without saving to a file.
    """
    # Ensure that the prompts and responses lists are of equal length
    if len(prompts) != len(responses):
        raise ValueError("Prompts and responses must have the same length.")
    
    data = []
    
    # Iterate through the prompts and responses to build the structured data
    for prompt, response in zip(prompts, responses):
        conversation = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        }
        data.append(conversation)
    
    # If training, save the data to a JSONL file
    if training:
        with open(json_path, 'w') as f:
            for conversation in data:
                f.write(json.dumps(conversation) + '\n')  # Write each dict as a JSON line
        print(f"Data has been written to {json_path}")
    else:
        # If not training, return the structured data for inference
        return data


def query(system, user_contents, assistant_contents, save_path=None, model='gpt-4', temperature=1, debug=False, max_retries=20):
    for user_content, assistant_content in zip(user_contents, assistant_contents):
        user_content = user_content[0].split("\n")
        assistant_content = assistant_content[0].split("\n")
        
        for u in user_content:
            print(u)
        print("=====================================")
        for a in assistant_content:
            print(a)
        print("=====================================")

    for u in user_contents[-1][0].split("\n"):
        print(u)

    if debug:
        import pdb; pdb.set_trace()
        return None

    print("=====================================")

    start = time.time()
    
    num_assistant_mes = len(assistant_contents)
    messages = []

    messages.append({"role": "system", "content": "{}".format(system)})
    for idx in range(num_assistant_mes):
        messages.append({"role": "user", "content": user_contents[idx][0]})
        if user_contents[idx][1]:
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]}]
            for image_url in user_contents[idx][1]:
                messages[-1]["content"].append({"type": "image_url", "image_url": {"url": image_url, "detail": "high"}})
            
        messages.append({"role": "assistant", "content": assistant_contents[idx][0]})
        if assistant_contents[idx][1]:
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]}]
            for image_url in assistant_contents[idx][1]:
                messages[-1]["content"].append({"type": "image_url", "image_url": {"url": image_url, "detail": "high"}})
    messages.append({"role": "user", "content": user_contents[-1][0]})
    
    # Add the base64 encoded image to the last user message
    if user_contents[-1][1]:
        messages[-1]["content"] = [
            {"type": "text", "text": messages[-1]["content"]}]
        for image_url in user_contents[-1][1]:
            messages[-1]["content"].append({"type": "image_url", "image_url": {"url": image_url, "detail": "high"}})

    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096
            )

            result = ''
            for choice in response.choices: 
                result += choice.message.content 

            end = time.time()
            used_time = end - start

            print(result)
            print()

            user_contents_text, assistant_contents_text = [], []
            for user_content in user_contents:
                user_contents_text.append(user_content[0])
            for assistant_content in assistant_contents:
                assistant_contents_text.append(assistant_content[0])

            if save_path is not None:
                with open(save_path, "w") as f:
                    json.dump({"used_time": used_time, "res": result, "system": system, "user": user_contents_text, "assistant": assistant_contents_text}, f, indent=4)
                with open(save_path, 'r') as f:
                    json_data = json.load(f)
            
            return json_data

        except OpenAIError as e:
            print(f"Error occurred: {str(e)}. Retrying...")
            retries += 1
            sleep(2)  # Adding a delay before retrying

    print(f"Failed after {max_retries} attempts.")
    return None
