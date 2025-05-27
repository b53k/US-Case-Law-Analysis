import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
import json
import os
import tiktoken
import time
import re
from google import genai
import numpy as np



def read_local_parquet():
    """
    Reads a local parquet file and returns it as a pandas DataFrame.
    """
    filepath = "data/ga_with_categorization.parquet" 
    try:
        table = pq.read_table(filepath)
        df = table.to_pandas()
        print(df.head()) 
        print(df.shape)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def estimate_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def readfile():
    """Reads the contents of a text file and returns it as a string."""
    file_path = "data/opinion.txt"  # Replace with the actual file path
    try:
        with open(file_path, "r") as file:
            file_content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        file_content = "File content not found."  # Provide a default message
    return file_content


prompt_text = "You have a task is two things: 1. Determine the case law category from the four categories (criminal, civil, bankruptcy, other) based on the opinion transcript. 2. Also, give me an estimate of how complex this case was between 1-10 (1 being simple and 10 being complex). Answer just in a few words. Sample answer format: '(criminal, 5)'.\n"
all_results = []


def extract_json(text):
    """Extract valid JSON array from response text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0).strip())
        raise ValueError("Could not extract valid JSON.")

def createBatchPrompt(df):
    """
    Creates a classification prompt for a batch of legal cases.
    Expects df to have at least 'id' and 'text' columns.
    Returns the prompt string.
    """
    instructions = """
You are a legal domain expert. For each legal case below, assign:

1. A **primary category** from this limited list:
   - Civil
   - Criminal
   - Constitutional
   - Administrative

2. A **subcategory** based on the case content, chosen from the options under each main category:

Civil:
- contracts
- torts
- property disputes
- family law
- other civil

Criminal:
- theft
- assault
- drug offenses
- murder
- fraud
- other criminal

Constitutional:
- free speech rights
- due process challenges
- equal protection cases
- other constitutional

Administrative:
- immigration rulings
- environmental regulations
- labor disputes
- other administrative

If the case does not clearly fall into a listed subcategory, choose the most appropriate "other ___" option.

3. Also, give me an estimate of the complexity of this case between 1-10 (1 being simple and 10 being complex).


You will be given a numbered list of cases (e.g., "Case 1", "Case 2", etc.).

Each case will have an ID like 
"-------------
Case 2:
[ID: 634964]
".

IMPORTANT: Do NOT miss any cases there should be about 40 cases!

Return your results in this **exact JSON format** (no markdown, no explanations):

[
  {"ID": "12165933", "category": "Criminal Law", "subcategory": "theft", "complexity": 6},
  {"ID": "634964", "category": "Civil Law", "subcategory": "contracts", "complexity": 3},
  {"ID": "1467263", "category": "Civil Law", "subcategory": "other civil", "complexity": 4},
  ...
]

Only return the JSON array.

Below are the cases:\n
"""

    # case_entries = []
    # for _, row in df.iterrows():
    #     case_block = f"[ID: {row['id']}]\n{row['text']}\n"
    #     case_entries.append(case_block)

    # prompt = instructions + "\n" + "\n\n-------------------------\n".join(case_entries)
    # return prompt

    case_entries = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        case_block = f"Case {i}:\n[ID: {row['id']}]\n{row['text']}\n"
        case_entries.append(case_block)

    prompt = instructions + "\n\n-------------------------\n".join(case_entries)
    return prompt



def startQuery(big_df):
    """Starts the query process for the legal cases."""

    batch_size = 40   # adjust based on token estimation
    start_batch = 2928  # Change this to the last completed batch (or next one you want to start with)

    # big_df = df.sample(80)  # For testing, use a smaller sample

    # for start in tqdm(range(0, len(big_df), batch_size)):
    #     batch_df = big_df.iloc[start:start + batch_size]
    #     prompt_text = createBatchPrompt(batch_df)
    #     print(prompt_text)


    # Create output directories
    os.makedirs("data/results", exist_ok=True)
    os.makedirs("data/prompt_logs", exist_ok=True)

    all_results = []

    for batch_num, start in enumerate(tqdm(range(start_batch * batch_size, len(big_df), batch_size)), start=start_batch):
        batch_df = big_df.iloc[start:start + batch_size]
        prompt_text = createBatchPrompt(batch_df)


        # Save the prompt to a .txt file
        prompt_file = f"data/prompt_logs/prompt_batch_{batch_num:04d}.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        print("estimated tokens", estimate_tokens(prompt_text))

        

        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            # model="gemma-3-27b-it",
            contents=prompt_text
        )
        print(response.text)
        time.sleep(6)

        with open(f"data/prompt_logs/response_batch_{batch_num:04d}.txt", "w", encoding="utf-8") as f:
            f.write(response.text)

        # Try parsing the JSON
        try:
            parsed = extract_json(response.text)
        except Exception as e:
            print(f"Error parsing batch {batch_num}: {e}")
            parsed = []

        all_results.extend(parsed)

        # Optional: save partial results every few batches
        if batch_num % 5 == 0:
            with open("data/results/partial_results.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)


    # Save final results
    with open("data/results/final_results.txt", "w", encoding="utf-8") as f:
        for entry in all_results:
            f.write(f"ID: {entry.get('id') or entry.get('ID')}, "
                    f"category: {entry['category']}, "
                    f"complexity: {entry.get('complexity', 'N/A')}\n")


    with open("data/results/final_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Also as CSV if needed
    pd.DataFrame(all_results).to_csv("data/results/final_results.csv", index=False)

###################################################################################

def combine_json_files(input_folder, output_file):
    """
    Combine multiple JSON files into a single JSON file.
    
    Args:
        input_folder (str): Path to the folder containing JSON files.
        output_file (str): Path to the output combined JSON file.
    """
    all_data = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.extend(data)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

# combine_json_files("data/results", "data/results/combined_results.json")

###################################################################################







# df_categories = pd.read_json("data/results/combined_results.json", lines=True)
# df_categories = pd.read_json("data/results/categorization_results.json", lines=True)
# df_categories = pd.DataFrame(df_categories)
# df_categories.rename(columns={'ID': 'id'}, inplace=True)
# print(df_categories)
# print(df_categories.head())

# mask = df_categories['subcategory'] == 'torts (like personal injury)'
# df_categories.loc[mask, 'subcategory'] = 'torts'
# df_categories.to_json('data/results/combined_results.json', orient='records', lines=True)



# df_categories['id'] = df_categories['id'].astype(str) # Ensure ID is string for merging
# df_categories.to_json('data/results/categorization_results.json', orient='records', lines=True)

# Merge the two dataframes using a left merge on 'ID'
# df_merged = pd.merge(df, df_categories, on='id', how='left')
# print(df_merged.shape)
# print(df_merged.head())
# print(df_merged['subcategory'].unique())
# print(df_merged['subcategory'].value_counts().to_string())
# # df_merged.to_json('data/results/ga_with_categorization.json', orient='records', lines=True)

# Use boolean indexing to select rows where the 'category' column is NaN
# nan_category_rows = df_merged[~df_merged['category'].isna()]

# Display the first couple of these rows (e.g., the first 2)
# print(nan_category_rows.head(2))

# df_merged.to_parquet("data/ga_with_categorization.parquet", index=False)

###################################################################################
def cleanUp(df):

    # Define the allowed subcategories for each major domain.
    allowed = {
        'civil': {"contracts", "torts", "property disputes", "family law", "other civil"},
        'criminal': {"theft", "assault", "drug offenses", "murder", "fraud", "other criminal"},
        'constitutional': {"free speech rights", "due process challenges", "equal protection cases", "other constitutional"},
        'administrative': {"immigration rulings", "environmental regulations", "labor disputes", "other administrative"}
    }

    # A mapping for common misspellings or synonyms that you want to force into one of the allowed values.
    synonyms = {
        "civil": {
            "personal injury": "torts",
            "other family law": "family law",
            "other family": "family law",
            "child support": "family law",
            "child custody": "family law",
            "divorce": "family law",
            "adoption": "family law",
            "personal injury": "torts",         
            "civil": "other civil",               
            "other property": "property disputes",
            "other property disputes": "property disputes"  # Also treat "other property disputes" as "property disputes"
            # If you don’t want these to be family law already, you can simply let the default of "other civil" work.
        },
        "criminal": {
            "muder": "murder",
            "armed robbery": "other criminal",
            "robbery": "other criminal",
            "burglary": "other criminal",
            "rape": "other criminal",
            "child molestation": "other criminal",
            "manslaughter": "other criminal",
            "arson": "other criminal",
            "kidnapping": "other criminal",
            "perjury": "other criminal",
            "incest": "other criminal",
            "forgery": "other criminal",
            "seduction": "other criminal",
            "dui": "other criminal",
            "bribery": "other criminal",
            "sexual assault": "assault",          # we map variants to "assault"
            "aggravated assault": "assault",
            "embezzlement": "other criminal",
            "conspiracy": "other criminal",
            "child abuse": "other criminal",
            "fleeing": "other criminal",
            "affray": "other criminal",
            "defamation": "other criminal",
            "cruelty to children": "other criminal",
            "larceny": "theft",
            "criminal": "other criminal"
        },
        "constitutional": {
            # add synonyms if needed
        },
        "administrative": {
            "insurance rulings": "other administrative"
        }
    }

    # Helper function to determine the primary domain of a row based on the category.
    def get_domain(category):
        cat_lower = category.lower()
        # Check for key terms—feel free to customize these rules
        if "civil" in cat_lower or "family" in cat_lower:
            return "civil"
        elif "criminal" in cat_lower:
            return "criminal"
        elif "constitutional" in cat_lower:
            return "constitutional"
        elif "administrative" in cat_lower:
            return "administrative"
        else:
            return None  # Unknown domain; you may choose to handle it differently

    # Function to map a subcategory to the allowed list (or its synonyms) for a given domain.
    def map_subcategory(subcat, domain):
        subcat = subcat.lower().strip()
        if domain is None:
            # If we cannot determine the domain, return the original value (or choose a default)
            return subcat

        # If the subcategory is already in the allowed set, keep it
        if subcat in allowed[domain]:
            return subcat

        # Use the synonyms dictionary if an alternative mapping is found
        if domain in synonyms and subcat in synonyms[domain]:
            return synonyms[domain][subcat]

        # (Optional) Additional rule example: for civil, if the term contains "family",
        # force it to "family law"
        if domain == "civil" and "family" in subcat:
            return "family law"

        # Default: assign to the "other" category for the domain
        default_mapping = {
            "civil": "other civil",
            "criminal": "other criminal",
            "constitutional": "other constitutional",
            "administrative": "other administrative"
        }
        return default_mapping[domain]

    # Apply the mapping to each row to produce a new column with the recoded subcategory.
    df['subcategory'] = df.apply(
        lambda row: map_subcategory(row['subcategory'], get_domain(row['category'])),
        axis=1
    )

    # Define the ambiguous subcategories to remove (in lowercase for consistent matching)
    ambiguous = {"other", "personal injury", "civil", "other property", "criminal", "other property disputes"}

    # Filter the DataFrame: keep only rows whose 'subcategory' (after stripping spaces and converting to lowercase) is not in the ambiguous set.
    df = df[~df['subcategory'].str.strip().str.lower().isin(ambiguous)]

    print(df['subcategory'].unique())
    print(df['subcategory'].value_counts().to_string())

    # Show the updated DataFrame
    print(df)
    df.to_json('data/results/combined_results_cleanedup.json', orient='records', lines=True)



def cleanUpCategory(df):
    cat_map = {
    'family law':                  'Civil',
    'other civil':                 'Civil',
    'torts (like personal injury)': 'Civil',
    'torts':                       'Civil',
    'property disputes':           'Civil',
    'other property':              'Civil',
    'other property disputes':     'Civil',

    'criminal':                    'Criminal',
    'other criminal':              'Criminal',

    'other constitutional':        'Constitutional',

    'other administrative':        'Administrative',
    'labor disputes':              'Administrative',

    'Other' :                   'Civil',
    'other' :                   'Civil',
    'NaN':                       'NaN',
    }

    # apply it, and for anything not in cat_map, Title‑case it (so "Civil" stays "Civil", etc.)
    df['category'] = (
        df['category']
        .str.strip()
        .str.lower()
        .map(cat_map)
        .fillna(df['category'].str.title())
    )
    counts = df['category'].value_counts()
    print(counts)
    stats = np.array(counts) / np.sum(counts)
    print(stats)
    print(df.shape)
    df.to_parquet("data/ga_with_categorization_fixed.parquet", index=False)


###################################################################################
###################################################################################
# ##### working GPU code.

def GPU1_LLM():
    from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
    import torch

    model_id = "google/gemma-3-1b-it"

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = Gemma3ForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)



    # prompt_text = "You have a task is two things: 1. Determine the case law category from the four categories (criminal, civil, bankruptcy, and others) based on the opinion transcript. 2. Also, give me an estimate of how complex this case was between 1-10 (1 being simple and 10 being complex). Answer just in a few words. Sample answer format: 'category: criminal, complexity: 5'. Here is the transcript:"
    # prompt_text = "You are a helpful assistant and have two tasks: 1. Determine the case law category from the four categories (criminal, civil, bankruptcy, and others) based on the opinion transcript. 2. Also, give me an estimate of how complex this case was between 1-10 (1 being simple and 10 being complex). Based on the transcript below answer in one sentence:"
    prompt_text = "You are a helpful assistant and have two tasks: 1. Determine the case law category from the four categories (criminal, civil, bankruptcy, and others) based on the opinion transcript. 2. Also, give me an estimate of how complex this case was as a number between 1-10. Answer in one sentence:"
    second_mssg = {
        "role": "user",
        "content": [{"type": "text", "text": readfile()},]
    }


    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt_text},]
            },
            second_mssg,
        ],
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device) # Remove the .to(torch.bfloat16) part.

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=512)

    outputs = tokenizer.batch_decode(outputs)[0]
    print("\n\n\nthe len is", len(outputs))

    print(outputs)

###################################################################################
###################################################################################
### for running the code on CPU
def CPU1_LLM():
    from transformers import AutoTokenizer, Gemma3ForCausalLM
    import torch

    model_id = "google/gemma-3-1b-it"

    model = Gemma3ForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"},]
            },
        ],
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=64)

    outputs = tokenizer.batch_decode(outputs)

    print(outputs)





# ###################################################################################
# ###################################################################################
### Gamma 4b model NOT quantized
def GPU2_LLM():
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    from PIL import Image
    import requests
    import torch

    model_id = "google/gemma-3-4b-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id)


    prompt_text = "You are a helpful assistant and have two tasks: 1. Determine the case law category from the four categories (criminal, civil, bankruptcy, and others) based on the opinion transcript. 2. Also, give me an estimate of how complex this case was as a number between 1-10. Answer in one sentence:"
    second_mssg = {
        "role": "user",
        "content": [{"type": "text", "text": readfile()},]
    }
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                # {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
                # {"type": "text", "text": "Describe this image in detail."}

                {"type": "text", "text": prompt_text},
                {"type": "text", "text": second_mssg}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)







# ###################################################################################
# ###################################################################################
# # # ##### 4b  GPU code working.
def GPU3_LLM():
    # from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    # from PIL import Image
    # import requests
    # import torch

    # model_id = "google/gemma-3-4b-it"

    # model = Gemma3ForConditionalGeneration.from_pretrained(
    #     model_id, device_map="auto"
    # ).eval()

    # processor = AutoProcessor.from_pretrained(model_id)
    from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
    import torch
    import accelerate # Good practice to import, though often used implicitly

    model_id = "google/gemma-3-4b-it"

    # --- Configuration ---
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # You might uncomment the line below if you still have issues and have a compatible GPU (Ampere or newer)
        # bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- Load Model ---
    # Add torch_dtype and device_map
    # Consider adding trust_remote_code=True if the below still fails
    model = Gemma3ForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16, # Specify compute dtype for stability
        device_map="auto",         # Let accelerate handle device placement
        # trust_remote_code=True,  # Often needed for newer/custom architectures like Gemma 3
    ).eval() # Keep eval()

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # --- Prepare Prompt ---
    prompt_text = "You have a task is two things: 1. Determine the case law category from the four categories (criminal, civil, bankruptcy, other) based on the opinion transcript. 2. Also, give me an estimate of how complex this case was between 1-10 (1 being simple and 10 being complex). Answer just in a few words. Sample answer format: '(criminal, 5)'.\n"
    second_mssg = {
        "role": "user",
        # "content": [{"type": "text", "text": "write a short poem about a cat"},] # Using a simpler task for testing
        "content": [{"type": "text", "text": readfile()},]
    }

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt_text},]
            },
            second_mssg,
        ],
    ]

    # --- Tokenize ---
    # No need to manually cast inputs to bfloat16 here, model handles internal types
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    for i in range(10):
        # --- Generate ---
        try:
            with torch.inference_mode():
                # Increase max_new_tokens if needed for the poem
                outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True) # Ensure sampling is on if desired

            # --- Decode ---
            # Decode only the generated part, skipping the input tokens
            output_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            print("\n--- Generated Output ---")
            print(output_text)
            print("------------------------")

            # For debugging: print the full output including prompt
            # full_outputs_decoded = tokenizer.batch_decode(outputs)[0]
            # print("\n--- Full Decoded Output ---")
            # print(full_outputs_decoded)
            # print("------------------------")

        except RuntimeError as e:
            print(f"\n--- ERROR ---")
            print(f"RuntimeError during generation: {e}")
            # print("Troubleshooting suggestions:")
            # If it still fails, try greedy decoding to isolate the issue:
            # print("6. Try testing with `do_sample=False` in `model.generate()`.")