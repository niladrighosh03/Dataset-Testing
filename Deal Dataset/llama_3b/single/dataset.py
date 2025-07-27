# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
print("Loading model and tokenizer...")
model_name = "/scratch/rohank__iitp/llama3.2"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# %%
def generate(prompt:str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    # Generate text
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        
    )

    # Decode and print response
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()

# %%
def model_response(dialogue):

    prompt = f"""
Instruction:
Continue the conversation as the Travel agent. Respond appropriately to the latest user message. 
And please be brief.

    Give the reply for this query: {dialogue}
    """
    return generate(prompt)

# %%
import pandas as pd

def create_dataset():

    # Make sure your CSV has the columns: 'conversation_id', 'turn_no', 'utterance', 'new_agent_reply'
    df = pd.read_csv('/home/rohank__iitp/Work/niladri/Deal Dataset/deal dataset.csv')
    # --- Response Generation and Incremental Saving ---

    if not df.empty:
        output_filename = '/home/rohank__iitp/Work/niladri/Deal Dataset/llama_3b/single/llama_3b_single_dataset.csv'
        header_written = False
        
        # Group by conversation_id to process one conversation at a time
        grouped = df.groupby('conversation_id')

        for conversation_id, group in grouped:
            print(f"\nProcessing Conversation ID: {conversation_id}")
            
            # Ensure the conversation turns are in chronological order
            group = group.sort_values('turn_no')
            conversation_history = ""
            processed_rows = []

            for index, row in group.iterrows():
                # Construct the prompt with the history plus the current user utterance
                sentence = "Conversation History:\n" + conversation_history + "Current Utterance: " + f"User: {row['utterance']}\nAgent:"
                # Your debugging print statements
                print("========================================================================================================================================")
                print(f"Generating for conv_id: {row['conversation_id']}, turn: {row['turn_no']}\nPROMPT:\n{sentence}")
                print("========================================================================================================================================")
                
                
                
                # Generate the response
                '''Change Here😆😆😆😆'''
                qwen_response = model_response(sentence)
                
                
                # Create a dictionary from the original row and add the new response
                current_row_data = row.to_dict()
                current_row_data['llama_3b Single Response'] = qwen_response
                processed_rows.append(current_row_data)

                # Update the history for the next turn in this conversation
                conversation_history += f"User: {row['utterance']}\nAgent: {row['new_agent_reply']}\n"
            
            # Create a DataFrame for the just-processed conversation
            processed_group_df = pd.DataFrame(processed_rows)

            # Append the processed conversation to the output CSV file
            if not header_written:
                # For the first conversation, write with header and overwrite the file
                processed_group_df.to_csv(output_filename, index=False, mode='w')
                header_written = True
            else:
                # For subsequent conversations, append without the header
                processed_group_df.to_csv(output_filename, index=False, mode='a', header=False)
            
            print(f"Conversation ID {conversation_id} has been processed and saved.")

        print(f"\nProcessing complete. All conversations have been saved to '{output_filename}'")

    else:
        print("\nDataFrame is empty. No responses were generated or saved.")

# %%
from datetime import datetime

print("Starting dataset creation...")
start_time = datetime.now()
print("Started at--->", start_time.strftime('%Y-%m-%d %H:%M:%S'))
create_dataset()
# End timer
end_time = datetime.now()
print("Finished time", end_time.strftime('%Y-%m-%d %H:%M:%S'))

# Print elapsed time
print(f"hey() completed in {end_time - start_time} seconds")


