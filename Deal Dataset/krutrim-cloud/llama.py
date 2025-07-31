# %%
print("Loading KrutrimCloud model...")
from krutrim_cloud import KrutrimCloud
from dotenv import load_dotenv
import pandas as pd
import time
import os
from datetime import datetime

# %%
load_dotenv()
api_key = os.getenv("KRUTRIM_API_KEY")
client = KrutrimCloud(api_key=api_key)

# %%
def generate(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful and concise travel agent assistant."},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="Llama-3.3-70B-Instruct",
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt+1} failed. Error: {e}")
            time.sleep(2)
    
    print("Skipping after multiple failures.")
    return None


def model_response(dialogue: str) -> str:
    prompt = f"""
Instruction:
Continue the conversation as the Travel agent. Respond appropriately to the latest user message. 
And please be brief.

Give the reply for this query: {dialogue}
"""
    return generate(prompt)


# %%
model_response("What are the best travel plans for motor?")  # Example usage

# %%
def create_dataset():
    df = pd.read_csv('/home/rohank__iitp/Work/niladri/Deal Dataset/deal dataset.csv')

    if not df.empty:
        output_filename = '/home/rohank__iitp/Work/niladri/Deal Dataset/krutrim-cloud/llama/llama_single_dataset.csv'
        header_written = False

        grouped = df.groupby('conversation_id')

        for conversation_id, group in grouped:
            print(f"\nProcessing Conversation ID: {conversation_id}")
            group = group.sort_values('turn_no')
            conversation_history = ""
            processed_rows = []

            for index, row in group.iterrows():
                sentence = "Conversation History:\n" + conversation_history + "Current Utterance: " + f"User: {row['utterance']}\nAgent:"

                print("========================================================================================================================================")
                print(f"Generating for conv_id: {row['conversation_id']}, turn: {row['turn_no']}\nPROMPT:\n{sentence}")
                print("========================================================================================================================================")
                
                # ⬇️ Krutrim model response
                krutrim_response = model_response(sentence)

                current_row_data = row.to_dict()
                current_row_data['LLama Single Response'] = krutrim_response
                processed_rows.append(current_row_data)

                # Update conversation history using actual agent reply
                conversation_history += f"User: {row['utterance']}\nAgent: {row['new_agent_reply']}\n"

            processed_group_df = pd.DataFrame(processed_rows)

            if not header_written:
                processed_group_df.to_csv(output_filename, index=False, mode='w')
                header_written = True
            else:
                processed_group_df.to_csv(output_filename, index=False, mode='a', header=False)

            print(f"Conversation ID {conversation_id} has been processed and saved.")

        print(f"\n✅ All conversations processed and saved to '{output_filename}'")

    else:
        print("\nDataFrame is empty. Nothing to process.")

# %%
print("Starting dataset creation...")
start_time = datetime.now()
print("Started at--->", start_time.strftime('%Y-%m-%d %H:%M:%S'))

create_dataset()

end_time = datetime.now()
print("Finished at--->", end_time.strftime('%Y-%m-%d %H:%M:%S'))
print(f"✅ Completed in {end_time - start_time} seconds")

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


