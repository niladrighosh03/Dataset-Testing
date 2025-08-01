# %% [markdown]
# ### Load fine tuned Qween for selector (GRPO RL)

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

finetune_model_name = "/scratch/rohank__iitp/grpo_finetune_qween"

finetune_model = AutoModelForCausalLM.from_pretrained(
    finetune_model_name,
    torch_dtype="auto",
    device_map="auto"
)
finetune_tokenizer = AutoTokenizer.from_pretrained(finetune_model_name)


# %%
def finetune_generate(prompt:str):
    inputs = finetune_tokenizer(prompt, return_tensors="pt").to(finetune_model.device)
    input_length = inputs['input_ids'].shape[1]
    # Generate text
    outputs = finetune_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

    # Decode and print response
    generated_tokens = outputs[0][input_length:]
    response = finetune_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()

finetune_generate("What is the capital of France?")

# %% [markdown]
# ### Load LLAMA 3b for others

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

# %% [markdown]
# #### Sentiment Expert

# %%
def sentiment_expert(text_input: str) -> str:

   prompt = f"""
You are an AI trained to act solely as a **sentiment expert**. Your job is to analyze the **emotional tone** of the input text and classify it into one of the following three categories:

- **Positive** – The text expresses happiness, satisfaction, excitement, appreciation, or any other positive emotion.
- **Negative** – The text expresses disappointment, frustration, anger, sadness, criticism, or other negative feelings.
- **Neutral** – The text is emotionally balanced, factual, or shows no strong emotional content.

Your response must only contain:

1. **Sentiment:** One of the three labels – `Positive`, `Negative`, or `Neutral`
2. **Explanation:** A concise reason that supports the label, based only on emotional tone, word choice, or sentiment-laden phrases.

You must not:
- Provide summaries
- Offer personal opinions
- Evaluate content quality or logic
- Infer intent beyond emotional expression

Stick strictly to **sentiment analysis**.

### Few-Shot Examples:

1. **Text:** "Absolutely love this app – it's made my life so much easier!"
   **Sentiment:** Positive
   **Explanation:** The phrase "absolutely love" strongly conveys enthusiasm and satisfaction.

2. **Text:** "I'm really disappointed with the service. It was slow and rude."
   **Sentiment:** Negative
   **Explanation:** Words like "disappointed", "slow", and "rude" clearly express dissatisfaction.

3. **Text:** "The package arrived on Tuesday as scheduled."
   **Sentiment:** Neutral
   **Explanation:** This sentence is factual with no emotional language.

4. **Text:** "Not sure how I feel about this – it's kind of a mixed bag."
   **Sentiment:** Neutral
   **Explanation:** Ambiguous phrasing and lack of strong emotion suggest a neutral sentiment.

5. **Text:** "This is the worst experience I've had in months."
   **Sentiment:** Negative
   **Explanation:** The phrase "worst experience" indicates strong dissatisfaction.

Now analyze the following text:

**Text:** "{text_input}"
"""


   return generate(prompt)

# %% [markdown]
# #### Persuassion Expert

# %%
def persuassion_expert(text_input: str) -> str:
   prompt = f"""You are a Persuasion Strategy Selector for a motor insurance dialogue system. 
   Based on the user's most recent utterance and the conversation history, you must recommend 
   the most suitable persuasion strategy the agent should use next to move the conversation forward and 
   help the user make a confident insurance decision.
   
Conversation History:
User: Hi, I'm looking to get motor insurance for my new electric vehicle. It's a 2024 Tesla finetune_model 3.  
Agent: Great choice! The Tesla finetune_model 3 is an excellent vehicle. Since you've opted for an EV, are you particularly interested in coverage specific to electric vehicles, like battery protection?  
User: Yes, battery protection is definitely a concern. It's a big investment, and I want to make sure it's covered.  
Agent: Absolutely. The battery is the heart of your Tesla. With Tata AIG, you get rapid claims resolution combining thorough coverage with rapid claims resolution. It integrates technology with traditional risk management practices, ensuring that claims are processed quickly and effectively.  


Current User Utterance:
User: What kind of coverage options do you have specifically for EVs?


You must choose from the following six persuasion strategies, each defined with use cases and examples:

 Persuasion Strategies:
Credibility Appeal
Definition: Emphasize the insurance provider’s reputation, trustworthiness, or long-standing service.
Use when: The user is hesitant, asks about reliability, or mentions concern over service quality.
Example:
"New India Assurance has one of the widest repair networks in India and a proven record of settling claims efficiently."

Logical Appeal
Definition: Use facts, comparisons, benefits, or pricing logic to persuade.
Use when: The user is analytical, budget-conscious, or asking for details or comparisons.
Example:
"HDFC ERGO’s policy includes 24/7 support and zero-depreciation coverage, which means more savings during repairs."

Persona-Based Appeal
Definition: Match the policy features to the user’s lifestyle, habits, or profile.
Use when: The user reveals driving habits, tech-savviness, family needs, or risk aversion.
Example:
"Since you often drive long distances, Tata AIG’s Telematics-Based Monitoring suits your tech-savvy lifestyle."

Emotional Appeal
Definition: Tap into feelings like fear, safety, or care for loved ones.
Use when: The user talks about family, emergencies, peace of mind, or personal safety.
Example:
"Imagine a late-night breakdown—our 24/7 roadside assistance gives you and your family peace of mind."

Personal Appeal
Definition: Use positive sentiment, social proof, or popularity of the plan.
Use when: The user is unsure or looking for recommendations.
Example:
"This plan is one of our most popular choices—users love the smooth claims experience."

Default Persuasion Strategy
Definition: Use when little context is available. Provide neutral, factual reassurance.
Use when: The user is vague or hasn’t revealed any preferences or concerns.
Example:
"This policy offers protection against theft, accidents, and includes access to cashless repairs."

Instructions:
Given the current user utterance and the conversation history, perform the following:
Suggest the next best strategy that could be used.
Give a brief justification (1–2 lines max).

And please be brief.



 Few-Shot Examples
Example 1
User Utterance:
"Is this company actually reliable when it comes to claims?"
Future Strategy: Credibility Appeal
Justification: The user directly questions the insurer’s reliability — trust needs to be reinforced.

Example 2
User Utterance:
"I travel a lot for work, so I need something flexible."
Future Strategy: Persona-Based Appeal
Justification: The user has revealed lifestyle habits that allow for a tailored recommendation.

Example 3
User Utterance:
"What does the policy cover exactly?"
Future Strategy: Logical Appeal
Justification: The user is asking for objective, factual details.

Example 4
User Utterance:
"What if my car breaks down at night while I’m driving with my kids?"
Future Strategy: Emotional Appeal
Justification: The user is expressing concern for family and emergency scenarios.

Example 5
User Utterance:
"I’m just looking for something people usually go for."
Future Strategy: Personal Appeal
Justification: The user is undecided and seeking reassurance based on others’ choices.

Example 6
User Utterance:
"Okay, what are the basic features?"
Future Strategy: Default Persuasion Strategy
Justification: The user hasn’t shared enough context — a neutral overview is appropriate.

Output Format

Future Strategy: [One of the six strategies]
Justification: [1–2 line explanation]

Here is my input:{text_input}

"""

   return generate(prompt)

# %% [markdown]
# #### Keyterm Expert

# %%
def keyterms_expert(text_input: str) -> str:

   prompt = f"""You are a Keyterm Expert specializing in the motor insurance domain. 
   Your job is to analyze the user’s most recent utterance, using the conversation history for context, 
   and identify one or more important motor insurance-related keyterms mentioned (explicitly or implicitly) by the user.

Conversation History:
User: Hi, I'm looking to get motor insurance for my new electric vehicle. It's a 2024 Tesla finetune_model 3.  
Agent: Great choice! The Tesla finetune_model 3 is an excellent vehicle. Since you've opted for an EV, are you particularly interested in coverage specific to electric vehicles, like battery protection?  
User: Yes, battery protection is definitely a concern. It's a big investment, and I want to make sure it's covered.  
Agent: Absolutely. The battery is the heart of your Tesla. With Tata AIG, you get rapid claims resolution combining thorough coverage with rapid claims resolution. It integrates technology with traditional risk management practices, ensuring that claims are processed quickly and effectively.  


Current User Utterance:
User: What kind of coverage options do you have specifically for EVs?

These keyterms help the system focus the conversation, match features, and determine relevant coverages.

Examples of Common Keyterms (but not limited to):
Comprehensive coverage
Third-party liability
Roadside assistance
Zero depreciation / depreciation
Claim settlement
Battery protection
Own damage
Add-on cover
Telematics
Engine protection
EV (Electric Vehicle)
Repair network
Policy premium
Cashless garages
Deductibles
Policy renewal
Personal accident cover
IDV (Insured Declared Value)

You may also extract user-specific or vehicle-specific keyterms that are relevant to insurance decisions (e.g., “Tesla finetune_model 3,” “EV,” “2024 vehicle”).

Instructions:
From the current user utterance (with conversation history for context), do the following:
Extract all relevant keyterms mentioned or implied in the user's message.
For each keyterm, provide a brief 1-line justification for why it’s relevant in the motor insurance domain.

Few-Shot Examples

Example 1
User Utterance:
"What’s the premium for a 2024 Tesla finetune_model 3?"
Extracted Keyterms: Policy premium, 2024 Tesla finetune_model 3  
Justification: The user is asking for a cost estimate tied to a specific vehicle, both of which are essential for determining appropriate motor insurance coverage and pricing.

Example 2
User Utterance:
"Does this plan include accident and theft protection?"
Extracted Keyterms: Comprehensive coverage  
Justification: The user is inquiring about accident and theft protection, which are typically included under comprehensive coverage plans.

Example 3
User Utterance:
"What happens if my EV breaks down far from home?"
Extracted Keyterms: Roadside assistance, EV  
Justification: The user is describing a breakdown scenario involving an electric vehicle, which is directly relevant to roadside assistance coverage for EVs.

Example 4
User Utterance:
"Does this cover things like roadside help if I’m stuck somewhere?"
Extracted Keyterm: Roadside assistance  
Justification: The user is asking about support in case of breakdowns, which is typically handled under roadside assistance.

Example 5
User Utterance:
"I'm looking for something that includes coverage for theft and accidents."
Extracted Keyterm: Comprehensive coverage  
Justification: Coverage for both theft and accidents implies a comprehensive motor insurance policy.

Example 6
User Utterance:
"I want to make sure the battery is protected—it’s the most expensive part of the car."
Extracted Keyterm: Battery protection  
Justification: The user expresses concern about the EV battery, which is typically covered under specific EV-related add-ons.

Example 7
User Utterance:
"What’s the premium for a 2024 Tesla finetune_model 3?"
Extracted Keyterm: Policy premium  
Justification: The user is asking about cost, which relates directly to the insurance premium.  
:
Output Format
For extracted keyterm, provide the following:
Extracted Keyterm: [Term]  
Justification: [Brief reason why it's relevant to motor insurance]

Here is my input sentence:{text_input}

"""

   return generate(prompt)


# %% [markdown]
# #### Intern Expert

# %%
def intent_expert(text_input: str) -> str:

   prompt = f"""You are an Intent Expert for a virtual assistant specializing in motor insurance.
   Your job is to analyze the current user utterance, using the conversation history for context,
   and determine the single most relevant intent expressed by the user.

Conversation History:
User: Hi, I'm looking to get motor insurance for my new electric vehicle. It's a 2024 Tesla finetune_model 3.  
Agent: Great choice! The Tesla finetune_model 3 is an excellent vehicle. Since you've opted for an EV, are you particularly interested in coverage specific to electric vehicles, like battery protection?  
User: Yes, battery protection is definitely a concern. It's a big investment, and I want to make sure it's covered.  
Agent: Absolutely. The battery is the heart of your Tesla. With Tata AIG, you get rapid claims resolution combining thorough coverage with rapid claims resolution. It integrates technology with traditional risk management practices, ensuring that claims are processed quickly and effectively.  


Current User Utterance:
User: What kind of coverage options do you have specifically for EVs?


You must select from a fixed set of six pre-defined intents (listed below), each with clear definitions, examples, and triggers relevant to the motor insurance domain.

🎯 Available Intents:
Request_Insurance_Quote
Definition: The user initiates interest in getting a motor insurance quote or policy.
Example: "Hi, I'm looking to get motor insurance for my Tesla finetune_model 3."
Trigger: User starts a new request related to getting insured.

Ask_Coverage_Details
Definition: The user asks about what types of protection the insurance provides, especially for specific parts (e.g., battery, accidents, theft).
Example: "What kind of coverage options do you have specifically for the battery?"
Trigger: User inquires about included benefits, policy terms, or protections.

Express_Concern
Definition: The user shares a specific concern or priority about what needs to be protected or covered.
Example: "Yes, battery protection is definitely a concern for me."
Trigger: User highlights what matters most to them or expresses worry.

Request_Additional_Info
Definition: The user requests clarification or a deeper explanation of a feature or condition.
Example: "Do you cover accidents caused by the battery?"
Trigger: User follows up with questions or asks how something works.

Confirm_Interest
Definition: The user agrees, approves, or explicitly indicates they want to proceed.
Example: "That sounds good. I’d like to proceed."
Trigger: User shows intent to buy, continue, or finalize the service.

Ask_Price_or_Premium
Definition: The user wants to know the cost or breakdown of the insurance premium.
Example: "How much would that cost?"
Trigger: User inquires about price, discounts, or cost factors.



Instructions:
Given the conversation history and the user’s most recent message:
Identify the intent most clearly reflected in the current user utterance, based on the above definitions.
Provide a brief 1–2 line justification for your selection, grounded in the user’s phrasing and conversational context.

Few-Shot Examples
Example 1
User Utterance:
"Hi, I'm looking to get insurance for my new Tesla."
Intent: Request_Insurance_Quote  
Justification: The user is initiating a conversation to obtain motor insurance for a specific vehicle.

Example 2
User Utterance:
"Do you cover damage to the battery?"
Intent: Ask_Coverage_Details  
Justification: The user is asking about a specific type of coverage related to their EV battery.

Example 3
User Utterance:
"Battery protection is definitely a concern for me."
Intent: Express_Concern  
Justification: The user is explicitly stating a personal worry or priority regarding coverage.

Example 4
User Utterance:
"Can you explain how the battery coverage works?"
Intent: Request_Additional_Info  
Justification: The user is asking for clarification or further explanation of a feature already mentioned.

Example 5
User Utterance:
"That sounds good. I’m ready to go ahead."
Intent: Confirm_Interest  
Justification: The user is showing a clear desire to move forward with the policy or service.

Example 6
User Utterance:
"How much will that cost me annually?"
Intent: Ask_Price_or_Premium  
Justification: The user is directly asking about the premium or cost of the insurance policy.

Output Format

Intent: [One of the six predefined intents]  
Justification: [1–2 line explanation of why this intent matches the user's message]
Here is my input:{text_input}
"""

   return generate(prompt)


# %% [markdown]
# ### Selecting Expert

# %%
''''using the fine tuned finetune_model'''
import re
def route_experts(sentence: str) -> list:
    prompt = f"""
You are an intelligent router that analyzes ongoing insurance conversations and activates only the most relevant expert(s) needed to support the next response. 
Use the conversation history to understand the context and evaluate the current user utterance. 
Select expert(s) based on what would best support crafting an effective, accurate, and customer-focused agent reply.

You MUST choose from the following list:
1 Intent Expert  
2 Keyterm Expert  
3 Persuasion Expert  
4 Sentiment Expert  

You may select 1, several, or all 4 — but only those that are clearly needed based on the text.

Always respond in **this exact format**:
Input: [original sentence]  
Selected Experts: [Expert1, Expert2, etc]  
Reason: [one sentence explaining why those experts were selected]

Below are few-shot examples to help you understand the format and reasoning:

Example #1  
Input: Can someone please help me reset my password?  
Selected Experts: [Intent Expert, Keyterm Expert]  
Reason: The sentence expresses a help request (intent) and refers to a specific technical issue (keyterm).

Example #2  
Input: This app is a complete disaster. It crashes every time I try to open it.  
Selected Experts: [Intent Expert, Sentiment Expert, Keyterm Expert]  
Reason: This is a complaint (intent), shows strong negative emotion (sentiment), and mentions specific app-related terms (keyterm).

Example #3  
Input: Reset password link not working again.  
Selected Experts: [Keyterm Expert]  
Reason: The sentence includes factual technical content focusing on a specific feature (keyterm).

Example #4  
Input: I love how smooth the new interface feels – you guys nailed it!  
Selected Experts: [Sentiment Expert, Persuasion Expert]  
Reason: The sentence conveys positive emotion (sentiment) and includes praise that reinforces the user’s positive perception (persuasion).

### Now process the following:
Input: {sentence}
"""

    try:

        response = finetune_generate(prompt)

        # response = finetune_model.generate_content(prompt).text.strip()
        selected_experts = []

        # Try regex to match the experts list
        match = re.search(r"Selected Experts:\s*\[(.*?)\]", response)
        if match:
            items = match.group(1).split(',')
            selected_experts = [item.strip().strip('"\'').lower() for item in items if item.strip()]

        return selected_experts
    except Exception as e:
        print("Error routing experts:", e)
        return []
    prompt = f"""
You are a well-trained expert selector.
Your job is to analyze a given input sentence and decide which expert modules should be activated, based on what the speaker is expressing or trying to do.

Available experts:
- Intent Expert: For purpose, request, question, or user goal
- Keyterm Expert: For extracting topic-specific or important terms
- Persuasion Expert: For emotional, persuasive, or rhetorical language
- Sentiment Expert: For emotional tone (positive, negative, or neutral)

Select ONLY the necessary experts based on content. Return 1, 2, 3, or 4 depending on relevance. Do NOT include experts unnecessarily.

### Output Format
Input: [sentence]
Selected Experts: [Expert1, Expert2, ...]
Reason: [Short explanation]

### Examples

Input: Can someone please help me reset my password?
Selected Experts: [Intent Expert, Keyterm Expert]
Reason: Request for help (intent), contains topic terms ("reset password")

Input: This app is a complete disaster. It crashes every time I try to open it.
Selected Experts: [Intent Expert, Sentiment Expert, Keyterm Expert]
Reason: Complaint (intent), frustration (sentiment), key terms mentioned

Input: Reset password link not working again.
Selected Experts: [Keyterm Expert]
Reason: Technical/factual content only

Input: {sentence}
"""

    # Generate response

    response = generate(prompt)

    # Extract list from "Selected Experts:"
    selected_experts = []
    for line in response.splitlines():
        if line.startswith("Selected Experts:"):
            try:
                raw = line.split(":", 1)[1].strip()
                expert_list = eval(raw)  # turns '[Intent Expert, Keyterm Expert]' into list
                selected_experts = [e.lower() for e in expert_list]
            except:
                pass
            break

    return selected_experts


# ---------- Synthesis Function ----------
def generate_combined_analysis(dialogue, intent=None, key=None, persu=None, senti=None):
    prompt = f"""You are a trained virtual support agent.
You are an Aggregator in a motor insurance virtual assistant.
You synthesize the outputs from domain-specific expert modules to generate a brief, clear, and personalized response as a professional insurance agent would.

You are given:

The conversation history

The current user utterance

A subset of outputs from the following possible experts (some may be missing):

Available Expert Modules:
- Intent: What the user wants or is trying to do  
- Keyterms: Important phrases or topics mentioned  
- Sentiment: The emotional tone of the message  
- Persuasion: How the user tries to express or influence  

**Strict Guidelines:**
- Always write your response as if you're a real human agent—empathetic, clear, and helpful.
- Never include or reference the original dialogue or the expert outputs in your reply.
- Use only the experts provided—do not invent or assume missing ones.
- Do not describe or explain expert analyses.
- Return **only the final agent reply**—no headings, formatting, or additional text.

Your tone should:
- Acknowledge and validate the user’s experience  
- Provide support, next steps, or context where needed  
- Persuade gently when relevant, always staying respectful  
- Maintain professionalism, regardless of tone or emotion

–––– Few-Shot Examples ––––

### Example 1 – 1 Expert Output (Intent only)
Conversation History:
User: Hi, I’m thinking of switching my car insurance provider.

Current User Utterance:
User: What are your premium rates like for SUVs?

Expert Outputs:
Intent: Ask_Pricing

Output (Aggregator Response):
Our premium rates for SUVs vary depending on the finetune_model, age of the vehicle, and coverage level you choose. I can guide you through a quick quote to give you an exact figure.


### Example 2 – 2 Expert Outputs (Intent + Sentiment)
Conversation History:
User: I recently had a minor accident, and the process with my last insurer was very stressful.

Current User Utterance:
User: How does your claim process work?

Expert Outputs:
Intent: Ask_Claim_Process
Sentiment: Frustrated/Concerned

Output (Aggregator Response):
I’m sorry to hear that your last experience was stressful. At our company, we focus on a smooth and fast claim process—most minor accident claims can be settled quickly, with minimal paperwork.


### Example 3 – 3 Expert Outputs (Intent + Keyterms + Persuasion)
Conversation History:
User: I just bought a new hybrid SUV, and I want something reliable.

Current User Utterance:
User: Can I get coverage for both battery and engine damage?

Expert Outputs:
Intent: Ask_Coverage_Details
Keyterms: Hybrid SUV, Battery coverage, Engine coverage
Persuasion: Logical appeal (User seeks clear details)

Output (Aggregator Response):
Yes, our hybrid insurance plan covers both the vehicle’s engine and the battery. You’ll also be protected against accidental damage, theft, and third-party liability, giving you complete peace of mind.


### Example 4 – 4 Expert Outputs (All)
Conversation History:
User: I need to renew my car insurance soon. My current provider keeps increasing the price.

Current User Utterance:
User: Can you offer me a better deal if I switch to you?

Expert Outputs:
Intent: Ask_Discount_Or_Switch_Offer
Keyterms: Renewal, Switching provider, Discount
Sentiment: Hopeful but skeptical
Persuasion: Negotiation appeal

Output (Aggregator Response):
We’d be happy to help you switch and save. If you move your policy to us, you could qualify for a loyalty discount and additional benefits like zero-depreciation coverage. I can prepare a customized quote for you today.


Now, using the insights below, respond like a real agent would.

**Important: Do not repeat or refer to the dialogue or expert outputs.  
Return only the final agent-style response. Nothing else.**

Dialogue: {dialogue}  
Intent: {intent}  
Keyterms: {key}  
Sentiment: {senti}  
Persuasion: {persu}  

Agent Reply:"""

    return generate(prompt)


# ---------- Main Selector Function ----------
def process_input_with_selector_finetune_model(sentence: str) -> str:
    selected_experts = route_experts(sentence)
    print(f"Selected Experts: {selected_experts}")

    # Initialize all expert variables
    intent = keyterms = sentiment = persuasion = None

    # Normalize expert names for safety
    selected_experts = [e.lower() for e in selected_experts]

    # Call only selected experts
    if "intent expert" in selected_experts:
        intent = intent_expert(sentence)
        print("Intent Expert O/p:--->",intent)
    if "keyterm expert" in selected_experts:
        keyterms = keyterms_expert(sentence)
        print("Keyterm Expert O/p:--->",keyterms)
    if "sentiment expert" in selected_experts:
        sentiment = sentiment_expert(sentence)
        print("Sentiment Expert O/p:--->",sentiment)
    if "persuasion expert" in selected_experts:
        persuasion = persuassion_expert(sentence)
        print("Persuasion Expert O/p:--->",persuasion)


    # Combine everything
    return generate_combined_analysis(
        dialogue=sentence,
        intent=intent,
        key=keyterms,
        persu=persuasion,
        senti=sentiment
    )



# %%
import pandas as pd

def create_dataset():

    # Make sure your CSV has the columns: 'conversation_id', 'turn_no', 'utterance', 'new_agent_reply'
    df = pd.read_csv('/home/rohank__iitp/Work/niladri/test_baseline dataset/train_conversation.csv')
    # --- Response Generation and Incremental Saving ---

    if not df.empty:
        output_filename = '/home/rohank__iitp/Work/niladri/model/model_our_dataset.csv'
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
                prompt = "Conversation History:\n" + conversation_history + "Current Utterance: " + f"User: {row['utterance']}\nAgent:"
                # Your debugging print statements
                print("========================================================================================================================================")
                print(f"Generating for conv_id: {row['conversation_id']}, turn: {row['turn_no']}\nPROMPT:\n{prompt}")
                print("========================================================================================================================================")
                
                
                
                # Generate the response
                '''Change Here😆😆😆😆'''
                qwen_response = process_input_with_selector_finetune_model(prompt)
                
                
                
                
                # Create a dictionary from the original row and add the new response
                current_row_data = row.to_dict()
                current_row_data['Qween Router Response'] = qwen_response
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


