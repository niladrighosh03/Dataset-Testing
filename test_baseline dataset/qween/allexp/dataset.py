# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Loading model...")  
model_name = "/scratch/rohank__iitp/qwen2_5_7b_instruct"

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

generate("What is the capital of France?")

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
User: Hi, I'm looking to get motor insurance for my new electric vehicle. It's a 2024 Tesla Model 3.  
Agent: Great choice! The Tesla Model 3 is an excellent vehicle. Since you've opted for an EV, are you particularly interested in coverage specific to electric vehicles, like battery protection?  
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
User: Hi, I'm looking to get motor insurance for my new electric vehicle. It's a 2024 Tesla Model 3.  
Agent: Great choice! The Tesla Model 3 is an excellent vehicle. Since you've opted for an EV, are you particularly interested in coverage specific to electric vehicles, like battery protection?  
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

You may also extract user-specific or vehicle-specific keyterms that are relevant to insurance decisions (e.g., “Tesla Model 3,” “EV,” “2024 vehicle”).

Instructions:
From the current user utterance (with conversation history for context), do the following:
Extract all relevant keyterms mentioned or implied in the user's message.
For each keyterm, provide a brief 1-line justification for why it’s relevant in the motor insurance domain.

Few-Shot Examples

Example 1
User Utterance:
"What’s the premium for a 2024 Tesla Model 3?"
Extracted Keyterms: Policy premium, 2024 Tesla Model 3  
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
"What’s the premium for a 2024 Tesla Model 3?"
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
User: Hi, I'm looking to get motor insurance for my new electric vehicle. It's a 2024 Tesla Model 3.  
Agent: Great choice! The Tesla Model 3 is an excellent vehicle. Since you've opted for an EV, are you particularly interested in coverage specific to electric vehicles, like battery protection?  
User: Yes, battery protection is definitely a concern. It's a big investment, and I want to make sure it's covered.  
Agent: Absolutely. The battery is the heart of your Tesla. With Tata AIG, you get rapid claims resolution combining thorough coverage with rapid claims resolution. It integrates technology with traditional risk management practices, ensuring that claims are processed quickly and effectively.  


Current User Utterance:
User: What kind of coverage options do you have specifically for EVs?


You must select from a fixed set of six pre-defined intents (listed below), each with clear definitions, examples, and triggers relevant to the motor insurance domain.

🎯 Available Intents:
Request_Insurance_Quote
Definition: The user initiates interest in getting a motor insurance quote or policy.
Example: "Hi, I'm looking to get motor insurance for my Tesla Model 3."
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
# ### Extra 5 tools as expert

# %% [markdown]
# #### 1)NER & POS

# %%
import spacy

# %%
# import spacy
# Load English model
nlp = spacy.load("en_core_web_sm")

# %%
def analyze_text(sentence):
    """
    Analyze a sentence for POS tagging and Named Entity Recognition,
    and return the results as a formatted string.
    
    Parameters:
    sentence (str): The input sentence to analyze.
    
    Returns:
    str: Formatted string with POS tags and Named Entities.
    """
    doc = nlp(sentence)
    result = []

    # POS tagging
    result.append("Part-of-Speech Tags:")
    for token in doc:
        result.append(f"{token.text} -> {token.pos_} ({token.tag_})")

    # Named Entity Recognition
    result.append("\nNamed Entities:")
    for ent in doc.ents:
        result.append(f"{ent.text} -> {ent.label_}")

    return "\n".join(result)

# analyze_text("I like cricket")

# %% [markdown]
# #### 2) Language Detection

# %%
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # For consistent results
import re

# %%
def detect_language(text):
    try:
        language = detect(text)
        language= 'Detected language is: ' + language
        return language
    except:
        return "Could not detect language"
detect_language("This is an English sentence.")

# %% [markdown]
# #### 3) Dependency persing

# %%
def get_dependencies(sentence):

    doc = nlp(sentence)
    
    # Build plain-text dependency list
    lines = ["Token        Dep          Head"]
    for token in doc:
        lines.append(f"{token.text:<12} -> {token.dep_:<12} -> {token.head.text}")
    
    return "\n".join(lines)

# Example usage
output = get_dependencies("The quick brown fox jumps over the lazy dog.")
print(output)

# %% [markdown]
# #### 4)Relation Extraction

# %%
def get_SVO_string(text):
    """
    Extract (Subject, Verb, Object) triples from input text and return them as a formatted string.

    Parameters:
    text (str): Input sentence or paragraph.

    Returns:
    str: SVO relations, one per line. Returns a message if no SVO found.
    """
    doc = nlp(text)
    triples = []

    for token in doc:
        if token.pos_ != "VERB":
            continue

        subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
        if not subjects:
            continue

        objects = [w for w in token.rights if w.dep_ == "dobj"]

        for prep in (w for w in token.rights if w.dep_ == "prep"):
            objects.extend([w for w in prep.rights if w.dep_ == "pobj"])

        objects.extend([w for w in token.rights if w.dep_ == "attr"])

        if subjects and objects:
            for s in subjects:
                for o in objects:
                    triples.append(f"Relation: ({s.text}, {token.lemma_}, {o.text})")

    return "\n".join(triples) if triples else "No Subject–Verb–Object relations found."

# Example usage
text = "Hi, I am interested in getting motor insurance for my bike. I just bought a new 2024 Royal Enfield Classic 350."
get_SVO_string(text)


# %% [markdown]
# ### Combine output

# %%
def generate_combined_analysis(
    dialogue: str,
    intent_output: str,
    keyterms_output: str,
    persuasion_output: str,
    sentiment_output: str,
    analyze_text_output: str,
    language_output: str,
    dependencies_output: str,
    svo_output: str
) -> str:

    prompt = f"""You are an advanced language model trained to generate professional, helpful, and natural-sounding agent responses.  
You receive internal insights from eight expert systems for a single user input:

- Intent Expert: Understands what the user is trying to express or request  
- Key Term Expert: Extracts main concepts and keywords  
- Sentiment Expert: Evaluates the emotional tone (positive, negative, skeptical, etc.)  
- Persuasion Expert: Identifies emotional or rhetorical tactics used  
- analyze_text: Provides part-of-speech tags and named entities  
- detect_language: Detects the user's input language  
- get_dependencies: Analyzes sentence structure and word relationships  
- get_SVO_string: Extracts subject–verb–object relations (e.g., Relation: (user, wants, feature))

Your task is to use **all expert insights** internally to craft one final, human-sounding response — **never repeat or explain the expert outputs**.

### Agent Response Guidelines:
- Be warm, empathetic, and respectful  
- Acknowledge and validate the user's emotion or concern  
- Offer context or clarity when helpful  
- Never sound robotic, technical, or condescending  
- Do not list points or restate expert content — just speak naturally

–––– Few-Shot Examples ––––

Example 1  
Dialogue: "I think electric cars are overrated and not really helping the environment."  
Intent: Opinion  
Keyterms: "electric cars", "overrated", "helping the environment"  
Sentiment: Skeptical  
Persuasion: Generalization  
analyze_text:  
Part-of-Speech Tags:\nI -> PRON (PRP)\nthink -> VERB (VBP)\nelectric -> ADJ (JJ)...  
Named Entities: None  
get_SVO_string: Relation: (cars, are, overrated)  

Response:  
Thank you for sharing your view — it’s completely valid to question the impact of electric vehicles. While no solution is perfect, many studies show EVs tend to produce fewer emissions over time, especially when powered by renewables.

---

Example 2  
Dialogue: "AI is going to take over every job and make humans useless."  
Intent: Expressing concern  
Keyterms: "AI", "every job", "humans useless"  
Sentiment: Negative  
Persuasion: Exaggeration  
detect_language: English  
get_SVO_string: Relation: (AI, take over, job)  

Response:  
I understand how that sounds — AI’s progress can feel overwhelming. But rather than replacing people, it’s often designed to work alongside us, creating new kinds of jobs and ways of working that didn’t exist before.

---

Example 3  
Dialogue: "Can you guys add dark mode to the settings?"  
Intent: Feature request  
Keyterms: "dark mode", "settings"  
Sentiment: Neutral  
analyze_text:  
Part-of-Speech Tags:\nCan -> AUX (MD)\nyou -> PRON (PRP)\nguys -> NOUN (NNS)\nadd -> VERB (VB)...  
get_SVO_string: Relation: (you, add, dark mode)  

Response:  
Thanks for the suggestion — dark mode is a popular request and makes a lot of sense. I’ll pass this along to our team for consideration.

---

Now generate a final, agent-like response for the following input.  
Use all expert insights provided, but **do not include or reference them** directly. Only output the final response.

Dialogue: "{dialogue}"  
Intent: {intent_output}  
Keyterms: {keyterms_output}  
Persuasion: {persuasion_output}  
Sentiment: {sentiment_output}  
analyze_text: {analyze_text_output}  
detect_language: {language_output}  
get_dependencies: {dependencies_output}  
get_SVO_string: {svo_output}  

Response:
"""

    return generate(prompt)


# %%
import pandas as pd

def create_dataset():

    # Make sure your CSV has the columns: 'conversation_id', 'turn_no', 'utterance', 'new_agent_reply'
    df = pd.read_csv('/home/rohank__iitp/Work/niladri/test_baseline dataset/train_conversation.csv')
    # --- Response Generation and Incremental Saving ---

    if not df.empty:
        output_filename = '/home/rohank__iitp/Work/niladri/test_baseline dataset/qween/allexp/qween_allexp_dataset.csv'
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
                
                # qwen_response = process_input_with_selector_model(prompt)
                persuasion_output=persuassion_expert(sentence)
                sentiment_output = sentiment_expert(sentence)
                keyterms_output = keyterms_expert(sentence)
                intent_output = intent_expert(sentence)
                
                #new expert tools
                analyze_text_output = analyze_text(sentence)
                detect_language_output = detect_language(sentence)
                get_dependencies_output = get_dependencies(sentence)
                get_SVO_output = get_SVO_string(sentence)
                
                qwen_response = generate_combined_analysis(
                dialogue=sentence,
                intent_output=intent_output,
                keyterms_output=keyterms_output,
                persuasion_output=persuasion_output,
                sentiment_output=sentiment_output,
                analyze_text_output=analyze_text_output,
                language_output=detect_language_output,
                dependencies_output=get_dependencies_output,
                svo_output=get_SVO_output)
                
                
                
                # Create a dictionary from the original row and add the new response
                current_row_data = row.to_dict()
                current_row_data['Qween Allexp Response'] = qwen_response
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


