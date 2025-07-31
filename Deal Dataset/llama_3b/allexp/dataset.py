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

   prompt = f"""You are a Persuasion Strategy Selector for a travel recommendation dialogue system. 
Based on the user's most recent utterance and the conversation history, you must recommend 
the most suitable persuasion strategy the travel agent should use next to move the conversation forward 
and help the user confidently choose a travel experience.

Conversation History:
User: Hi, I’m planning a vacation in Europe next spring, and I’m thinking about visiting Italy.  
Agent: Wonderful! Italy is full of culture and amazing food. Are you more interested in exploring historic cities like Rome and Florence or relaxing in scenic coastal spots like the Amalfi Coast?  
User: I think I’d like to see both, but I’m a bit worried about managing time between locations.  
Agent: That makes sense. Many travelers enjoy combining a few days in Rome with a short scenic train trip to the Amalfi Coast, which balances sightseeing with relaxation.  

Current User Utterance:
User: What are some travel packages that include both Rome and the Amalfi Coast?

You must choose from the following six persuasion strategies, each defined with use cases and examples:

Persuasion Strategies:
Credibility Appeal  
Definition: Emphasize the travel company’s reputation, expertise, or trusted service.  
Use when: The user is hesitant, asks about reliability, or mentions concerns about planning or quality.  
Example:  
"Our agency has helped over 50,000 travelers experience Italy with top-rated local guides."

Logical Appeal  
Definition: Use facts, comparisons, itineraries, or value-for-money reasoning to persuade.  
Use when: The user is analytical, budget-conscious, or asking for detailed options.  
Example:  
"This 7-day package includes 3 nights in Rome and 3 in Amalfi, plus high-speed train transfers—saving you both time and cost."

Persona-Based Appeal  
Definition: Match the travel plan to the user’s interests, habits, or travel style.  
Use when: The user reveals preferences like adventure, relaxation, family travel, or cultural focus.  
Example:  
"Since you enjoy both history and scenic views, this tour combines ancient Rome exploration with peaceful coastal days."

Emotional Appeal  
Definition: Tap into feelings like excitement, relaxation, or creating memorable experiences.  
Use when: The user talks about dreams, bucket lists, family bonding, or once-in-a-lifetime experiences.  
Example:  
"Imagine savoring gelato on the Spanish Steps, then watching the sunset over the Amalfi Coast."

Personal Appeal  
Definition: Use positive sentiment, social proof, or popularity of a trip.  
Use when: The user is unsure or looking for recommendations.  
Example:  
"This is one of our most booked Italy packages—travelers love the perfect balance of city and coast."

Default Persuasion Strategy  
Definition: Use when little context is available. Provide a neutral, reassuring recommendation.  
Use when: The user is vague or hasn’t shared preferences yet.  
Example:  
"We offer a variety of Italy trips with both guided and flexible options to suit different travel styles."

Instructions:
Given the current user utterance and the conversation history, perform the following:  
- Suggest the next best strategy that could be used.  
- Give a brief justification (1–2 lines max).  

And please be brief.

Few-Shot Examples  
Example 1  
User Utterance:  
"Is your agency trustworthy for planning international trips?"  
Future Strategy: Credibility Appeal  
Justification: The user is questioning reliability—trust needs to be reinforced.

Example 2  
User Utterance:  
"I love photography and want scenic spots."  
Future Strategy: Persona-Based Appeal  
Justification: The user’s travel style invites a tailored recommendation.

Example 3  
User Utterance:  
"What’s included in the 7-day package?"  
Future Strategy: Logical Appeal  
Justification: The user is seeking detailed, factual information.

Example 4  
User Utterance:  
"I just want a relaxing trip where I can unwind by the ocean."  
Future Strategy: Emotional Appeal  
Justification: The user is expressing a desire for a specific emotional experience.

Example 5  
User Utterance:  
"What trips do most people choose?"  
Future Strategy: Personal Appeal  
Justification: The user is seeking reassurance through popularity.

Example 6  
User Utterance:  
"Okay, what trips do you offer in Italy?"  
Future Strategy: Default Persuasion Strategy  
Justification: The user hasn’t provided enough context for a specific strategy.

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

   prompt = f"""You are a Keyterm Expert specializing in the tourism and travel domain. 
Your job is to analyze the user’s most recent utterance, using the conversation history for context, 
and identify one or more important travel- or tourism-related keyterms mentioned (explicitly or implicitly) by the user.

Conversation History:
User: Hi, I'm planning a vacation to Italy next summer.  
Agent: That sounds amazing! Are you looking for cultural experiences like museums and historical tours, or more of a food and wine trip?  
User: I’m really into art and history, but I’d also love to enjoy some authentic Italian cuisine.  
Agent: Perfect! Florence and Rome are incredible for art and history, and the local trattorias will give you the best culinary experience.  

Current User Utterance:
User: Can you suggest some guided tours for exploring the art museums in Florence?

These keyterms help the system focus the conversation, match travel recommendations, and determine relevant experiences or destinations.

Examples of Common Keyterms (but not limited to):
Historic landmarks
City tours
Museum passes
Cultural experiences
Food and wine tours
Adventure activities
Local guides
Guided tours
Travel packages
Popular attractions
Hidden gems
Day trips
Seasonal events
Art museums
Walking tours
City passes
Scenic routes
Destination-specific terms (e.g., Florence, Amalfi Coast, Tuscany)

You may also extract user-specific or destination-specific keyterms that are relevant to tourism decisions (e.g., “Florence,” “art museums,” “Italian cuisine”).

Instructions:
From the current user utterance (with conversation history for context), do the following:
Extract all relevant keyterms mentioned or implied in the user's message.
For each keyterm, provide a brief 1-line justification for why it’s relevant in the tourism domain.

Few-Shot Examples

Example 1
User Utterance:
"What’s the best way to see all the major landmarks in Paris?"
Extracted Keyterms: City tours, Popular attractions, Paris  
Justification: The user wants to see famous sites, implying a need for guided or structured tours in Paris.

Example 2
User Utterance:
"Are there any local guides for exploring Tuscany’s vineyards?"
Extracted Keyterms: Local guides, Food and wine tours, Tuscany  
Justification: The user is interested in local experiences and wine tourism specific to Tuscany.

Example 3
User Utterance:
"I want to visit the Colosseum and Vatican in one day."
Extracted Keyterms: Historic landmarks, Day trips, Rome  
Justification: The user is referencing iconic historical attractions and a single-day sightseeing plan in Rome.

Example 4
User Utterance:
"Are there any walking tours that focus on Florence’s art scene?"
Extracted Keyterms: Walking tours, Art museums, Florence  
Justification: The user is asking about art-focused walking tours in Florence, which are key tourism activities.

Example 5
User Utterance:
"I’d like to experience a local food festival in Spain."
Extracted Keyterms: Seasonal events, Local cuisine, Spain  
Justification: The user wants to attend a cultural event tied to food in Spain.

Output Format
For each extracted keyterm, provide the following:
Extracted Keyterm: [Term]  
Justification: [Brief reason why it's relevant to tourism]

Here is my input sentence:{text_input}
"""

   return generate(prompt)


# %% [markdown]
# #### Intern Expert

# %%
def intent_expert(text_input: str) -> str:

   prompt = f"""You are an Intent Expert for a virtual travel assistant specializing in tourism and trip planning.
   Your job is to analyze the current user utterance, using the conversation history for context,
   and determine the single most relevant travel intent expressed by the user.

Conversation History:
User: Hi, I'm planning a vacation to Italy next spring. I really want to explore the countryside and some famous cities.  
Agent: That sounds wonderful! Are you interested in guided tours, or do you prefer a more self-paced experience?  
User: I think a mix of both. I also love trying local food and cultural experiences.  
Agent: Great! We have several packages that combine sightseeing with culinary tours and hands-on cultural activities.  

Current User Utterance:
User: What kind of tour options do you have specifically for Tuscany?

You must select from a fixed set of six pre-defined intents (listed below), each with clear definitions, examples, and triggers relevant to the tourism domain.

🎯 Available Intents:
Request_Travel_Package
Definition: The user expresses interest in booking or learning about a travel package or trip plan.
Example: "Hi, I'm planning a trip to Italy next spring."
Trigger: User starts a new request related to a destination or travel plan.

Ask_Tour_Details
Definition: The user asks about what experiences, tours, or activities are available in a destination.
Example: "What kind of tours do you have in Tuscany?"
Trigger: User inquires about available itineraries, sightseeing options, or activities.

Express_Travel_Preference
Definition: The user shares a specific interest, priority, or travel style preference.
Example: "I really want to try local food and visit historical sites."
Trigger: User highlights personal preferences, interests, or goals for their trip.

Request_Additional_Info
Definition: The user asks for clarification or a deeper explanation of a tour, activity, or service.
Example: "Can you explain how the wine tasting tour works?"
Trigger: User requests more details or asks how something in the travel plan works.

Confirm_Interest
Definition: The user agrees, approves, or explicitly indicates they want to proceed with a plan.
Example: "That sounds amazing. I’d like to book it."
Trigger: User shows clear intent to book, continue, or finalize the trip plan.

Ask_Price_or_Cost
Definition: The user asks about the cost or breakdown of a travel package or experience.
Example: "How much would that Tuscany tour cost?"
Trigger: User inquires about pricing, deals, or cost factors.

Instructions:
Given the conversation history and the user’s most recent message:
Identify the intent most clearly reflected in the current user utterance, based on the above definitions.
Provide a brief 1–2 line justification for your selection, grounded in the user’s phrasing and conversational context.

Few-Shot Examples
Example 1
User Utterance:
"Hi, I want to book a vacation to Japan next summer."
Intent: Request_Travel_Package  
Justification: The user is initiating a conversation to book or learn about a travel package for a destination.

Example 2
User Utterance:
"Do you have any day trips from Florence?"
Intent: Ask_Tour_Details  
Justification: The user is asking about available tours in a specific location.

Example 3
User Utterance:
"I really love hiking and nature photography."
Intent: Express_Travel_Preference  
Justification: The user is sharing a specific travel interest that will shape their itinerary.

Example 4
User Utterance:
"Can you explain how the cooking class tour works?"
Intent: Request_Additional_Info  
Justification: The user is asking for details about a specific experience mentioned in the conversation.

Example 5
User Utterance:
"That sounds perfect. Let’s go ahead and book it."
Intent: Confirm_Interest  
Justification: The user shows a clear intent to proceed with the offered travel plan.

Example 6
User Utterance:
"How much would the Venice boat tour cost per person?"
Intent: Ask_Price_or_Cost  
Justification: The user is directly asking about the cost of the travel experience.

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

    prompt = f"""You are an advanced virtual travel assistant trained to generate professional, friendly, and natural-sounding responses for users seeking tourism guidance.  
You receive internal insights from eight expert systems for each user input:

- Intent Expert: Understands what the traveler is asking or expressing  
- Key Term Expert: Extracts main destinations, activities, or travel-related keywords  
- Sentiment Expert: Evaluates the user’s emotional tone (excited, worried, curious, etc.)  
- Persuasion Expert: Identifies emotional or rhetorical tactics used  
- analyze_text: Provides part-of-speech tags and named entities (like city names or landmarks)  
- detect_language: Detects the user’s input language  
- get_dependencies: Analyzes sentence structure and word relationships  
- get_SVO_string: Extracts subject–verb–object relations (e.g., Relation: (user, wants, city tour))

Your job is to synthesize **all expert insights** internally and generate one natural, traveler-friendly response — **never reveal or explain the expert outputs**.

### Travel Agent Response Guidelines:
- Sound warm, helpful, and approachable  
- Acknowledge and validate the traveler’s feelings or interests  
- Offer guidance, suggestions, or clarity when helpful  
- Keep the tone human and conversational, never robotic or overly technical  
- Do not list expert outputs or analyze the user’s message in the reply  

–––– Few-Shot Tourism Examples ––––  

Example 1  
Dialogue: "I’m visiting Paris for the first time and don’t want to miss anything important."  
Intent: Seeking recommendations  
Keyterms: "Paris", "first time", "important attractions"  
Sentiment: Excited  
Persuasion: None  

Response:  
That sounds amazing! For a first visit, I’d suggest exploring the Eiffel Tower, Louvre Museum, and a sunset walk along the Seine. A river cruise or a stop at a local café will really give you that classic Parisian experience.

---

Example 2  
Dialogue: "I’m nervous about traveling to Tokyo because I don’t speak Japanese."  
Intent: Expressing concern  
Keyterms: "Tokyo", "don’t speak Japanese"  
Sentiment: Worried  
Persuasion: None  

Response:  
It’s completely normal to feel that way! The good news is Tokyo is very traveler-friendly, with plenty of English signs and helpful locals. Using a translation app can also make your trip much smoother.

---

Example 3  
Dialogue: "Are there any hidden beaches in Bali that tourists don’t usually know about?"  
Intent: Seeking insider tips  
Keyterms: "hidden beaches", "Bali"  
Sentiment: Curious  
Persuasion: None  

Response:  
Absolutely! While places like Kuta and Seminyak are popular, you might love beaches like Nyang Nyang or Green Bowl for a quieter experience. They’re a bit off the beaten path but offer stunning views.

---

Now, generate a final, human-like response for the following input.  
Use all expert insights internally but **do not include or reference them** directly. Only output the final response.

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
    df = pd.read_csv('/home/rohank__iitp/Work/niladri/Deal Dataset/deal dataset.csv')
    # --- Response Generation and Incremental Saving ---

    if not df.empty:
        output_filename = '/home/rohank__iitp/Work/niladri/Deal Dataset/llama_3b/allexp/llama_3b_allexp_dataset.csv'
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
                current_row_data['llama_3b Allexp Response'] = qwen_response
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


