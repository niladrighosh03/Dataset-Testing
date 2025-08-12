# %%
print("Loading model and tokenizer...")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "/scratch/rohank__iitp/Phi-3-medium-128k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# %%
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.7,
    "do_sample": True,
}

# %%
def generate(sentence:str):
    prompt = f"""You are a Persuasion Strategy labeling expert.
    You are given an utterance.  
Your task is to identify and label the persuasion strategy it uses from the defined list of six strategies.  
You must also clearly explain why that strategy was chosen, and your explanation MUST be enclosed inside <reason> and </reason> tags exactly.  



 Guidelines for labeling:  
- Read the utterance carefully and determine its primary persuasion approach.  
- Select the persuasion strategy whose description aligns most closely with the utterance's content and intent.  
- When writing the reasoning, explain the specific features, keywords, or message style in the utterance that make it fit the chosen persuasion strategy.  
- The reasoning MUST be enclosed inside <reason> and </reason> tags — this is mandatory.  
- Be concise but clear, ensuring the explanation makes it obvious why that persuasion type applies.  



### Persuasion strategies:  

1. Logical appeal – Uses factual reasoning, statistics, verifiable claims, comparisons, or cost–benefit arguments to persuade.  
   - Even if one factual statement or claim is present (e.g., a price comparison, performance data, or measurable outcome), it qualifies as logical appeal, unless another strategy’s intent is more dominant.  
   - Examples: “ Welcome! I'm glad to help you with motor insurance for your new Tesla Model 3. Our policies offer comprehensive coverage for electric vehicles, including battery protection. Let's discuss the options that best suit your needs.” / “ We have several add-on covers that can enhance your insurance policy for the Royal Enfield Himalayan. Consider options like engine protection, zero depreciation, and roadside assistance to ensure comprehensive coverage for your bike. Let's discuss which add-ons would be most beneficial for you.”  

2. Credibility appeal – Builds trust by emphasizing the trustworthiness, expertise, qualifications, or proven track record of the provider or source.  
   - This includes references to awards, rankings, certifications, industry recognition, or customer satisfaction ratings.  
   - Example indicators: “ Bajaj Allianz is a trusted name in the insurance industry, known for its comprehensive coverage and quick claim settlement process. They offer a range of motor insurance plans tailored to meet the specific needs of bike owners like you. Let's explore the options together to find the perfect fit for your Royal Enfield Interceptor 650.”, “ Of course! I'd be happy to provide a quote for comprehensive coverage and roadside assistance for your 2022 Royal Enfield Classic 350. Let's work together to find the best plan that fits your needs and budget. I'll send you the quote shortly.”  

3. Emotional appeal – Persuades by evoking emotions such as safety, comfort, fear, pride, hope, or peace of mind.  
   - Often uses reassuring or emotive language to create a personal emotional response rather than presenting objective facts.  
   - Example indicators: “ We're glad to hear you're interested in roadside assistance. Our partner, Reliance General, offers excellent roadside assistance services, including battery jump-starts, tire changes, and towing, to ensure you're never stranded on the road. Let's review the plans that include this valuable feature.”, “ Our comprehensive plan also includes roadside assistance specifically for electric vehicles. This means you'll have help whenever you need it, whether it's charging assistance, towing, or battery jump-starting. We want to make sure you're always on the move with confidence.”  

4. Personal appeal – Directly tailors the recommendation to the specific needs, concerns, or requests that the user has previously expressed.  
   - This strategy explicitly references what the user wants or has stated, making it a targeted recommendation.  
   - Example indicators: “ Wonderful! I'm delighted to help you secure comprehensive coverage for your 2022 Tesla Model 3. Let's proceed with setting up a policy that suits your needs and budget. I'm here to guide you through the process and answer any questions you may have along the way. Looking forward to working with you!”, “ I'm glad to hear that you find our motor insurance policy for your 2021 Yamaha MT-07 reassuring. We're here to help you protect your bike and ensure you're covered in case of an accident. Let's proceed with the policy details and finalize the coverage that best suits your needs and budget. If you have any questions or concerns, feel free to ask. We're here to help!”  

5. Persona appeal – Connects the recommendation to the user’s lifestyle, habits, values, or identity rather than a specific request.  
   - Often uses “if you are the kind of person who…” framing or aligns features with an implied personality type.  
   - Example indicators: “ Welcome! I'm delighted to help you with your motor insurance needs for your new 2024 Tesla Model 3. To ensure we provide the best coverage for your electric vehicle, I'd be happy to provide a personalized quote. Let's get started!”, “ I'm delighted to help you with your insurance needs for your new 2024 Tata Nexon EV. We have tailored coverage options specifically for electric vehicles, ensuring your investment is well-protected. Let's discuss the best plan for you. Shall we get started?”  

6. Default – Provides neutral, purely descriptive information without persuasive elements.  
   - No attempt to influence the decision is made; it is strictly informational.  
   - Example indicators: “ You're welcome! I'm glad you find the quote helpful. If you have any questions or need further assistance, feel free to ask. We're here to help you make an informed decision about your motor insurance. Have a great day!” , " You're welcome! I'm glad I could help you find an affordable and comprehensive motor insurance policy for your Yamaha R15. I've sent a personalized quote to your email. If you have any questions or need further assistance, feel free to reach out. Enjoy your weekend rides with peace of mind!" 



Output format:  

Label: <strategy name>  
<reason>Reasoning for why this strategy was chosen, based on utterance features.</reason>  


Example 1:  
Sentence: "This plan saves you $200 annually compared to your current policy while offering the same coverage."  
Label: Logical appeal  
<reason>The sentence presents a clear and measurable financial benefit ($200 savings) and directly compares it to the user’s current policy, while also providing an assurance that coverage remains unchanged. This approach relies entirely on verifiable, factual claims rather than emotional language or lifestyle framing. The persuasive force comes from the logical inference that if two options are equal in quality but one costs less, the cheaper option is preferable. This kind of reasoning appeals to rational cost–benefit analysis, which is characteristic of a logical appeal.</reason>  

Example 2:  
Sentence: "Our company has been rated #1 in customer satisfaction for five years in a row."  
Label: Credibility appeal  
<reason>The statement uses an external, presumably reliable ranking (“#1 in customer satisfaction”) sustained over a significant period (“five years in a row”) to establish trust and authority. This is not about the emotional comfort of the user or about personal needs; rather, it’s about presenting the company as a credible and trustworthy provider based on an objective performance measure. By showing consistent recognition over multiple years, it signals reliability and industry leadership, which are core traits of a credibility appeal.</reason>  

Example 3:  
Sentence: "With this policy, you can relax knowing your family is protected in any situation."  
Label: Emotional appeal  
<reason>This sentence primarily targets the reader’s emotions, specifically feelings of safety, comfort, and peace of mind. It does not rely on statistics, factual comparisons, or credentials; instead, it paints a reassuring scenario where the user can “relax” because their “family is protected.” The emotional core lies in evoking the relief and security one feels when loved ones are safeguarded. This is a direct attempt to influence the decision by appealing to the desire for emotional stability and personal well-being, making it a classic example of an emotional appeal.</reason>  

Example 4: 
Sentence: "If you enjoy weekend camping trips, this plan covers damage from off-road driving."
Label: Persona appeal
<reason>The sentence clearly ties the insurance plan’s benefits to a specific lifestyle and recreational activity—weekend camping trips combined with off-road driving. Rather than appealing to the user’s emotions or providing factual comparisons, it frames the plan as particularly suited to individuals whose hobbies and identity involve outdoor adventure. This creates a sense of personal relevance based on lifestyle alignment: if the listener identifies with this activity, the plan will feel naturally “for them.” Such targeting based on hobbies, identity, or habitual activities is the hallmark of a persona appeal.</reason>  

Example 5: 
Sentence: "Coverage applies to vehicles up to 10 years old, with claims processed within 7 business days."
Label: Default
<reason>This statement is purely informational, providing specific operational details about the policy—vehicle age limit and claim processing timeframe. It does not attempt to influence the decision through emotional triggers, credibility signals, or lifestyle connections. Nor does it present a comparative or cost–benefit argument; it simply conveys service terms in a factual manner. Because the language is neutral and the intent is to inform rather than persuade, it fits the definition of a default case, where content is descriptive without embedded persuasive strategy.</reason>  

Now, identify the persuasion strategy and give the reasoning inside <reason> tags as instructed.  

Sentence: "{sentence}"  
Label:  
<reason></reason>  
"""
  
    messages = [
        {"role": "system", "content": "you are a helpful assistant. Label the utterance based on Persuasion Strategy and provide the reason"},
        {"role": "user",  "content": prompt},
    ]

    output = pipe(messages, **generation_args)
    response= output[0]['generated_text']
    return response.strip()


# %%
print(generate("Imagine being stranded after an accident and knowing help is just one call away. Our cover ensures you and your loved ones are protected at all times, so you can focus on the journey, not the what-ifs"))

# %%
import pandas as pd
import os
import re

# Regex patterns to parse the extract() output
LABEL_RE = re.compile(r"Label\s*:\s*(.+?)(?:\s*(?:\n|$|<))", re.IGNORECASE | re.DOTALL)
REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.IGNORECASE | re.DOTALL)

def parse_extraction(text: str) ->tuple:
    """Extract label and reason from the returned string."""
    if not text:
        return "", ""
    m_label = LABEL_RE.search(text)
    m_reason = REASON_RE.search(text)
    label = m_label.group(1).strip() if m_label else ""
    reason = m_reason.group(1).strip() if m_reason else ""
    return label, reason

def process_csv_with_pandas(input_path: str, output_path: str, start_from: int = 0):
    """
    Reads input CSV, processes each row, and writes to output CSV incrementally.
    Requires an extract(sentence) or generate(sentence) function in scope.

    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        start_from: Row index (0-based) to start processing from
    """
    # Check if we are resuming or starting fresh
    if os.path.exists(output_path):
        processed_df = pd.read_csv(output_path)
        processed_count = len(processed_df)
    else:
        processed_df = pd.DataFrame()
        processed_count = 0

    # Read the whole input file
    df = pd.read_csv(input_path)
    if "new_agent_reply" not in df.columns:
        raise KeyError("Input CSV must contain a 'new_agent_reply' column.")

    # Determine actual starting point
    start_index = max(start_from, processed_count)

    # Process remaining rows one by one
    for idx in range(start_index, len(df)):
        sentence = str(df.loc[idx, "new_agent_reply"]).strip()
        result = generate(sentence)  # <-- your function here
        label, reason = parse_extraction(result)

        # Append label & reason to dataframe
        row = df.loc[[idx]].copy()
        row["label"] = label
        row["reason"] = reason

        # Append to the processed dataframe
        processed_df = pd.concat([processed_df, row], ignore_index=True)

        # Write back to CSV immediately so you can check progress
        processed_df.to_csv(output_path, index=False)
        os.sync()  # Force write to disk on some systems

    print(f"Processing complete. Output saved to {output_path}")



# %%

from datetime import datetime
start_time = datetime.now()
print("Started at--->", start_time.strftime('%Y-%m-%d %H:%M:%S'))
input_path="/home/rohank__iitp/Work/niladri/Expert Datatset/Original Dataset.csv"
output_path="/home/rohank__iitp/Work/niladri/Expert Datatset/Persuasion/Persuasion_results.csv"
process_csv_with_pandas(input_path, output_path, start_from=0)

end_time = datetime.now()
print("Finished time--->", end_time.strftime('%Y-%m-%d %H:%M:%S'))
print(f" completed in {end_time - start_time} seconds")