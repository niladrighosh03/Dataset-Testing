# %%
import re
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

print("Loading model and tokenizer...")
MODEL_NAME = "/scratch/rohank__iitp/Phi-3-medium-128k-instruct"
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")

tok.padding_side = "right"
if model.config.pad_token_id is None:
    model.config.pad_token_id = tok.eos_token_id


# %%
INTENTS = {
    "Request_Insurance_Quote",
    "Ask_Coverage_Details",
    "Express_Concern",
    "Request_Additional_Info",
    "Confirm_Interest",
    "Ask_Price_or_Premium",
}

PRICE   = ["price","cost","how much","premium","monthly","annual","rate","fees","discount","emi"]
REQUEST = ["get insurance","get a quote","need insurance","need a quote","buy insurance","looking to get","want insurance","start policy","sign up","apply","new policy","renew my policy","purchase a policy","enroll","quote"]
COVERAGE= ["coverage","cover","covered","benefit","protection","exclusion","deductible","battery","accident","theft","fire","own damage","third party","comprehensive","zero dep","zero depreciation","roadside","idv","engine","flood","natural calamity","personal accident"]
CONCERN = ["concern","worried","worry","priority","afraid","risk","must have","need to make sure","i want to ensure","make certain","ensure","make sure"]
CONFIRM = ["sounds good","go ahead","proceed","let's do it","i agree","i’d like to proceed","i would like to proceed","i want to proceed","i'm interested","sign me up","yes please","continue","confirm"]


# %%
def lc(s): return re.sub(r"\s+", " ", (s or "")).strip().lower()
def contains_any(s, kws): return any(k in lc(s) for k in kws)
def first_snippet(s, kws):
    t = s or ""; tl = t.lower()
    for k in kws:
        i = tl.find(k)
        if i >= 0:
            return t[max(0, i-12): min(len(t), i+len(k)+18)].strip()
    return ""

# %%
FEW_SHOTS = """
User: Do you cover battery fires and thermal incidents for EVs?
<intent>Ask_Coverage_Details</intent>
<reason>From the phrasing, this is a targeted query about protections rather than process or price. It specifically names protection types (“battery fires”, “thermal incidents”), which are hallmarks of a coverage inquiry, as evidenced by “cover battery fires and thermal incidents”.</reason>

User: I’d like a quote for motor insurance on my 2024 Honda Civic.
<intent>Request_Insurance_Quote</intent>
<reason>This message initiates the purchase flow by explicitly asking to obtain a policy quote. The intent is to start insurance for a specific vehicle and year, as evidenced by “quote for motor insurance on my 2024 Honda Civic”.</reason>

User: How much would the monthly premium be for a standard plan?
<intent>Ask_Price_or_Premium</intent>
<reason>The focus is on cost rather than features or process. It requests a price figure at a monthly cadence, which clearly frames a premium inquiry, as evidenced by “How much would the monthly premium be”.</reason>
""".strip()

PROMPT_TMPL = f"""You are an intent classifier for motor-insurance conversations.

Label exactly ONE of these intents:
1) Request_Insurance_Quote — user initiates interest in getting a quote/policy.
2) Ask_Coverage_Details — user asks what is covered (battery, accidents, theft, exclusions, deductibles).
3) Express_Concern — user states a specific concern/priority.
4) Request_Additional_Info — user asks for clarification/process/docs/eligibility/timelines.
5) Confirm_Interest — user agrees/approves/wants to proceed.
6) Ask_Price_or_Premium — user asks about cost/premium/discounts/price breakdown.

Guidelines:
- Use ONLY the user message (ignore any agent text).
- Choose exactly ONE label from the list above.
- Provide a high-quality explanation inside <reason>...</reason> (1–2 sentences). It should be specific, reference category-defining cues, and end with “as evidenced by …” + a short quote/snippet.
- Vary openings; do not always begin with “The user…”.
- If the message is empty/unclear, default to Request_Additional_Info and note the lack of content.

Few-shot examples:
{FEW_SHOTS}

Now classify and output ONLY:
<intent>OneOf: Request_Insurance_Quote | Ask_Coverage_Details | Express_Concern | Request_Additional_Info | Confirm_Interest | Ask_Price_or_Premium</intent>
<reason>1–2 precise sentences ending with “as evidenced by …” + a short quote.</reason>

User: {{user_utt}}
"""

# %%
@torch.no_grad()
def predict_intent_reason(user_utt: str, max_new_tokens=180):
    messages = [
        {"role": "system", "content": "You are a careful, concise intent classifier."},
        {"role": "user", "content": PROMPT_TMPL.format(user_utt=user_utt or "")},
    ]
    chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,   # deterministic
        do_sample=False,
        pad_token_id=model.config.pad_token_id,
    )
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    im = re.search(r"<intent>\s*([^<]+?)\s*</intent>", gen, flags=re.I|re.S)
    rm = re.search(r"<reason>\s*([^<]+?)\s*</reason>", gen, flags=re.I|re.S)
    intent = (im.group(1).strip() if im else "")
    reason = (rm.group(1).strip() if rm else "")

    if intent not in INTENTS:
        intent = "Request_Additional_Info"
    if reason and not reason.endswith("."):
        reason += "."
    return intent, reason

# %%
def correct_intent_and_reason(intent, reason, user_utt):
    user = user_utt or ""

    if intent not in INTENTS or not intent:
        intent = "Request_Additional_Info"

    if intent == "Request_Additional_Info":
        if contains_any(user, COVERAGE):
            intent = "Ask_Coverage_Details"
            if not reason:
                snip = first_snippet(user, COVERAGE) or "coverage details"
                reason = f"This reads as a request for specific protections rather than process or cost, as evidenced by “{snip}”."
        elif contains_any(user, PRICE):
            intent = "Ask_Price_or_Premium"
            if not reason:
                snip = first_snippet(user, PRICE) or "pricing"
                reason = f"The focus is on cost and premium amount instead of features, as evidenced by “{snip}”."
        elif contains_any(user, REQUEST):
            intent = "Request_Insurance_Quote"
            if not reason:
                snip = first_snippet(user, REQUEST) or "get a quote"
                reason = f"The message initiates obtaining a policy/quote for a vehicle, as evidenced by “{snip}”."
        elif contains_any(user, CONFIRM):
            intent = "Confirm_Interest"
            if not reason:
                snip = first_snippet(user, CONFIRM) or "proceed"
                reason = f"It expresses agreement to move ahead rather than ask questions, as evidenced by “{snip}”."
        elif contains_any(user, CONCERN):
            intent = "Express_Concern"
            if not reason:
                snip = first_snippet(user, CONCERN) or "concern"
                reason = f"It highlights a priority or worry about protection, as evidenced by “{snip}”."

    if reason and "as evidenced by" not in reason.lower():
        snip = first_snippet(user, COVERAGE+PRICE+REQUEST+CONCERN+CONFIRM)
        reason = reason.rstrip(".") + f', as evidenced by “{snip}”.'
    if reason and not reason.endswith("."):
        reason += "."
    return intent, reason

# %%
INPUT_CSV  = "/home/rohank__iitp/Work/niladri/Expert Datatset/Original Dataset.csv"  
OUTPUT_CSV = "/home/rohank__iitp/Work/niladri/Expert Datatset/Persuasion/Intent_results.csv"

df = pd.read_csv(INPUT_CSV)
if "user_utterance" not in df.columns:
    df["user_utterance"] = ""
df["intent"] = ""
df["reason"] = ""

if os.path.exists(OUTPUT_CSV):
    done_df = pd.read_csv(OUTPUT_CSV)
    processed_ids = set(done_df.index.tolist())
else:
    processed_ids = set()
    done_df = df.copy()

##TIme Adding

from datetime import datetime
start_time = datetime.now()
print("Started at--->", start_time.strftime('%Y-%m-%d %H:%M:%S'))


for idx, row in df.iterrows():
    if idx in processed_ids:
        continue  

    intent, reason = predict_intent_reason(row["user_utterance"])
    intent, reason = correct_intent_and_reason(intent, reason, row["user_utterance"])

    df.at[idx, "intent"] = intent
    df.at[idx, "reason"] = reason
    done_df.loc[idx] = df.loc[idx]
    done_df.to_csv(OUTPUT_CSV, index=False)
    
    
    
end_time = datetime.now()
print("Finished time--->", end_time.strftime('%Y-%m-%d %H:%M:%S'))
print(f" completed in {end_time - start_time} seconds")
print("Processing complete. Results saved incrementally to:", OUTPUT_CSV)

# %%



