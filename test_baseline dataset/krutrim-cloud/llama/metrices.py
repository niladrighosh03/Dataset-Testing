# %%
print("hi")

# %%
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


import math
import numpy as np
def calculate_perplexity(text: str) -> float | None:
    model_name = "Llama-3.3-70B-Instruct"
    client = KrutrimCloud(api_key=api_key)

    print(f"Calculating perplexity for model '{model_name}'...")

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": text}],
        max_tokens=1,
        logprobs=True,
        temperature=0.0,
    )

    logprobs_dict = response.choices[0].logprobs

    if not logprobs_dict or 'content' not in logprobs_dict:
        print("Error: Log probabilities content was not returned in the API response.")
        return None

    logprobs_content = logprobs_dict['content']

    log_probs = [token_info['logprob'] for token_info in logprobs_content]

    if log_probs and log_probs[0] == 0.0:
        log_probs = log_probs[1:]

    if not log_probs:
        print("Error: No valid log probabilities found to calculate perplexity.")
        return None

    mean_log_prob = np.mean(log_probs)
    perplexity = np.exp(-mean_log_prob)

    return perplexity


# %%

# %%
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu2(candidate_sentence: str, reference_sentence: str) -> float:

    # Tokenize
    candidate_tokens = candidate_sentence.strip().split()
    reference_tokens = reference_sentence.strip().split()
    
    # BLEU-2 with weights (0.5, 0.5) → unigram and bigram equally
    weights = (0.5, 0.5)
    
    # Optional smoothing to avoid zero scores for short or imperfect matches
    smoothing = SmoothingFunction().method1

    # Compute score
    bleu2_score = sentence_bleu([reference_tokens], candidate_tokens, weights=weights, smoothing_function=smoothing)
    
    return bleu2_score




# %%
from bert_score import score

def compute_bert_score_f1(candidates, references, lang="en", model_type="bert-base-uncased", verbose=False):

    P, R, F1 = score(candidates, references, lang=lang, model_type=model_type, verbose=verbose)
    x=F1.tolist()
    return float(x[0])   # Convert torch.Tensor to list for usability



# %%
from collections import Counter

def distinct_2(text):

    words = text.strip().split()
    if len(words) < 2:
        return 0.0  # No bigrams can be formed

    bigrams = [(words[i], words[i+1]) for i in range(len(words) - 1)]
    total_bigrams = len(bigrams)
    unique_bigrams = len(set(bigrams))

    return unique_bigrams / total_bigrams


# %%
from rouge_score import rouge_scorer

def calculate_rouge1(reference, candidate):

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rouge1'].fmeasure  # returns a namedtuple with precision, recall, fmeasure



# %%
import pandas as pd

# Load your dataset
df = pd.read_csv("/home/rohank__iitp/Work/niladri/test_baseline dataset/krutrim-cloud/llama/llama_single_dataset.csv")

# Initialize empty columns
df['PPL'] = 0.0
df['BLEU-2'] = 0.0
df['BERTScore-F1'] = 0.0
df['Distinct-2'] = 0.0
df['ROUGE-1'] = 0.0


from datetime import datetime
start_time = datetime.now()
print("Started at--->", start_time.strftime('%Y-%m-%d %H:%M:%S'))

# Process each row and update CSV after each change
for i in range(len(df)):
    agent_reply = str(df.loc[i, 'new_agent_reply'])
    model_reply = str(df.loc[i, 'llama Single Response'])

    # Compute metrics and round to 3 decimal places
    df.loc[i, 'PPL'] = round(calculate_perplexity(model_reply), 3)
    df.loc[i, 'BLEU-2'] = round(compute_bleu2(model_reply, agent_reply), 3)
    df.loc[i, 'BERTScore-F1'] = round(compute_bert_score_f1([model_reply], [agent_reply]), 3)
    df.loc[i, 'Distinct-2'] = round(distinct_2(model_reply), 3)
    df.loc[i, 'ROUGE-1'] = round(calculate_rouge1(agent_reply, model_reply), 3)

    # Write to CSV after every row
    df.to_csv("/home/rohank__iitp/Work/niladri/test_baseline dataset/krutrim-cloud/llama/Comparision_llama_api.csv", index=False)


end_time = datetime.now()
print("Finished time--->", end_time.strftime('%Y-%m-%d %H:%M:%S'))
print(f" completed in {end_time - start_time} seconds")

