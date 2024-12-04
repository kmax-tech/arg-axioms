import settings as s
import llm_prompting_qrels.gpt_prompting as gpt
import llm_prompting_qrels.gemini_prompting as gemini
import llm_prompting_qrels.claude_prompting as claude
from collections import defaultdict

import re

general_llm_promt_instructions = \
"""You are given a topic, a document, the title of the document and guidelines for annotation.
Please follow the annotation guidelines and return the appropriate relevance and quality annotations.
Give your answer exactly according to the following pattern, where the words “RELEVANCE” and “QUALITY” indicate the values you have chosen for relevance and quality:

Relevance:RELEVANCE
Quality:QUALITY

In the following you are provided with the required information.

Topic: {topic}
Document Title: {title}
Document: {document}

Annotation Guidelines: 

{annotation_guidelines}


Read everything carefully and take your time before answering."""


def average_scores(annotation , label_scores,prefix='') :

    # Convert annotations to numerical scores
    scores = [label_scores[label] for label in annotation]

    # Manually calculate the average (sum divided by number of scores)
    total_score = sum(scores)
    avg_score = total_score / len(scores)

    # Assign final label based on average score (threshold-based or round)
    if avg_score >= 2.5 :
        return prefix + '2'
    elif avg_score >= 1.5 :
        return prefix +'1'
    elif avg_score >= 0.5 :
        return prefix + '0'
    else :
        return prefix +'x'




def majority_voting(annotations) :
    countdict = defaultdict(int)

    for sample in annotations :
        countdict[sample] += 1

    sorted_tuples_desc = sorted(countdict.items() , key=lambda item : item[1] , reverse=True)

    if len(sorted_tuples_desc) == 1:
        return sorted_tuples_desc[0][0]

    if sorted_tuples_desc[0][1] == sorted_tuples_desc[1][1] :
        return None

    return sorted_tuples_desc[0][0]


def convert_to_integer(text):
    if text == 'x':
        return -2
    try:
        # Attempt to convert the string to an integer
        return int(text)
    except ValueError:
        # If conversion fails, handle the error
        print(f"Cannot convert '{text}' to an integer.")
        return None

def find_relevance_and_quality_string(text):
    text = text.lower()
    relevance = None
    quality = None

    lines = text.split('\n')
    for line in lines:
        match = re.search(r'^relevance:\s*(r-.)\s*$', line)
        if match and (relevance is None):
            relevance = match.group(1).strip()
        match = re.search(r'^quality:\s*(q-.)\s*$', line)
        if match and (quality is None):
            quality = match.group(1).strip()
    return relevance, quality


relevance_scores = {
    'r-2': 3,  # Highly relevant
    'r-1': 2,  # Relevant
    'r-0': 1,  # Not relevant, but an argument
    'r-x': 0   # Not an argument
}

quality_scores = {
    'q-2': 3,  # high quality
    'q-1': 2,  # quality
    'q-0': 1,  # bad quality
    'q-x': 0   # should not exist
}

def clean_text(text,remove_part):
    if text is None:
        return None
    text_trimmed = text.replace(remove_part , '')
    text_converted = convert_to_integer(text_trimmed)
    return text_converted

def query_llm(data_dict):
    instructions_path = s.PROJECT_ROOT / 'llm_prompting_qrels' / 'prompting_instructions.txt'
    with open(instructions_path, 'r') as file:
        annotation_guidelines = file.read()
    data_dict['annotation_guidelines'] = annotation_guidelines
    instructions = general_llm_promt_instructions.format(**data_dict)
    qid = data_dict['qid']
    docno = data_dict['docno']

    relevance_claude= None
    quality_claude= None
    relevance_gpt= None
    quality_gpt= None
    relevance_gemini= None
    quality_gemini = None

    if s.CLAUDE in s.LLMS_TO_USE:
        print("Using Claude")
        answer_claude = claude.get_llm_response(instructions)
        relevance_claude, quality_claude = find_relevance_and_quality_string(answer_claude)
        if relevance_claude == None:
            relevance_claude = 'r-x'
        if quality_claude == None:
            quality_claude = 'q-0'

    if s.GPT in s.LLMS_TO_USE:
        print("Using GPT")
        answer_gpt = gpt.get_llm_response(instructions)
        relevance_gpt, quality_gpt = find_relevance_and_quality_string(answer_gpt)
        if relevance_gpt == None:
            relevance_gpt = 'r-x'
        if quality_gpt == None:
            quality_gpt = 'q-0'


    if s.GEMINI in s.LLMS_TO_USE:
        print("Using Gemini")
        answer_gemini = gemini.get_llm_response(instructions)
        relevance_gemini, quality_gemini = find_relevance_and_quality_string(answer_gemini)
        if relevance_gemini == None:
            relevance_gemini = 'r-x'
        if quality_gemini == None:
            quality_gemini = 'q-0'

    all_relevance = [relevance_claude, relevance_gpt, relevance_gemini]
    all_relevance = [r for r in all_relevance if r is not None]
    final_relevance = majority_voting(all_relevance)
    if final_relevance is None:
        final_relevance = average_scores(all_relevance , relevance_scores , prefix='r-')
    final_relevance= clean_text(final_relevance,'r-')

    all_quality = [quality_claude,quality_gpt,quality_gemini]
    all_quality = [q for q in all_quality if q is not None]
    final_quality = majority_voting(all_quality)
    if final_quality is None:
        final_quality = average_scores(all_quality , quality_scores , prefix='q-')
    final_quality = clean_text(final_quality,'q-')

    return {
            'relevance_llm' : final_relevance,
            'quality_llm': final_quality,
            'relevance_claude' : clean_text(relevance_claude,'r-' ),
            'quality_claude' : clean_text(quality_claude,'q-'),
            'relevance_gpt' : clean_text(relevance_gpt,'r-'),
            'quality_gpt' : clean_text(quality_gpt,'q-'),
            'relevance_gemini' : clean_text(relevance_gemini,'r-'),
            'quality_gemini' : clean_text(quality_gemini,'q-'),
            'qid' : qid,
            'docno' : docno
        }

if __name__ == '__main__':
    data_dict = {
        'title' : 'Cool Document',
        'topic' : 'I want to be a cool document',
        'document' : 'I want to be a cool document',
    }
    query_llm(data_dict)