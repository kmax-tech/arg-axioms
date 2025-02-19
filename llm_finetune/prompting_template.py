general_llm_prompt_instructions = \
"""You are tasked with evaluating the relevance of a document to a given topic based on the provided annotation guideline. Carefully review the topic, document title, and document content, and follow the annotation guidelines to assign the appropriate relevance score.

Annotation Guideline:
{annotation_guideline}

Please provide your answer in the following format, replacing "RELEVANCE" with the chosen relevance value:

Relevance: RELEVANCE

Take your time to ensure your response aligns with the guideline. In the following are the information you need for this task:

Topic: {topic}
Document Title: {title}
Document: {document}
"""

with open("annotation_guideline_refactored.txt","r") as file:
    annotation_guideline = file.read()

def do_annotation(topic=None,title=None,document=None):
    return general_llm_prompt_instructions.format(topic=topic,title=title,document=document,annotation_guideline=annotation_guideline)


def create_prompt_list(document_list):
    final_list = []

    for document_dict in document_list:
        topic = document_dict["topic"]
        title = document_dict["title"]
        document = document_dict["document"]
        label_raw = document_dict["label"]

        prompt = do_annotation(topic=topic,title=title,document=document)
        label = "Relevance: " + reverse_relevance_scores[label_raw]

        data_sample = {
        "messages" : [{"role" : "user", "content" : prompt}, {"role" : "assistant", "content" : label}]}
        final_list.append(data_sample)
    return final_list






relevance_scores = {
    'R-2': 2,  # Highly relevant
    'R-1': 1,  # Relevant
    'R-0': 0,  # Not relevant, but an argument
    'R-X': -2   # Not an argument
}
reverse_relevance_scores = {v: k for k, v in relevance_scores.items()}