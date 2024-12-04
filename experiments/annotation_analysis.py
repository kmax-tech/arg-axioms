import pickle

a = 'correlation_data_qrels_llms_21_improved_prompt.pkl'
b = 'correlation_data_qrels_llms_21_improved_prompt-v1.pkl'

def convert_data_frame(evaluated_df):
    evaluated_df = evaluated_df.fillna(-2)
    evaluated_df['label'] = evaluated_df['label'].astype(float)
    evaluated_df['quality'] = evaluated_df['quality'].astype(float)
    evaluated_df['relevance_llm'] = evaluated_df['relevance_llm'].astype(float)
    evaluated_df['quality_llm'] = evaluated_df['quality_llm'].astype(float)
    evaluated_df['relevance_claude'] = evaluated_df['relevance_claude'].astype(float)
    evaluated_df['relevance_gpt'] = evaluated_df['relevance_gpt'].astype(float)
    evaluated_df['relevance_gemini'] = evaluated_df['relevance_gemini'].astype(float)
    return evaluated_df

with open(a, 'rb') as file:
    data_1 = pickle.load(file)
    data_1 = convert_data_frame(data_1)

with open(b, 'rb') as file:
    data_2 = pickle.load(file)
    data_2 = convert_data_frame(data_2)

assert data_1[['label']].equals(data_2[['label']])
mewo =1



def calculate_corrlation(evaluated_df):
    cut_off = len(evaluated_df['relevance_llm'])
    # Calculate correlation coefficient
    correlation = (evaluated_df['label'][:cut_off]).corr(evaluated_df['relevance_llm'][:cut_off])
    print("Correlation coefficient:", correlation)
    correlation = evaluated_df['quality'][:cut_off].corr(evaluated_df['quality_llm'][:cut_off])
    print(f"Correlation for quality: {correlation}")
    correlation = evaluated_df['label'][:cut_off].corr(evaluated_df['relevance_claude'][:cut_off])
    print(f"Correlation for relevance claude: {correlation}")
    correlation = evaluated_df['label'][:cut_off].corr(evaluated_df['relevance_gpt'][:cut_off])
    print(f"Correlation for relevance gpt: {correlation}")
    correlation = evaluated_df['label'][:cut_off].corr(evaluated_df['relevance_gemini'][:cut_off])
    print(f"Correlation for relevance gemini: {correlation}")

calculate_corrlation(data_1)
calculate_corrlation(data_2)