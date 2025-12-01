import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk import ngrams
from collections import Counter
import math


#  Extract Executable File  #
def extract_exe(command_line: str):

    pattern = r'\\([^\\]+?\.(exe|bat|cmd|ps1|com|vbs))'
    match = re.search(pattern, str(command_line), re.IGNORECASE)
    return match.group(0) if match else ""


# General n-gram Entropy  #
def ngram_entropy(s, n=1):
    grams = [''.join(g) for g in ngrams(s, n)]
    freq = Counter(grams)
    length = len(grams)

    if length == 0:
        return 0

    entropy = -sum((count / length) * math.log2(count / (length*0.75)) for count in freq.values())

    num_unique = len(freq)
    if num_unique > 1:
        entropy /= math.log2(num_unique)

    return entropy


#Load CSV File
file_Name_log = input("Enter the CSV filename (with or without .csv):\n")
if '.csv' not in file_Name_log:
    file_Name_log += '.csv'
df = pd.read_csv(file_Name_log)

# Extract Executable Path
if 'Run Keys:Command Line' in df.columns:
    df['Executable Path'] = df['Run Keys:Command Line'].apply(extract_exe)
elif 'Run Once Keys:Command Line' in df.columns:
    df['Executable Path'] = df['Run Once Keys:Command Line'].apply(extract_exe)
else:
    raise ValueError("No valid 'Command Line' column found in CSV.")

#Compute Entropies
# Skip entropy if Executable Path is empty or NaN
df['Unigram Entropy'] = df['Executable Path'].apply(lambda x: ngram_entropy(x, 1) if pd.notnull(x) and len(x) > 0 else None)
df['Bigram Entropy']  = df['Executable Path'].apply(lambda x: ngram_entropy(x, 2) if pd.notnull(x) and len(x) > 0 else None)
df['Trigram Entropy'] = df['Executable Path'].apply(lambda x: ngram_entropy(x, 3) if pd.notnull(x) and len(x) > 0 else None)

# Only compute weighted entropy if all components exist
df['Weighted Entropy'] = df.apply(
    lambda row: (row['Unigram Entropy'] * 0.2 +
                 row['Bigram Entropy'] * 0.3 +
                 row['Trigram Entropy'] * 0.5)
    if pd.notnull(row['Unigram Entropy']) and pd.notnull(row['Bigram Entropy']) and pd.notnull(row['Trigram Entropy'])
    else None,
    axis=1
)


# Top & Bottom Sorting
df_sorted_big = df.sort_values(by='Weighted Entropy', ascending=False)
df_sorted_small = df.sort_values(by='Weighted Entropy', ascending=True)

top_10 = df_sorted_big.head(10)
bottom_10 = df_sorted_small.head(10)

nameRunKeys = 'Run Keys:Name' if 'Run Keys:Name' in df.columns else 'Run Once Keys:Name'

# Output Top & Bottom
print("\nTop 10 Entries Based on Entropy:")
print(top_10[[nameRunKeys, 'Executable Path', 'Weighted Entropy']])

print("\nBottom 10 Entries Based on Entropy:")
print(bottom_10[[nameRunKeys, 'Executable Path', 'Weighted Entropy']])

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_10[nameRunKeys], top_10['Weighted Entropy'], color='green')
plt.xlabel('Weighted Entropy')
plt.ylabel('Executable Name')
plt.title('Top 10 Entries by Entropy')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.barh(bottom_10[nameRunKeys], bottom_10['Weighted Entropy'], color='red')
plt.xlabel('Weighted Entropy')
plt.ylabel('Executable Name')
plt.title('Bottom 10 Entries by Entropy')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


'''
vertical graph

y      by       x
count by  0.80-.81, 0.9-.91 etc.
df.["entropy-score"].hist() the plot and then i can adjust accordingly
plot title, make it pretty, look fancy
no executable names
'''