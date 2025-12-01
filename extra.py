'''
Python Entropy of FileNames in a CSV

to look through CSV, extract the filepaths, calculate the entropy score
using Shannon entropy, display top 10 and bottom 10

change filename on line 56 if necessary

this works for and for run once
Run Keys:Type,Run Keys:Name,Run Keys:Command Line, Count
with 5 lines csv


but has a issue with bigfile.csv with 100 lines as it is not calcuating properly


Author: Taemoor Hasan
'''

#Search up standard math library for entropy python

import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import words


# Shannon entropy function
import math
from collections import Counter


#entropy of bigrams
#try other algoritms


from nltk import ngrams


def ngram_entropy(s, n=2):
    grams = [''.join(g) for g in ngrams(s, n)]
    freq = Counter(grams)
    length = len(grams)
    if length == 0:
        return 0  # Avoid division by zero for short strings
    entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())

    num_unique = len(freq)
    if num_unique > 1:
        entropy /= math.log2(num_unique)

    return entropy


#def shannon_entropy(s :str)
def shannon_entropy(s):

    freq = Counter(s)
    length = len(s)

    #length might be playing a huge factor in this then it supposed to
    #math.log on length?
    entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())

    # avoid math domain error for one unique character
    num_unique = len(freq)
    if num_unique > 1:
        entropy /= math.log2(num_unique)

    return entropy


# Function to extract exe name or  file path
def extract_exe(command_line: str):

    # Use regex to find the .exe file in the command line
    #matehches everything after the last slash that ends with .exe
    #+).exe
    #use the dollar sign at the end
    #(),exe,pipe,ps1,cmd, extensions
    #learn regex 101

    #NOTE: some programs do not end with .exe

    # temp is a big file to search
    match = re.search(r'\\([^\\]+\.exe)', str(command_line))
    if match:
        return match.group(0)
    return ""



#change file name if necessary
#df = pd.read_csv('logs.csv')


#User input for filename (if needed)

file_Name_log=input("What is the filename?\n")

if '.csv' in file_Name_log:
    df = pd.read_csv(file_Name_log)
else:
    df = pd.read_csv(file_Name_log+'.csv')



#to extract the .exe filename or path
if 'Run Keys:Command Line' in df.columns:
    df['Executable Path'] = df['Run Keys:Command Line'].apply(extract_exe)
elif 'Run Once Keys:Command Line' in df.columns:
    df['Executable Path'] = df['Run Once Keys:Command Line'].apply(extract_exe)

# calculate entropy extracted executable path
# Step 1: Calculate individual entropy scores (if not already done)
df['Unigram Entropy'] = df['Executable Path'].apply(lambda x: shannon_entropy(x))
df['Bigram Entropy'] = df['Executable Path'].apply(lambda x: ngram_entropy(x, 2))
df['Trigram Entropy'] = df['Executable Path'].apply(lambda x: ngram_entropy(x, 3))

# Step 2: Now calculate the weighted entropy
df['Weighted Entropy'] = (df['Unigram Entropy'] * 0.2 +
                          df['Bigram Entropy'] * 0.3 +
                          df['Trigram Entropy'] * 0.5)


df['Avg Entropy'] = df[['Unigram Entropy', 'Bigram Entropy', 'Trigram Entropy']].mean(axis=1)



df_sorted_big = df.sort_values(by='Weighted Entropy', ascending=False)
df_sorted_small = df.sort_values(by='Weighted Entropy', ascending=True)

#top 10 entries based on entropy (highest entropy)
top_10 = df_sorted_big.head(10)

# bottom 10 entries based on entropy (lowest entropy)
bottom_10 = df_sorted_small.head(10)



nameRunKeys= 'Run Keys:Name' if 'Run Keys:Name' in df.columns else 'Run Once Keys:Name'

# Show the top 10 entries with the name, path, and entropy
print("Top 10 Entries Based on Entropy:")
print(top_10[[nameRunKeys, 'Executable Path', 'Entropy']])


print("\nBottom 10 Entries Based on Entropy:")
print(bottom_10[[nameRunKeys, 'Executable Path', 'Weighted Entropy']])



# Plotting the top 10 entries
plt.figure(figsize=(10, 6))
plt.barh(top_10[nameRunKeys], top_10['Weighted Entropy'], color='green')

plt.xlabel('Entropy')
plt.ylabel('Executable Name')
plt.title('Top 10 Entries Based on Entropy')
plt.gca().invert_yaxis()  # Invert the y-axis to display the highest entropy at the top
plt.show()

# Plotting the bottom 10 entries
plt.figure(figsize=(10, 6))
plt.barh(bottom_10[nameRunKeys], bottom_10['Weighted Entropy'], color='red')

plt.xlabel('Entropy')
plt.ylabel('Executable Name')
plt.title('Bottom 10 Entries Based on Entropy')
plt.gca().invert_yaxis()  # Invert the y-axis to display the highest entropy at the top
plt.show()



print("\n Manual Entropy Debug:")

testNames = [
    "quickInstaller",
    "CustomExecutable",
    "updateConfig",
    "SampleExecutable",
    "jdssjffhgukh",
    "aaaaaaaaaaaa",
    "abcabcabcabc",
    "x9v7tz8wnm"
]

for name in testNames:
    print(f"{name} entropy: {shannon_entropy(name)}")


'''
Chatgpt says entorpy measures chaarcter diversity 
not our randomness of letters

jdssjffhugukh has reptiitons of s,f,h
but a quickinstaller has 14 unique letters
the shannon entrooy equation would be more doffernt 
log2(15)

we need to test it on the actual data to see resulst


bigrams
trigrams
unigram entopry filename 

use same algorithms and use weigh the bigrams.trigrams,unigrams

next step:


'''