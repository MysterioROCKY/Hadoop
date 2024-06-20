# Hadoop SVM and Common Words Projects

This repository contains two main projects: a Common Words counting program and an SVM (Support Vector Machine) implementation using Hadoop. Below are the detailed instructions on how to use and run these projects.

## Common Words Project

### Description
The Common Words project is designed to identify and count common words in given text files, excluding specified stop words.

### Files
- `CommonWords.java`: The main Java program that performs the common words counting.
- `stopwords.txt`: A text file containing the stop words to be excluded from the counting.
- `task1-input1.txt`: Sample input text file 1.
- `task1-input2.txt`: Sample input text file 2.

### Instructions

1. **Compile the Java Program**

   ```sh
   javac CommonWords.java

2. **Run the Program**

   ```sh
   java CommonWords stopwords.txt task1-input1.txt task1-input2.txt
   
