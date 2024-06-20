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
   

## Hadoop SVM Implementation

### Description
This project implements a Support Vector Machine (SVM) for data classification using the Hadoop framework. SVM is a supervised learning algorithm used for binary classification tasks. The project leverages the distributed computing capabilities of Hadoop to handle large-scale datasets efficiently.

### Files
- **SVM.java**: Main Java program implementing the SVM using Hadoop.
- **SVM.jar**: Compiled JAR file containing the SVM program.
- **Input_AIDS_Classification_5000.txt**: Input data file used for SVM training and classification.
- **Output_AIDS_Classification_1000.txt**: Output file containing the classified results after SVM execution.
- **Hadoop_SVM_Report.pdf**: Detailed report providing insights into the SVM implementation, methodology, and results.
