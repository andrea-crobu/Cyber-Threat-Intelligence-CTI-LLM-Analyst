# Cyber Threat Intelligence (CTI) Analysis Project

This project focuses on building an AI model capable of performing **Cyber Threat Intelligence (CTI) analysis**. The model is trained to analyze threat reports, extract relevant entities and their relationships, and generate a comprehensive diagnosis of the threat.

The project is divided into three main phases: **Data Preparation**, **Model Training**, and **Evaluation**.


## 📊 **Dataset**

- **Source**: [Hugging Face - swaption2009/cyber-threat-intelligence-custom-data](https://huggingface.co/datasets/swaption2009/cyber-threat-intelligence-custom-data)  
- **Content**: Threat reports containing details about malware, threat actors, and their relationships.  
- **Data Split**:
  - **80%** for Training  
  - **10%** for Validation  
  - **10%** for Testing  


## ⚙️ **Phase 1: Data Preparation**

1. **Custom Prompt Design**  
   Each threat report is formatted into a prompt for the model as follows:

   **Input Prompt**:
   ```
   You are a skilled AI Agent capable of doing CTI Analysis.

   Given this threat report:

   {threat report}

   You will extract the main entities and their relations; finally, you will generate a diagnosis of the threat.
   ```

   **Expected Output**:
   ```
   Entities: {list of entities}  
   Relations: {list of relations}  
   Diagnosis: {diagnosis}
   ```
   During training, the expected output is added to the input prompt, to let the model learn the proper sequence of tokens.

2. **Masking Strategy**  
   - Since a **decoder-only model** is used, the input prompt is integrated into the expected output but is **masked** using the special token `-100`.  
   - This ensures that the loss function only focuses on predicting the relevant output and not the prompt.

3. **Tokenization and Formatting**  
   - The data is tokenized and formatted into a Hugging Face dataset, ready for training.


## 🔧 **Phase 2: Model Training**

- **Base Model**: `"meta-llama/Llama-3.2-3B-Instruct"`  
- **Fine-Tuning Technique**: **LoRA (Low-Rank Adaptation)** for efficient training with reduced computational costs.

### Training Highlights
- The LoRA approach updates only a subset of the model's parameters, allowing for efficient and effective fine-tuning.  
- A **data collector** is used to handle variable-length inputs and dynamic masking, ensuring efficient batch processing.  
- The fine-tuned model learns to extract entities, establish relations, and generate diagnoses accurately from CTI reports.


## ✅ **Phase 3: Model Evaluation**

- Both the **base model** and **fine-tuned model** were evaluated on the test set using **ROUGE** metrics.

### **Performance Comparison**

| Metric    | Base Model | Fine-Tuned Model |
|-----------|------------|------------------|
| **ROUGE-1**   | 0.2895     | **0.6537**         |
| **ROUGE-2**   | 0.1145     | **0.4022**         |
| **ROUGE-L**   | 0.1946     | **0.5090**         |
| **ROUGE-Lsum**| 0.2620     | **0.5719**         |

- The **fine-tuned model** shows significant improvements across all metrics, confirming its enhanced understanding and performance in CTI analysis.

### **Example Prediction**

**Input Prompt**:
```
You are a skilled AI Agent capable of doing CTI Analysis.

Given this threat report:
Three of the backdoors, NFlog, PoisonIvy, and NewCT have previously been publicly associated with DragonOK.

You will extract the main entities and their relations; finally, you will generate a diagnosis of the threat.
```

**Ground Truth**:
```
Entities: NFlog (malware), PoisonIvy (malware), NewCT (malware), DragonOK (threat-actor)  
Relations: NFlog to DragonOK (authored-by), PoisonIvy to DragonOK (authored-by), NewCT to DragonOK (authored-by)  
Diagnosis: Possible diagnosis: The entity DragonOK has been associated with the creation of three malware backdoors, NFlog, PoisonIvy, and NewCT, as they were previously publicly linked to this threat actor. This suggests that DragonOK may be involved in...
```

**Fine-Tuned Model Prediction**:
```
Entities: DragonOK (threat-actor), NFlog (malware), PoisonIvy (malware), NewCT (malware)  
Relations: DragonOK to NFlog (uses), DragonOK to PoisonIvy (uses), DragonOK to NewCT (uses)  
Diagnosis: The threat actor DragonOK has previously been associated with the use of three malware backdoors: NFlog, PoisonIvy, and NewCT. This indicates a potential cybersecurity breach and highlights the importance of monitoring and securing networks against such threats.
```


## 🚀 **Conclusion**

- The fine-tuned model significantly outperforms the base model in extracting entities, understanding relationships, and providing comprehensive threat diagnoses.  
- This project showcases an efficient approach to fine-tuning large language models for domain-specific tasks using LoRA.  
- The final model is well-suited for assisting in CTI analysis by generating accurate and detailed threat assessments.


## 📚 **Future Work**

- Expand the dataset with more diverse threat reports for broader generalization.  
- Experiment with other fine-tuning techniques or model architectures.  
- Integrate the model into a real-time CTI analysis pipeline.


## 💡 **Key Technologies Used**

- **Python**  
- **Hugging Face Transformers**  
- **LoRA Fine-Tuning**  
- **ROUGE Evaluation Metrics**  
