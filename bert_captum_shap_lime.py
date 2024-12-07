!pip install captum
!pip install transformers
!pip install shap transformers
!pip install lime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertForQuestionAnswering, BertConfig
import nltk
import shap
import torch.nn as nn
from lime.lime_text import LimeTextExplainer
from captum.attr import IntegratedGradients, InterpretableEmbeddingBase, TokenReferenceBase, visualization, configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from captum.attr import LayerConductance, LayerIntegratedGradients
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

nltk.download('reuters')
from nltk.corpus import reuters

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Reuters dataset
document_ids = reuters.fileids()
documents = [reuters.raw(doc_id) for doc_id in document_ids]
labels = [reuters.categories(doc_id)[0] for doc_id in document_ids]
label2idx = {label: idx for idx, label in enumerate(set(labels))}
encoded_labels = [label2idx[label] for label in labels]

# Define the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label2idx))

# Tokenize the input documents
tokenized_inputs = tokenizer(documents, padding=True, truncation=True, return_tensors='pt')

# Prepare the dataset
input_ids = tokenized_inputs['input_ids']
attention_mask = tokenized_inputs['attention_mask']
labels_tensor = torch.tensor(encoded_labels)

# Split the dataset into training and testing
train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = train_test_split(input_ids, attention_mask, labels_tensor, test_size=0.2, random_state=42)

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Set the model in training mode
model.train()
model.to(device)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(2):  # Run for 2 epochs
    epoch_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

        # Calculate accuracy
        predicted_labels = outputs.logits.argmax(dim=1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    average_loss = epoch_loss / len(train_loader)
    
    print(f"Epoch {epoch + 1}/{2} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.4f}")

# Save the trained model
model.save_pretrained("./saved_model")

# Set the model in evaluation mode
# Set the device (CPU or GPU)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained("./saved_model")
# Set the model in evaluation mode
model.eval()
model.to(device)

# Assuming test_loader is your DataLoader object for test data
test_predictions = []
test_labels_list = []
for batch in test_loader:
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label_idx = torch.argmax(probabilities, dim=1)
        test_predictions.extend(predicted_label_idx.tolist())
        test_labels_list.extend(labels.tolist())

# Calculate Accuracy
accuracy = accuracy_score(test_labels_list, test_predictions)
print(f"Accuracy: {accuracy}")

# Calculate Precision, Recall, F1-Score
precision, recall, f1, _ = precision_recall_fscore_support(test_labels_list, test_predictions, average='weighted')
print(f"Precision: {precision}\nRecall: {recall}\nF1-Score: {f1}")

unique_labels = np.unique(np.hstack([test_labels_list, test_predictions]))
idx2label = {idx: label for label, idx in label2idx.items()}
target_names = [idx2label[i] for i in unique_labels if i in idx2label]

# Print the classification report
print(classification_report(test_labels_list, test_predictions, target_names=target_names))

for label, idx in label2idx.items():
    print(f"Label: {label}, Index: {idx}")

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained("./saved_model")
# Set the model in evaluation mode
model.eval()
model.to(device)

# Define a new sentence
sentence = "Global stock markets have witnessed a substantial surge amidst renewed investor confidence"

# Tokenize the sentence
inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

# Get the inputs ready for the model
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

# Forward pass: get the outputs of the model for these inputs
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)

# The outputs are logits: apply the softmax function to get probabilities
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get the label with the highest probability
predicted_label_idx = probabilities.argmax().item()

# Decode the predicted label
predicted_label = list(label2idx.keys())[list(label2idx.values()).index(predicted_label_idx)]

print(f"The predicted label is: {predicted_label} and its index is: {predicted_label_idx}")

"""### CAPTUM:
CAPTUM
"""

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained("./saved_model")

# We need to split forward pass into two part: 
# 1) embeddings computation
# 2) classification

def compute_bert_outputs(model_bert, embedding_output, attention_mask=None, head_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    extended_attention_mask = extended_attention_mask.to(dtype=next(model_bert.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(dtype=next(model_bert.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * model_bert.config.num_hidden_layers

    encoder_outputs = model_bert.encoder(embedding_output,
                                         extended_attention_mask,
                                         head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = model_bert.pooler(sequence_output)
    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)    


class BertModelWrapper(nn.Module):
    
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, embeddings):        
        outputs = compute_bert_outputs(self.model.bert, embeddings)
        pooled_output = outputs[1]
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return torch.softmax(logits, dim=1)  # Return probabilities for all classes

    
bert_model_wrapper = BertModelWrapper(model)
ig = IntegratedGradients(bert_model_wrapper)

# accumalate couple samples in this array for visualization purposes
vis_data_records_ig = []

def interpret_sentence(model_wrapper, sentence, target_label_idx):

    model_wrapper.eval()
    model_wrapper.zero_grad()
    
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])
    input_embedding = model_wrapper.model.bert.embeddings(input_ids)
    
    # predict
    preds = model_wrapper(input_embedding)
    pred_ind = preds.argmax().item()  # Get the index of the highest probability

    # compute attributions and approximation delta using integrated gradients
    attributions_ig, delta = ig.attribute(input_embedding, target=target_label_idx, n_steps=500, return_convergence_delta=True)

    print('pred: ', pred_ind, '(', '%.2f' % preds[0, pred_ind].item(), ')', ', delta: ', abs(delta))

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].numpy().tolist())    
    add_attributions_to_visualizer(attributions_ig, tokens, preds[0, pred_ind].item(), pred_ind, target_label_idx, delta, vis_data_records_ig)
    
    
def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().numpy()
    
    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            pred_ind,
                            label,
                            "label",
                            attributions.sum(),       
                            tokens,
                            delta))

interpret_sentence(bert_model_wrapper, sentence="Global stock markets have witnessed a substantial surge amidst renewed investor confidence", target_label_idx=9)
visualization.visualize_text(vis_data_records_ig)

interpret_sentence(bert_model_wrapper, sentence="Global stock markets have witnessed a substantial surge amidst renewed investor confidence", target_label_idx=10)
visualization.visualize_text(vis_data_records_ig)

interpret_sentence(bert_model_wrapper, sentence="Global stock markets have witnessed a substantial surge amidst renewed investor confidence", target_label_idx=63)
visualization.visualize_text(vis_data_records_ig)

sentences = [
    "Oil giant Exxon Mobil sees record profits amid rising global demand.",
    "Federal Reserve raises interest rates to combat inflation.",
    "Starbucks sees a surge in coffee sales in the third quarter.",
    "Major tech companies are investing heavily in artificial intelligence.",
    "Strong dollar impacts the global trade negatively."
]

label_indices = [78, 18, 73, 8, 56]  

for i, sentence in enumerate(sentences):
    print(i)
    interpret_sentence(bert_model_wrapper, sentence=sentence, target_label_idx=label_indices[i])
    visualization.visualize_text(vis_data_records_ig)

"""### SHAP:
SHAP (SHapley Additive exPlanations) is a unified measure of feature importance that allocates the contribution of each feature to the prediction for each sample. This is achieved by averaging the marginal contributions of a feature across all possible subsets of features. SHAP values not only tell you which features are important, but also the magnitude of that feature's effect, and the direction of the effect (positive or negative).
"""

def predictor(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = inputs.to(model.device)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions.detach().cpu().numpy()

# Define a prediction function
def f(texts):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in texts])
    attention_mask = (tv!=0).type(torch.int64)
    outputs = model(tv, attention_mask=attention_mask)[0].detach().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    return scores

# Create an explainer object
explainer = shap.Explainer(f, tokenizer)

# Generate SHAP values for a sample text
shap_values = explainer(["Global stock markets have witnessed a substantial surge amidst renewed investor confidence"])
shap_values

# Visualize the SHAP values
shap.plots.text(shap_values)

sentences = [
    "Oil giant Exxon Mobil sees record profits amid rising global demand.",
    "Federal Reserve raises interest rates to combat inflation.",
    "Starbucks sees a surge in coffee sales in the third quarter.",
    "Major tech companies are investing heavily in artificial intelligence.",
    "Strong dollar impacts the global trade negatively."
]

label_indices = [78, 18, 73, 8, 56]  
shap_ls = []
for i, sentence in enumerate(sentences):
    print(i)
    shap_values = explainer([sentence])
    shap_ls.append(shap_values)
    shap.plots.text(shap_values)

"""### LIME:
LIME
"""

def predictor(texts):
    BATCH_SIZE = 16  # you can adjust this depending on your available memory
    predictions = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        
        # Move inputs to GPU
        inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

        outputs = model(**inputs)
        predictions_batch = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Move predictions back to CPU so they can be used by LIME
        predictions.append(predictions_batch.detach().cpu().numpy())

    return np.concatenate(predictions)

labels = list(label2idx.keys())
explainer = LimeTextExplainer(class_names=labels) # change class names based on your task

# explain a prediction
exp = explainer.explain_instance("Global stock markets have witnessed a substantial surge amidst renewed investor confidence", predictor, num_features=6)

exp.show_in_notebook(text=True)

sentences = [
    "Oil giant Exxon Mobil sees record profits amid rising global demand.",
    "Federal Reserve raises interest rates to combat inflation.",
    "Starbucks sees a surge in coffee sales in the third quarter.",
    "Major tech companies are investing heavily in artificial intelligence.",
    "Strong dollar impacts the global trade negatively."
]

label_indices = [78, 18, 73, 8, 56]  
exp_ls = []
for i, sentence in enumerate(sentences):
    print(i)
    exp = explainer.explain_instance(sentence, predictor, num_features=6)
    exp_ls.append(exp)
    exp.show_in_notebook(text=True)

