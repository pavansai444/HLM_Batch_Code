import argparse
import torch
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch
from sklearn.metrics import f1_score
import os
import warnings

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run training with configurable batch size and learning rate.")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for training.")
parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for training.")
parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization.")
parser.add_argument("--encoder_model", type=str, default="FacebookAI/xlm-roberta-base", help="Encoder model to use (e.g., bert, xlm-roberta).")
parser.add_argument("--decoder_model", type=str, default="google/gemma-2-2b", help="Decoder model to use (e.g., llama 3.2, gpt, google/gemma-2-2b).")
parser.add_argument("--result_dir", type=str, default="results", help="Directory to save results.")
parser.add_argument("--data_dir", type=str, default="engtel", help="Directory containing the dataset files train test and validation.")
parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use (default: 0).")
args = parser.parse_args()
batch_size = args.batch_size
lr = args.learning_rate
num_epochs = args.num_epochs
max_length = args.max_length
encoder_model = args.encoder_model
decoder_model = args.decoder_model
result_dir = args.result_dir
data_dir = args.data_dir
device_id = args.gpu

# Print the parsed arguments for the user
print("Arguments:")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {lr}")
print(f"Number of Epochs: {num_epochs}")
print(f"Maximum Sequence Length: {max_length}")
print(f"Encoder Model: {encoder_model}")
print(f"Decoder Model: {decoder_model}")
print(f"Result Directory: {result_dir}")
print(f"Data Directory: {data_dir}")
print(f"GPU Device ID: {device_id}")

# Suppress warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available():       
    device = torch.device(f"cuda:{device_id}")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name())
else:
    
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Load the dataset
X_train = []
y_train = []
X_eval = []
y_eval = []
X_test=[]
y_test = []
with open(os.path.join(data_dir, 'train.txt'), encoding='UTF-8') as rf:
    for lin in rf.readlines():
        da = lin.split('\t')
        X_train.append(da[0])
        lab = da[1].strip()
        y_train.append(0 if lab == "neutral" else 1 if lab == "positive" else 2)
with open(os.path.join(data_dir, 'validation.txt'), encoding='UTF-8') as rf:
    for lin in rf.readlines():
        da = lin.split('\t')
        X_eval.append(da[0])
        lab = da[1].strip()
        y_eval.append(0 if lab == "neutral" else 1 if lab == "positive" else 2)
with open(os.path.join(data_dir, 'test.txt'),encoding='UTF-8') as rf:
    lines = rf.readlines()
    for lin in lines:
        da = lin.split('\t')
        lab = da[1].strip()
        X_test.append(da[0])
        if lab == "neutral":
            y_test.append(0)
        if lab == "positive":
            y_test.append(1)
        if lab == "negative":
            y_test.append(2)

# Model and tokenizer setup
# BERT
berttokenizer = AutoTokenizer.from_pretrained(encoder_model)
bertmodel = AutoModel.from_pretrained(encoder_model).to(device)

# LLM with LoRA
llm_model_name = decoder_model  # Use the provided decoder model name
gpttokenizer = AutoTokenizer.from_pretrained(llm_model_name)
gpttokenizer.add_special_tokens({'pad_token': '[PAD]'})
llmmodel = AutoModel.from_pretrained(llm_model_name, trust_remote_code=True)
llmmodel.resize_token_embeddings(len(gpttokenizer))  # Fix: Resize embeddings
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "down_proj", "gate_proj", "up_proj"]
)
gptmodel = get_peft_model(llmmodel, lora_config).to(device)# Model and tokenizer setup

# Classifier (without softmax)
class Classifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 768)
        self.fc2 = nn.Linear(768, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Raw logits
        return x

classifier = Classifier(bertmodel.config.hidden_size + gptmodel.config.hidden_size, 3).to(device)

# Optimizer and loss
optimizer = torch.optim.AdamW(
    list(bertmodel.parameters()) + list(gptmodel.parameters()) + list(classifier.parameters()),
    lr=5e-4
)
criterion = nn.CrossEntropyLoss()

# batch_size = 4  # Increased batch size for better efficiency

def batch_tokenize(tokenizer, texts, device):
    # Tokenize a list of texts and return tensors directly on the specified device
    encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length = max_length)
    return {k: v.to(device) for k, v in encoded.items()}

def predict(bertmodel, gptmodel, classifier, device=device, cur_epoch=0):
    bertmodel.eval()
    gptmodel.eval()
    classifier.eval()

    y_pred = []
    y_act = []

    n = len(X_eval)
    for i in tqdm(range(0, n, batch_size)):
        batch_texts = X_eval[i:i+batch_size]
        batch_labels = y_eval[i:i+batch_size]

        # Tokenize and move to device in one step
        inputs_bert = batch_tokenize(berttokenizer, batch_texts, device)
        inputs_gpt = batch_tokenize(gpttokenizer, batch_texts, device)

        with torch.no_grad():
            # Get model outputs
            outputs_bert = bertmodel(**inputs_bert)
            outputs_gpt = gptmodel(**inputs_gpt)

            # Extract hidden states properly
            bert_hidden_states = outputs_bert[0].mean(dim=1)  # Using indexing for consistency
            gpt_hidden_states = outputs_gpt[0].mean(dim=1)    # Using indexing for consistency

            # Combine representations
            representation = torch.cat([bert_hidden_states, gpt_hidden_states], dim=1)

            # Get predictions
            logits = classifier(representation)
            preds = torch.argmax(logits, dim=1).cpu().tolist()

        y_pred.extend(preds)
        y_act.extend(batch_labels)

    correct = sum([p == a for p, a in zip(y_pred, y_act)])
    accuracy = correct / n
    f1 = f1_score(y_act, y_pred, average='weighted')
    # Save validation results
    with open(f'{result_dir}/validation_results_epoch_{cur_epoch+1}_batch_{batch_size}_lr_{lr:.0e}.txt', 'w') as file:
        mapper = {0: 'neutral', 1: 'positive', 2: 'negative'}
        file.write("sentence\tground truth\tpredicted\n")
        for i in range(len(X_eval)):
            file.write(f"{X_eval[i]}\t{mapper[y_eval[i]]}\t{mapper[y_pred[i]]}\n")
            
    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation F1 Score: {f1:.4f}')
    
    return accuracy, f1

def test(bertmodel, gptmodel, classifier, device=device):
    bertmodel.eval()
    gptmodel.eval()
    classifier.eval()

    y_pred = []
    n = len(X_test)
    
    for i in tqdm(range(0, n, batch_size)):
        batch_texts = X_test[i:i+batch_size]
        
        # Tokenize and move to device in one step
        inputs_bert = batch_tokenize(berttokenizer, batch_texts, device)
        inputs_gpt = batch_tokenize(gpttokenizer, batch_texts, device)

        with torch.no_grad():
            # Get model outputs
            outputs_bert = bertmodel(**inputs_bert)
            outputs_gpt = gptmodel(**inputs_gpt)

            # Extract hidden states properly
            bert_hidden_states = outputs_bert[0].mean(dim=1)
            gpt_hidden_states = outputs_gpt[0].mean(dim=1)

            # Combine representations
            representation = torch.cat([bert_hidden_states, gpt_hidden_states], dim=1)

            # Get predictions
            logits = classifier(representation)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            
        y_pred.extend(preds)
    
    return y_pred

# Check if the result directory exists, if not, create it
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    print(f"Created directory: {result_dir}")

test_f1s = []
val_f1s = []

def train(bertmodel, gptmodel, classifier, device=device, num_epochs=3):
    bertmodel.train()
    gptmodel.train()
    classifier.train()
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        list(bertmodel.parameters()) + 
        list(gptmodel.parameters()) + 
        list(classifier.parameters()), 
        lr=lr  # Slightly lower learning rate for stability
    )
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    n = len(X_train)
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        bertmodel.train()
        gptmodel.train()
        classifier.train()

        # Create indices and shuffle them for each epoch
        indices = torch.randperm(n).tolist()
        
        for i in tqdm(range(0, n, batch_size)):
            # Get batch indices
            batch_indices = indices[i:i+batch_size]
            if not batch_indices:  # Skip empty batches
                continue
                
            # Get batch data
            batch_texts = [X_train[idx] for idx in batch_indices]
            batch_labels = [y_train[idx] for idx in batch_indices]
            
            # Tokenize and move to device
            inputs_bert = batch_tokenize(berttokenizer, batch_texts, device)
            inputs_gpt = batch_tokenize(gpttokenizer, batch_texts, device)
            
            # Forward pass
            outputs_bert = bertmodel(**inputs_bert)
            outputs_gpt = gptmodel(**inputs_gpt)

            # Get hidden states
            bert_hidden_states = outputs_bert[0].mean(dim=1)
            gpt_hidden_states = outputs_gpt[0].mean(dim=1)

            # Combine embeddings
            representation = torch.cat([bert_hidden_states, gpt_hidden_states], dim=1)

            # Get predictions
            logits = classifier(representation)

            # Calculate loss
            target = torch.tensor(batch_labels).to(device)
            loss = criterion(logits, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                list(bertmodel.parameters()) + 
                list(gptmodel.parameters()) + 
                list(classifier.parameters()),
                max_norm=1.0
            )
            
            optimizer.step()

            running_loss += loss.item() * len(batch_labels)

        avg_loss = running_loss / n
        print(f"------------\nEPOCH : {epoch+1}/{num_epochs} | Average Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_acc, val_f1 = predict(bertmodel, gptmodel, classifier, device, epoch)
        val_f1s.append(f"Epoch {epoch + 1} : Accuracy {val_acc:.4f} : F1 {val_f1:.4f}")
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"New best F1 score: {best_f1:.4f} - Saving model")
            torch.save({
                'bert_state_dict': bertmodel.state_dict(),
                'gpt_state_dict': gptmodel.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'f1_score': val_f1
            }, f'{result_dir}/best_model_epoch_{num_epochs}_batch_{batch_size}_lr_{lr:.0e}.pt')
        
        # Generate predictions for test set
        print("Generating test predictions...")
        y_pred = test(bertmodel, gptmodel, classifier, device)
        
        # Save predictions
        with open(f'{result_dir}/results_batch_{batch_size}_epoch_{epoch+1}_lr_{lr:.0e}.txt', 'w') as file:
            mapper = {0: 'neutral', 1: 'positive', 2: 'negative'}
            file.write("sentence\tground truth\tpredicted\n")
            for i in range(len(X_test)):
                file.write(f"{X_test[i]}\t{mapper[y_test[i]]}\t{mapper[y_pred[i]]}\n")
        # Append epoch number and F1 score to test_f1s as a string
        test_f1s.append(f"Epoch {epoch + 1} : f1_score {f1_score(y_test, y_pred, average='weighted'):.4f}")
    return bertmodel, gptmodel, classifier

bert, gpt, classifier = train(bertmodel, gptmodel, classifier, device=device, num_epochs=num_epochs)

# Save the test F1 scores to a file
with open(f'{result_dir}/test_f1s_score_b_{batch_size}_ep_{num_epochs}_lr_{lr:.0e}.txt', 'w') as f:
    for score in test_f1s:
        f.write(score + '\n')
print(f"Test F1 scores saved to {result_dir}/test_f1s_score_b_{batch_size}_ep_{num_epochs}_lr_{lr:.0e}.txt")

# Save the validation F1 scores to a file
with open(f'{result_dir}/val_f1s_score_b_{batch_size}_ep_{num_epochs}_lr_{lr:.0e}.txt', 'w') as f:
    for score in val_f1s:
        f.write(score + '\n')
print(f"Validation F1 scores saved to {result_dir}/val_f1s_score_b_{batch_size}_ep_{num_epochs}_lr_{lr:.0e}.txt")
