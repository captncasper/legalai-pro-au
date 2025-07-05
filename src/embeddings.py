"""Generate and manage document embeddings"""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
from typing import List, Tuple


class LegalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings"""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
