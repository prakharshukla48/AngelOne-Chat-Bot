from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv
import re

load_dotenv()

class LLMInterface:
    def __init__(self, model_type="flan-t5"):
        print("Loading local language model...")
        self.model_type = model_type
        
        try:
            if model_type == "flan-t5":
                model_name = "google/flan-t5-small"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                print("Loaded FLAN-T5 model for question answering")
                
            elif model_type == "gpt2":
                model_name = "distilgpt2"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                print("Loaded DistilGPT-2 model for text generation")
            
            else:
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    tokenizer="distilbert-base-cased-distilled-squad"
                )
                self.tokenizer = None
                self.model = None
                print("Loaded DistilBERT Q&A pipeline")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to simple template responses...")
            self.model = None
            self.tokenizer = None
            self.qa_pipeline = None
    
    def generate_response(self, query: str, context: str, max_tokens: int = 150) -> str:

        if not self._is_meaningful_query(query):
            return "I can only answer questions related to insurance, trading, and customer support. Please ask a relevant question."
        

        if not context or len(context.strip()) < 20:
            return "I don't know. Please ask about insurance policies, trading, account opening, or customer support topics."

        context = self._clean_context(context)
        
        try:
            if self.model_type == "flan-t5" and self.model and self.tokenizer:
                return self._generate_with_flan_t5(query, context, max_tokens)
            
            elif self.model_type == "gpt2" and self.model and self.tokenizer:
                return self._generate_with_gpt2(query, context, max_tokens)
            
            elif hasattr(self, 'qa_pipeline') and self.qa_pipeline:
                return self._generate_with_qa_pipeline(query, context)
            
            else:
                return self._fallback_response(query, context)
                
        except Exception as e:
            print(f"Error during generation: {e}")
            return self._fallback_response(query, context)
    
    def _is_meaningful_query(self, query: str) -> bool:
        
        import re
        

        words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
        
        if len(words) == 0:
            return False
        
        
        gibberish_patterns = [
            r'(.)\1{3,}',  
            r'^(da|ba|ja|ka|la|ma|na|pa|ra|sa|ta|wa|za){2,}$',  
            r'^[aeiou]+$',  
            r'^[bcdfghjklmnpqrstvwxyz]+$'  
        ]
        
        meaningful_words = 0
        for word in words:
            is_gibberish = any(re.search(pattern, word) for pattern in gibberish_patterns)
            if not is_gibberish and len(set(word)) >= 2:
                meaningful_words += 1
        
        
        return meaningful_words > 0
    
    def _generate_with_flan_t5(self, query: str, context: str, max_tokens: int) -> str:
        prompt = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_tokens,
                num_beams=2,
                temperature=0.7,
                do_sample=True,
                early_stopping=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    
    def _generate_with_gpt2(self, query: str, context: str, max_tokens: int) -> str:
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=400, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + max_tokens,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(prompt):].strip()
        
        answer = answer.split('\n')[0]
        return answer if answer else "I don't Know."
    
    def _generate_with_qa_pipeline(self, query: str, context: str) -> str:
        try:
            result = self.qa_pipeline(question=query, context=context)
            confidence = result['score']
            
            if confidence > 0.1:
                return result['answer']
            else:
                return "I don't Know."
        except:
            return self._fallback_response(query, context)
    
    def _fallback_response(self, query: str, context: str) -> str:
        query_lower = query.lower()
        context_sentences = context.split('.')
        
        relevant_sentences = []
        query_words = set(query_lower.split())
        
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            if query_words.intersection(sentence_words):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return ". ".join(relevant_sentences[:2]) + "."
        
        keywords_responses = {
            "account": "To open an account, please visit our website and follow the registration process.",
            "login": "For login issues, please check your credentials and contact support.",
            "trading": "For trading queries, please refer to our trading guidelines.",
            "fees": "Fee information can be found in your policy documents.",
            "claim": "To file a claim, contact our claims department with required documents.",
            "support": "For additional support, please contact our customer service."
        }
        
        for keyword, response in keywords_responses.items():
            if keyword in query_lower:
                return response
        
        return "I Don't know"
    
    def _clean_context(self, context: str) -> str:
        context = re.sub(r'\s+', ' ', context.strip())
        if len(context) > 800:
            context = context[:800] + "..."
        return context
    
    def format_context(self, search_results):
        context_parts = []
        for i, (text, score) in enumerate(search_results[:3]):
            truncated_text = text[:250] + "..." if len(text) > 250 else text
            context_parts.append(truncated_text)
        
        return " ".join(context_parts)

if __name__ == "__main__":
    print("Testing LLM Interface...")
    
    for model_type in ["flan-t5", "qa-pipeline"]:
        print(f"\n=== Testing with {model_type} ===")
        try:
            llm = LLMInterface(model_type=model_type)
            
            context = "AngelOne is a stock trading platform. To open an account, visit our website and click 'Open Account'. You need to provide KYC documents."
            query = "How do I open an account?"
            
            response = llm.generate_response(query, context)
            print(f"Query: {query}")
            print(f"Response: {response}")
            
        except Exception as e:
            print(f"Error testing {model_type}: {e}")
