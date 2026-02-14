import google.generativeai as genai
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_answer_batched(self, question: str, chunks: List[Dict], batch_size: int = 5) -> Dict:
        """
        Generate answer using batched prompts with Gemini API
        Groups chunks into batches and sends them together
        """
        try:
            # Group chunks into batches
            batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
            
            # Process each batch and collect results
            all_results = []
            for batch_num, batch in enumerate(batches):
                # Create prompt with all chunks in this batch
                prompt = self._create_prompt(question, batch)
                
                # Send to Gemini API
                response = self.model.generate_content(prompt)
                
                all_results.append({
                    'batch': batch_num,
                    'response': response.text,
                    'usage': {
                        'prompt_tokens': len(prompt.split()),
                        'candidates_tokens': len(response.text.split()),
                        'total_tokens': len(prompt.split()) + len(response.text.split())
                    }
                })
            
            # Combine results from all batches
            final_answer = self._combine_answers(all_results, question)
            
            # Calculate total token usage
            total_tokens = {
                'prompt_tokens': sum(r['usage']['prompt_tokens'] for r in all_results),
                'candidates_tokens': sum(r['usage']['candidates_tokens'] for r in all_results),
                'total_tokens': sum(r['usage']['total_tokens'] for r in all_results)
            }
            
            return {
                'answer': final_answer,
                'batch_size': len(batches),
                'tokens_used': total_tokens
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def _create_prompt(self, question: str, chunks: List[Dict]) -> str:
        """Create prompt with question and context chunks"""
        prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the provided text chunks. 
Cite the chunk IDs for each part of your answer. If the answer is not in the provided text, say so.

USER_QUESTION: "{question}"

CONTEXT_CHUNKS:
"""
        
        for chunk in chunks:
            prompt += f"\n[CHUNK {chunk['chunk_id']}]: {chunk['text'][:500]}..."
        
        prompt += "\n\nPlease provide a clear and concise answer with citations."
        
        return prompt
    
    def _combine_answers(self, results: List[Dict], question: str) -> str:
        """Combine answers from multiple batches"""
        if len(results) == 1:
            return results[0]['response']
        
        # Create a summary combining all batch results
        summary_prompt = f"""Based on the following answers to the question "{question}", provide a consolidated response:

"""
        
        for i, result in enumerate(results):
            summary_prompt += f"\nAnswer from batch {i+1}: {result['response']}"
        
        summary_prompt += "\n\nProvide a single, coherent answer."
        
        response = self.model.generate_content(summary_prompt)
        return response.text
