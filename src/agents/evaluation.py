"""
Evaluation Agent for RAG System

This agent evaluates the quality of RAG responses using various metrics including
faithfulness, relevance, fluency, and other quality indicators.
"""

import time
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import json

# Core ML and NLP imports
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Advanced evaluation imports
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False

# Sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Local imports
from ..models.document import GenerationResponse, EvaluationMetrics, RetrievalResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationAgent:
    """
    Agent responsible for evaluating RAG response quality using multiple metrics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Evaluation Agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Evaluation configuration
        self.eval_config = self.config.get('evaluation', {})
        self.metrics = self.eval_config.get('metrics', ['faithfulness', 'relevance', 'fluency'])
        self.batch_size = self.eval_config.get('batch_size', 10)
        self.save_results = self.eval_config.get('save_results', True)
        
        # Model initialization
        self.similarity_model = None
        self.rouge_scorer = None
        self.tfidf_vectorizer = None
        
        # Download required NLTK data
        self._setup_nltk()
        
        # Initialize components
        self._initialize_models()
        
        logger.info(f"Evaluation Agent initialized with metrics: {self.metrics}")
    
    def _setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
    
    def _initialize_models(self):
        """Initialize evaluation models."""
        # Initialize sentence transformer for semantic similarity
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                model_name = self.eval_config.get('similarity_model', 'all-MiniLM-L6-v2')
                self.similarity_model = SentenceTransformer(model_name)
                logger.info(f"Loaded similarity model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load similarity model: {str(e)}")
                self.similarity_model = None
        
        # Initialize ROUGE scorer
        if 'relevance' in self.metrics:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize TF-IDF vectorizer for fallback similarity
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    def evaluate_response(self, response: GenerationResponse, 
                         ground_truth: Optional[str] = None,
                         reference_answers: Optional[List[str]] = None) -> EvaluationMetrics:
        """
        Evaluate a single RAG response.
        
        Args:
            response: GenerationResponse object to evaluate
            ground_truth: Ground truth answer (optional)
            reference_answers: List of reference answers (optional)
            
        Returns:
            EvaluationMetrics object
        """
        start_time = time.time()
        
        logger.info(f"Evaluating response for query: '{response.query[:50]}...'")
        
        # Initialize metrics object
        metrics = EvaluationMetrics(response_id=response.id)
        
        try:
            # Evaluate faithfulness (how well grounded in sources)
            if 'faithfulness' in self.metrics:
                metrics.faithfulness_score = self._evaluate_faithfulness(response)
            
            # Evaluate relevance (how well it answers the query)
            if 'relevance' in self.metrics:
                metrics.relevance_score = self._evaluate_relevance(
                    response, ground_truth, reference_answers
                )
            
            # Evaluate fluency (language quality)
            if 'fluency' in self.metrics:
                metrics.fluency_score = self._evaluate_fluency(response.answer)
            
            # Evaluate coherence
            if 'coherence' in self.metrics:
                metrics.coherence_score = self._evaluate_coherence(response.answer)
            
            # Evaluate groundedness (similarity to source content)
            if 'groundedness' in self.metrics:
                metrics.groundedness_score = self._evaluate_groundedness(response)
            
            # Additional metrics
            additional_metrics = self._calculate_additional_metrics(
                response, ground_truth, reference_answers
            )
            metrics.evaluation_metadata.update(additional_metrics)
            
            # Set evaluator model
            metrics.evaluator_model = "EvaluationAgent"
            
            evaluation_time = time.time() - start_time
            metrics.evaluation_metadata['evaluation_time'] = evaluation_time
            
            logger.info(f"Evaluation completed in {evaluation_time:.3f}s")
            
            # Save results if configured
            if self.save_results:
                self._save_evaluation_result(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            metrics.evaluation_metadata['error'] = str(e)
            return metrics
    
    def _evaluate_faithfulness(self, response: GenerationResponse) -> float:
        """
        Evaluate how well the answer is supported by the retrieved context.
        
        Args:
            response: GenerationResponse object
            
        Returns:
            Faithfulness score between 0 and 1
        """
        try:
            if not response.sources or not response.answer:
                return 0.0
            
            # Combine all source content
            source_texts = [result.chunk.content for result in response.sources]
            combined_sources = " ".join(source_texts)
            
            if not combined_sources.strip():
                return 0.0
            
            # Calculate semantic similarity between answer and sources
            if self.similarity_model:
                # Use sentence transformer
                answer_embedding = self.similarity_model.encode([response.answer])
                sources_embedding = self.similarity_model.encode([combined_sources])
                similarity = cosine_similarity(answer_embedding, sources_embedding)[0][0]
            else:
                # Fallback to TF-IDF similarity
                similarity = self._tfidf_similarity(response.answer, combined_sources)
            
            # Check for factual consistency (simple heuristic)
            consistency_score = self._check_factual_consistency(response.answer, source_texts)
            
            # Combine similarity and consistency
            faithfulness_score = 0.7 * similarity + 0.3 * consistency_score
            
            return min(max(faithfulness_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating faithfulness: {str(e)}")
            return 0.0
    
    def _evaluate_relevance(self, response: GenerationResponse,
                          ground_truth: Optional[str] = None,
                          reference_answers: Optional[List[str]] = None) -> float:
        """
        Evaluate how well the answer addresses the original query.
        
        Args:
            response: GenerationResponse object
            ground_truth: Ground truth answer
            reference_answers: List of reference answers
            
        Returns:
            Relevance score between 0 and 1
        """
        try:
            answer = response.answer
            query = response.query
            
            if not answer or not query:
                return 0.0
            
            # Query-answer semantic similarity
            if self.similarity_model:
                query_embedding = self.similarity_model.encode([query])
                answer_embedding = self.similarity_model.encode([answer])
                query_relevance = cosine_similarity(query_embedding, answer_embedding)[0][0]
            else:
                query_relevance = self._tfidf_similarity(query, answer)
            
            # If ground truth is available, compare against it
            ground_truth_relevance = 0.0
            if ground_truth:
                if self.similarity_model:
                    gt_embedding = self.similarity_model.encode([ground_truth])
                    answer_embedding = self.similarity_model.encode([answer])
                    ground_truth_relevance = cosine_similarity(gt_embedding, answer_embedding)[0][0]
                else:
                    ground_truth_relevance = self._tfidf_similarity(ground_truth, answer)
                
                # Use ROUGE scores for additional comparison
                rouge_scores = self.rouge_scorer.score(ground_truth, answer)
                rouge_avg = np.mean([
                    rouge_scores['rouge1'].fmeasure,
                    rouge_scores['rouge2'].fmeasure,
                    rouge_scores['rougeL'].fmeasure
                ])
                ground_truth_relevance = 0.6 * ground_truth_relevance + 0.4 * rouge_avg
            
            # If reference answers are available
            ref_relevance = 0.0
            if reference_answers:
                ref_similarities = []
                for ref in reference_answers:
                    if self.similarity_model:
                        ref_embedding = self.similarity_model.encode([ref])
                        answer_embedding = self.similarity_model.encode([answer])
                        sim = cosine_similarity(ref_embedding, answer_embedding)[0][0]
                    else:
                        sim = self._tfidf_similarity(ref, answer)
                    ref_similarities.append(sim)
                ref_relevance = max(ref_similarities) if ref_similarities else 0.0
            
            # Combine different relevance measures
            if ground_truth and reference_answers:
                relevance_score = 0.4 * query_relevance + 0.4 * ground_truth_relevance + 0.2 * ref_relevance
            elif ground_truth:
                relevance_score = 0.5 * query_relevance + 0.5 * ground_truth_relevance
            elif reference_answers:
                relevance_score = 0.7 * query_relevance + 0.3 * ref_relevance
            else:
                relevance_score = query_relevance
            
            return min(max(relevance_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating relevance: {str(e)}")
            return 0.0
    
    def _evaluate_fluency(self, answer: str) -> float:
        """
        Evaluate the linguistic fluency of the answer.
        
        Args:
            answer: Generated answer text
            
        Returns:
            Fluency score between 0 and 1
        """
        try:
            if not answer or len(answer.strip()) < 10:
                return 0.0
            
            # Basic fluency metrics
            scores = []
            
            # 1. Sentence structure (average sentence length)
            sentences = nltk.sent_tokenize(answer)
            if sentences:
                avg_sentence_length = np.mean([len(sent.split()) for sent in sentences])
                # Optimal range is 15-25 words per sentence
                length_score = 1.0 - abs(avg_sentence_length - 20) / 20
                length_score = max(0.0, min(1.0, length_score))
                scores.append(length_score)
            
            # 2. Repetition penalty
            words = answer.lower().split()
            if words:
                unique_words = set(words)
                repetition_score = len(unique_words) / len(words)
                scores.append(repetition_score)
            
            # 3. Punctuation and capitalization
            punct_score = self._evaluate_punctuation(answer)
            scores.append(punct_score)
            
            # 4. Grammar heuristics (basic)
            grammar_score = self._evaluate_basic_grammar(answer)
            scores.append(grammar_score)
            
            # 5. Coherence indicators
            coherence_indicators = self._check_coherence_indicators(answer)
            scores.append(coherence_indicators)
            
            # Average all scores
            fluency_score = np.mean(scores) if scores else 0.0
            
            return min(max(fluency_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating fluency: {str(e)}")
            return 0.0
    
    def _evaluate_coherence(self, answer: str) -> float:
        """
        Evaluate the coherence and logical flow of the answer.
        
        Args:
            answer: Generated answer text
            
        Returns:
            Coherence score between 0 and 1
        """
        try:
            if not answer:
                return 0.0
            
            sentences = nltk.sent_tokenize(answer)
            if len(sentences) < 2:
                return 1.0  # Single sentence is inherently coherent
            
            # Calculate sentence-to-sentence similarity
            similarities = []
            
            if self.similarity_model:
                # Use sentence transformer for semantic coherence
                sentence_embeddings = self.similarity_model.encode(sentences)
                
                for i in range(len(sentences) - 1):
                    similarity = cosine_similarity(
                        [sentence_embeddings[i]], 
                        [sentence_embeddings[i + 1]]
                    )[0][0]
                    similarities.append(similarity)
            else:
                # Fallback to word overlap
                for i in range(len(sentences) - 1):
                    similarity = self._word_overlap_similarity(sentences[i], sentences[i + 1])
                    similarities.append(similarity)
            
            # Check for logical connectors
            connector_score = self._check_logical_connectors(answer)
            
            # Combine metrics
            if similarities:
                semantic_coherence = np.mean(similarities)
                coherence_score = 0.7 * semantic_coherence + 0.3 * connector_score
            else:
                coherence_score = connector_score
            
            return min(max(coherence_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating coherence: {str(e)}")
            return 0.0
    
    def _evaluate_groundedness(self, response: GenerationResponse) -> float:
        """
        Evaluate how well the answer is grounded in the provided sources.
        
        Args:
            response: GenerationResponse object
            
        Returns:
            Groundedness score between 0 and 1
        """
        try:
            if not response.sources or not response.answer:
                return 0.0
            
            answer = response.answer
            source_contents = [result.chunk.content for result in response.sources]
            
            # Calculate coverage: what percentage of the answer is supported by sources
            answer_sentences = nltk.sent_tokenize(answer)
            supported_sentences = 0
            
            for sentence in answer_sentences:
                max_similarity = 0.0
                
                for source_content in source_contents:
                    if self.similarity_model:
                        sentence_emb = self.similarity_model.encode([sentence])
                        source_emb = self.similarity_model.encode([source_content])
                        similarity = cosine_similarity(sentence_emb, source_emb)[0][0]
                    else:
                        similarity = self._tfidf_similarity(sentence, source_content)
                    
                    max_similarity = max(max_similarity, similarity)
                
                # Threshold for considering a sentence "supported"
                if max_similarity > 0.3:
                    supported_sentences += 1
            
            # Calculate percentage of supported sentences
            if answer_sentences:
                groundedness_score = supported_sentences / len(answer_sentences)
            else:
                groundedness_score = 0.0
            
            return min(max(groundedness_score, 0.0), 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating groundedness: {str(e)}")
            return 0.0
    
    def _calculate_additional_metrics(self, response: GenerationResponse,
                                    ground_truth: Optional[str] = None,
                                    reference_answers: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate additional evaluation metrics."""
        additional_metrics = {}
        
        try:
            # Response length metrics
            additional_metrics['response_length'] = len(response.answer)
            additional_metrics['response_word_count'] = len(response.answer.split())
            additional_metrics['num_sentences'] = len(nltk.sent_tokenize(response.answer))
            
            # Source utilization metrics
            additional_metrics['num_sources_used'] = len(response.sources)
            if response.sources:
                additional_metrics['avg_source_score'] = np.mean([r.score for r in response.sources])
                additional_metrics['max_source_score'] = max([r.score for r in response.sources])
                additional_metrics['min_source_score'] = min([r.score for r in response.sources])
            
            # Context metrics
            additional_metrics['context_length'] = len(response.context_used)
            additional_metrics['context_utilization'] = len(response.answer) / max(len(response.context_used), 1)
            
            # BLEU score if ground truth available
            if ground_truth:
                bleu_score = self._calculate_bleu_score(response.answer, ground_truth)
                additional_metrics['bleu_score'] = bleu_score
            
            # BERTScore if available and ground truth provided
            if BERT_SCORE_AVAILABLE and ground_truth:
                try:
                    P, R, F1 = bert_score([response.answer], [ground_truth], lang='en', verbose=False)
                    additional_metrics['bert_score_precision'] = P.item()
                    additional_metrics['bert_score_recall'] = R.item()
                    additional_metrics['bert_score_f1'] = F1.item()
                except Exception as e:
                    logger.warning(f"BERTScore calculation failed: {str(e)}")
            
            # Processing time metrics
            if response.processing_time:
                additional_metrics['generation_time'] = response.processing_time
            
            return additional_metrics
            
        except Exception as e:
            logger.warning(f"Error calculating additional metrics: {str(e)}")
            return additional_metrics
    
    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity between two texts."""
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def _word_overlap_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate word overlap similarity between sentences."""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _check_factual_consistency(self, answer: str, source_texts: List[str]) -> float:
        """Check for basic factual consistency using simple heuristics."""
        try:
            # Extract numbers and entities from answer and sources
            answer_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', answer)
            answer_entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', answer)
            
            source_numbers = []
            source_entities = []
            
            for source in source_texts:
                source_numbers.extend(re.findall(r'\b\d+(?:\.\d+)?\b', source))
                source_entities.extend(re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', source))
            
            # Check if answer numbers/entities are supported by sources
            supported_numbers = sum(1 for num in answer_numbers if num in source_numbers)
            supported_entities = sum(1 for entity in answer_entities if entity in source_entities)
            
            total_items = len(answer_numbers) + len(answer_entities)
            supported_items = supported_numbers + supported_entities
            
            if total_items == 0:
                return 1.0  # No factual claims to verify
            
            return supported_items / total_items
            
        except Exception:
            return 0.5  # Default moderate score if checking fails
    
    def _evaluate_punctuation(self, text: str) -> float:
        """Evaluate punctuation and capitalization."""
        try:
            # Check for proper sentence endings
            sentences = nltk.sent_tokenize(text)
            proper_endings = sum(1 for sent in sentences if sent.strip().endswith(('.', '!', '?')))
            ending_score = proper_endings / len(sentences) if sentences else 0.0
            
            # Check for proper capitalization
            capital_score = 1.0 if text[0].isupper() else 0.0
            
            return (ending_score + capital_score) / 2
            
        except Exception:
            return 0.5
    
    def _evaluate_basic_grammar(self, text: str) -> float:
        """Basic grammar evaluation using heuristics."""
        try:
            # Simple heuristics for grammar
            scores = []
            
            # Subject-verb agreement (very basic)
            # This is a simplified check
            words = text.lower().split()
            
            # Check for common grammar patterns
            common_errors = ['a data', 'datas', 'informations']
            error_count = sum(1 for error in common_errors if error in text.lower())
            error_score = max(0.0, 1.0 - (error_count * 0.2))
            scores.append(error_score)
            
            # Check for reasonable sentence structure
            sentences = nltk.sent_tokenize(text)
            avg_words_per_sentence = np.mean([len(sent.split()) for sent in sentences]) if sentences else 0
            
            # Penalize very short or very long sentences
            if 5 <= avg_words_per_sentence <= 30:
                structure_score = 1.0
            else:
                structure_score = max(0.0, 1.0 - abs(avg_words_per_sentence - 15) / 15)
            scores.append(structure_score)
            
            return np.mean(scores) if scores else 0.5
            
        except Exception:
            return 0.5
    
    def _check_coherence_indicators(self, text: str) -> float:
        """Check for coherence indicators like transition words."""
        try:
            transition_words = [
                'however', 'therefore', 'furthermore', 'moreover', 'additionally',
                'consequently', 'nevertheless', 'meanwhile', 'similarly', 'in contrast',
                'for example', 'for instance', 'in conclusion', 'first', 'second',
                'finally', 'also', 'thus', 'hence', 'accordingly'
            ]
            
            text_lower = text.lower()
            transition_count = sum(1 for word in transition_words if word in text_lower)
            
            # Normalize by text length
            words_count = len(text.split())
            if words_count < 50:
                # Short texts don't need many transitions
                return 1.0 if transition_count >= 1 else 0.8
            else:
                # Longer texts benefit from more transitions
                transition_ratio = transition_count / (words_count / 50)  # Expected 1 per 50 words
                return min(1.0, transition_ratio)
                
        except Exception:
            return 0.5
    
    def _check_logical_connectors(self, text: str) -> float:
        """Check for logical connectors in the text."""
        try:
            connectors = [
                'because', 'since', 'as', 'due to', 'owing to',
                'if', 'unless', 'provided that', 'although', 'even though',
                'while', 'whereas', 'but', 'yet', 'however'
            ]
            
            text_lower = text.lower()
            connector_count = sum(1 for connector in connectors if connector in text_lower)
            
            sentences = nltk.sent_tokenize(text)
            if len(sentences) <= 1:
                return 1.0
            
            # Expect some connectors in multi-sentence text
            expected_connectors = max(1, len(sentences) // 3)
            connector_score = min(1.0, connector_count / expected_connectors)
            
            return connector_score
            
        except Exception:
            return 0.5
    
    def _calculate_bleu_score(self, hypothesis: str, reference: str) -> float:
        """Calculate BLEU score between hypothesis and reference."""
        try:
            # Tokenize
            hypothesis_tokens = hypothesis.lower().split()
            reference_tokens = reference.lower().split()
            
            # Calculate BLEU score
            smoothing = SmoothingFunction().method1
            score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing)
            
            return score
            
        except Exception as e:
            logger.warning(f"BLEU score calculation failed: {str(e)}")
            return 0.0
    
    def _save_evaluation_result(self, metrics: EvaluationMetrics):
        """Save evaluation results to file."""
        try:
            import os
            os.makedirs('results', exist_ok=True)
            
            filename = f"results/evaluation_{metrics.response_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2, default=str)
            
            logger.debug(f"Evaluation results saved to {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save evaluation results: {str(e)}")
    
    def evaluate_batch(self, responses: List[GenerationResponse],
                      ground_truths: Optional[List[str]] = None) -> List[EvaluationMetrics]:
        """
        Evaluate multiple responses in batch.
        
        Args:
            responses: List of GenerationResponse objects
            ground_truths: List of ground truth answers (optional)
            
        Returns:
            List of EvaluationMetrics objects
        """
        logger.info(f"Starting batch evaluation of {len(responses)} responses")
        
        results = []
        
        for i, response in enumerate(responses):
            ground_truth = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            
            try:
                metrics = self.evaluate_response(response, ground_truth)
                results.append(metrics)
            except Exception as e:
                logger.error(f"Failed to evaluate response {i}: {str(e)}")
                # Create error metrics
                error_metrics = EvaluationMetrics(
                    response_id=response.id,
                    evaluation_metadata={'error': str(e)}
                )
                results.append(error_metrics)
        
        # Calculate batch statistics
        self._calculate_batch_statistics(results)
        
        logger.info(f"Batch evaluation completed: {len(results)} responses evaluated")
        return results
    
    def _calculate_batch_statistics(self, results: List[EvaluationMetrics]) -> Dict[str, float]:
        """Calculate statistics across a batch of evaluation results."""
        try:
            valid_results = [r for r in results if r.overall_score is not None]
            
            if not valid_results:
                return {}
            
            stats = {}
            
            # Overall score statistics
            overall_scores = [r.overall_score for r in valid_results if r.overall_score is not None]
            if overall_scores:
                stats['avg_overall_score'] = np.mean(overall_scores)
                stats['std_overall_score'] = np.std(overall_scores)
                stats['min_overall_score'] = np.min(overall_scores)
                stats['max_overall_score'] = np.max(overall_scores)
            
            # Individual metric statistics
            for metric in ['faithfulness_score', 'relevance_score', 'fluency_score', 'coherence_score', 'groundedness_score']:
                scores = [getattr(r, metric) for r in valid_results if getattr(r, metric) is not None]
                if scores:
                    stats[f'avg_{metric}'] = np.mean(scores)
                    stats[f'std_{metric}'] = np.std(scores)
            
            logger.info(f"Batch statistics calculated: avg_overall_score = {stats.get('avg_overall_score', 0):.3f}")
            return stats
            
        except Exception as e:
            logger.warning(f"Error calculating batch statistics: {str(e)}")
            return {}
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation capabilities and configuration."""
        return {
            'available_metrics': self.metrics,
            'models': {
                'similarity_model': self.similarity_model.get_sentence_embedding_dimension() if self.similarity_model else None,
                'rouge_scorer': 'available' if self.rouge_scorer else 'not available',
                'bert_score': 'available' if BERT_SCORE_AVAILABLE else 'not available'
            },
            'configuration': {
                'batch_size': self.batch_size,
                'save_results': self.save_results
            },
            'capabilities': {
                'faithfulness': 'measures grounding in sources',
                'relevance': 'measures query answering quality',
                'fluency': 'measures language quality',
                'coherence': 'measures logical flow',
                'groundedness': 'measures source support'
            }
        }