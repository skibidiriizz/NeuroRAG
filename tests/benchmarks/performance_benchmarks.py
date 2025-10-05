"""
Performance benchmarking suite for RAG Agent System.

This module provides comprehensive performance testing and benchmarking
capabilities to measure system performance under various conditions.
"""

import pytest
import time
import statistics
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any
import json
import psutil
import os
from unittest.mock import Mock

from src.core.rag_system import RAGSystem
from src.models.document import Document, ProcessingStatus, GenerationResponse


class PerformanceMetrics:
    """Class to track and analyze performance metrics."""
    
    def __init__(self):
        self.execution_times = []
        self.memory_usage = []
        self.cpu_usage = []
        self.throughput_metrics = []
        self.error_counts = 0
        self.success_counts = 0
    
    def add_timing(self, duration: float):
        """Add execution time measurement."""
        self.execution_times.append(duration)
    
    def add_memory_usage(self, usage_mb: float):
        """Add memory usage measurement."""
        self.memory_usage.append(usage_mb)
    
    def add_cpu_usage(self, usage_percent: float):
        """Add CPU usage measurement."""
        self.cpu_usage.append(usage_percent)
    
    def record_success(self):
        """Record successful operation."""
        self.success_counts += 1
    
    def record_error(self):
        """Record failed operation."""
        self.error_counts += 1
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics."""
        if not self.execution_times:
            return {}
        
        return {
            "mean": statistics.mean(self.execution_times),
            "median": statistics.median(self.execution_times),
            "std_dev": statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0.0,
            "min": min(self.execution_times),
            "max": max(self.execution_times),
            "p95": sorted(self.execution_times)[int(0.95 * len(self.execution_times))],
            "p99": sorted(self.execution_times)[int(0.99 * len(self.execution_times))]
        }
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if not self.memory_usage:
            return {}
        
        return {
            "mean_mb": statistics.mean(self.memory_usage),
            "peak_mb": max(self.memory_usage),
            "min_mb": min(self.memory_usage)
        }
    
    def get_throughput_stats(self) -> Dict[str, float]:
        """Get throughput statistics."""
        if not self.execution_times:
            return {"operations_per_second": 0.0}
        
        total_time = sum(self.execution_times)
        total_operations = len(self.execution_times)
        
        return {
            "operations_per_second": total_operations / total_time if total_time > 0 else 0.0,
            "total_operations": total_operations,
            "total_time_seconds": total_time
        }
    
    def get_error_rate(self) -> float:
        """Get error rate percentage."""
        total = self.success_counts + self.error_counts
        return (self.error_counts / total * 100) if total > 0 else 0.0
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "timing_stats": self.get_timing_stats(),
            "memory_stats": self.get_memory_stats(),
            "throughput_stats": self.get_throughput_stats(),
            "error_rate_percent": self.get_error_rate(),
            "success_count": self.success_counts,
            "error_count": self.error_counts
        }


class BenchmarkRunner:
    """Utility class for running performance benchmarks."""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.metrics = PerformanceMetrics()
        self.process = psutil.Process()
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """Monitor current system resource usage."""
        return {
            "memory_mb": self.process.memory_info().rss / 1024 / 1024,
            "cpu_percent": self.process.cpu_percent()
        }
    
    def time_operation(self, operation, *args, **kwargs):
        """Time an operation and record metrics."""
        start_resources = self.monitor_system_resources()
        start_time = time.time()
        
        try:
            result = operation(*args, **kwargs)
            self.metrics.record_success()
            return result
        except Exception as e:
            self.metrics.record_error()
            raise
        finally:
            end_time = time.time()
            end_resources = self.monitor_system_resources()
            
            duration = end_time - start_time
            self.metrics.add_timing(duration)
            self.metrics.add_memory_usage(end_resources["memory_mb"])
            self.metrics.add_cpu_usage(end_resources["cpu_percent"])
    
    def run_batch_operation(self, operation, items: List[Any], max_workers: int = 1):
        """Run operation on batch of items with optional parallelism."""
        if max_workers == 1:
            # Sequential processing
            results = []
            for item in items:
                result = self.time_operation(operation, item)
                results.append(result)
            return results
        else:
            # Parallel processing
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {
                    executor.submit(self.time_operation, operation, item): item 
                    for item in items
                }
                
                for future in concurrent.futures.as_completed(future_to_item):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Operation failed: {e}")
                        results.append(None)
            
            return results


@pytest.mark.slow
@pytest.mark.benchmark
class TestDocumentProcessingBenchmarks:
    """Benchmarks for document processing performance."""
    
    @pytest.fixture
    def benchmark_rag_system(self, mock_config):
        """Create RAG system for benchmarking."""
        rag = RAGSystem()
        rag.config = mock_config
        
        # Mock agents with realistic delays
        rag.document_parser = Mock()
        rag.chunking_embedding = Mock()
        
        def mock_parse_with_delay(file_path):
            time.sleep(0.01)  # Simulate parsing time
            return Document(
                filename=Path(file_path).name,
                file_path=file_path,
                content=f"Processed content from {file_path}",
                document_type="text",
                processing_status=ProcessingStatus.PROCESSED
            )
        
        def mock_chunk_with_delay(document):
            time.sleep(0.005)  # Simulate chunking time
            return ["chunk1", "chunk2", "chunk3"]
        
        rag.document_parser.parse_document.side_effect = mock_parse_with_delay
        rag.chunking_embedding.process_document.side_effect = mock_chunk_with_delay
        
        return rag
    
    def test_single_document_processing_performance(self, benchmark_rag_system, temp_dir):
        """Benchmark single document processing."""
        rag = benchmark_rag_system
        runner = BenchmarkRunner(rag)
        
        # Create test document
        test_file = Path(temp_dir) / "benchmark_doc.txt"
        test_file.write_text("This is benchmark content " * 100)  # ~2.7KB
        
        # Benchmark single document processing
        num_iterations = 10
        for i in range(num_iterations):
            runner.time_operation(rag.add_document, str(test_file))
        
        # Generate performance report
        report = runner.metrics.generate_report()
        
        # Performance assertions
        timing_stats = report["timing_stats"]
        assert timing_stats["mean"] < 0.1  # Should process in under 100ms on average
        assert report["error_rate_percent"] == 0.0  # No errors expected
        
        print(f"\nSingle Document Processing Performance:")
        print(f"  Mean time: {timing_stats['mean']:.3f}s")
        print(f"  P95 time: {timing_stats['p95']:.3f}s")
        print(f"  Memory peak: {report['memory_stats']['peak_mb']:.1f}MB")
    
    def test_batch_document_processing_performance(self, benchmark_rag_system, temp_dir):
        """Benchmark batch document processing."""
        rag = benchmark_rag_system
        runner = BenchmarkRunner(rag)
        
        # Create multiple test documents
        document_sizes = [1, 5, 10, 20, 50]  # KB
        test_files = []
        
        for size_kb in document_sizes:
            content = "Test content " * (size_kb * 80)  # Approximate KB
            file_path = Path(temp_dir) / f"doc_{size_kb}kb.txt"
            file_path.write_text(content)
            test_files.append(str(file_path))
        
        # Benchmark batch processing
        results = runner.run_batch_operation(rag.add_document, test_files)
        
        # Verify all documents processed successfully
        assert len(results) == len(test_files)
        assert all(doc.processing_status == ProcessingStatus.PROCESSED for doc in results)
        
        # Generate performance report
        report = runner.metrics.generate_report()
        timing_stats = report["timing_stats"]
        
        # Performance assertions
        assert timing_stats["mean"] < 0.2  # Average under 200ms per document
        assert report["error_rate_percent"] == 0.0
        
        print(f"\nBatch Document Processing Performance ({len(test_files)} documents):")
        print(f"  Mean time per doc: {timing_stats['mean']:.3f}s")
        print(f"  Total throughput: {report['throughput_stats']['operations_per_second']:.1f} docs/sec")
        print(f"  Memory usage: {report['memory_stats']['mean_mb']:.1f}MB average")
    
    def test_concurrent_document_processing(self, benchmark_rag_system, temp_dir):
        """Benchmark concurrent document processing."""
        rag = benchmark_rag_system
        runner = BenchmarkRunner(rag)
        
        # Create test documents
        test_files = []
        for i in range(20):
            content = f"Document {i} content " * 50
            file_path = Path(temp_dir) / f"concurrent_doc_{i}.txt"
            file_path.write_text(content)
            test_files.append(str(file_path))
        
        # Test with different levels of concurrency
        for max_workers in [1, 2, 4, 8]:
            runner.metrics = PerformanceMetrics()  # Reset metrics
            
            start_time = time.time()
            results = runner.run_batch_operation(
                rag.add_document, 
                test_files, 
                max_workers=max_workers
            )
            total_time = time.time() - start_time
            
            successful_results = [r for r in results if r and r.processing_status == ProcessingStatus.PROCESSED]
            throughput = len(successful_results) / total_time
            
            print(f"\nConcurrency Level {max_workers}:")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Throughput: {throughput:.1f} docs/sec")
            print(f"  Success rate: {len(successful_results)}/{len(test_files)}")


@pytest.mark.slow
@pytest.mark.benchmark
class TestQueryProcessingBenchmarks:
    """Benchmarks for query processing performance."""
    
    @pytest.fixture
    def query_benchmark_rag_system(self, mock_config):
        """Create RAG system for query benchmarking."""
        rag = RAGSystem()
        rag.config = mock_config
        
        # Mock agents with realistic delays
        rag.retrieval = Mock()
        rag.generation = Mock()
        
        def mock_retrieve_with_delay(query, top_k=5, score_threshold=0.0):
            time.sleep(0.02)  # Simulate retrieval time
            return [
                {"content": f"Retrieved content {i} for {query[:20]}", "score": 0.9 - i*0.05}
                for i in range(top_k)
            ]
        
        def mock_generate_with_delay(query, retrieval_results, **kwargs):
            time.sleep(0.05)  # Simulate generation time
            return GenerationResponse(
                answer=f"Generated answer for: {query}",
                sources=retrieval_results,
                query=query,
                model_used="benchmark_model",
                processing_time=0.05
            )
        
        rag.retrieval.retrieve.side_effect = mock_retrieve_with_delay
        rag.generation.generate_response.side_effect = mock_generate_with_delay
        
        return rag
    
    def test_query_response_time_benchmark(self, query_benchmark_rag_system):
        """Benchmark query response times."""
        rag = query_benchmark_rag_system
        runner = BenchmarkRunner(rag)
        
        # Test queries of different lengths and complexity
        test_queries = [
            "What is AI?",
            "How does machine learning work in practice?",
            "Explain the differences between supervised, unsupervised, and reinforcement learning approaches in detail.",
            "What are the ethical implications of artificial intelligence deployment in healthcare systems, and how can we ensure fairness and transparency in algorithmic decision-making processes?"
        ]
        
        # Benchmark each query type
        for i, query in enumerate(test_queries):
            query_metrics = PerformanceMetrics()
            query_runner = BenchmarkRunner(rag)
            query_runner.metrics = query_metrics
            
            # Run multiple iterations
            for _ in range(5):
                query_runner.time_operation(rag.query, query)
            
            report = query_metrics.generate_report()
            timing_stats = report["timing_stats"]
            
            print(f"\nQuery Length {len(query)} chars:")
            print(f"  Mean response time: {timing_stats['mean']:.3f}s")
            print(f"  P95 response time: {timing_stats['p95']:.3f}s")
            print(f"  Query: {query[:50]}...")
            
            # Performance assertions based on query complexity
            max_expected_time = 0.1 + (len(query) / 1000)  # Base time + complexity factor
            assert timing_stats['mean'] < max_expected_time, f"Query too slow: {timing_stats['mean']:.3f}s"
    
    def test_concurrent_query_processing(self, query_benchmark_rag_system):
        """Benchmark concurrent query processing."""
        rag = query_benchmark_rag_system
        
        # Simulate multiple users querying simultaneously
        queries = [
            f"What is artificial intelligence topic {i}?" 
            for i in range(50)
        ]
        
        def process_query(query):
            start_time = time.time()
            result = rag.query(query)
            duration = time.time() - start_time
            return {"query": query, "result": result, "duration": duration}
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        
        for max_workers in concurrency_levels:
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_query, queries[:20]))  # Process 20 queries
            
            total_time = time.time() - start_time
            avg_response_time = sum(r["duration"] for r in results) / len(results)
            throughput = len(results) / total_time
            
            print(f"\nConcurrent Queries (workers={max_workers}):")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Average response time: {avg_response_time:.3f}s")
            print(f"  Throughput: {throughput:.1f} queries/sec")
            print(f"  Successful queries: {len([r for r in results if r['result']])}/{len(results)}")
    
    def test_query_parameter_impact(self, query_benchmark_rag_system):
        """Benchmark impact of different query parameters."""
        rag = query_benchmark_rag_system
        runner = BenchmarkRunner(rag)
        
        base_query = "What are the applications of machine learning?"
        
        # Test different top_k values
        top_k_values = [1, 3, 5, 10, 20]
        
        for top_k in top_k_values:
            metrics = PerformanceMetrics()
            test_runner = BenchmarkRunner(rag)
            test_runner.metrics = metrics
            
            # Run benchmark for this top_k value
            for _ in range(5):
                test_runner.time_operation(rag.query, base_query, top_k=top_k)
            
            report = metrics.generate_report()
            timing_stats = report["timing_stats"]
            
            print(f"\nTop-K = {top_k}:")
            print(f"  Mean response time: {timing_stats['mean']:.3f}s")
            print(f"  Response time std dev: {timing_stats['std_dev']:.3f}s")
            
            # Performance should scale reasonably with top_k
            expected_max_time = 0.05 + (top_k * 0.005)  # Base + linear scaling
            assert timing_stats['mean'] < expected_max_time


@pytest.mark.slow
@pytest.mark.benchmark
class TestSystemScalabilityBenchmarks:
    """Benchmarks for system scalability and resource usage."""
    
    def test_memory_usage_scaling(self, mock_config, temp_dir):
        """Test memory usage as system scale increases."""
        document_counts = [10, 50, 100, 200]
        memory_results = []
        
        for doc_count in document_counts:
            # Create fresh RAG system for each test
            rag = RAGSystem()
            rag.config = mock_config
            
            # Mock agents
            rag.document_parser = Mock()
            rag.chunking_embedding = Mock()
            
            # Create mock documents in memory
            mock_documents = []
            for i in range(doc_count):
                mock_doc = Document(
                    filename=f"doc_{i}.txt",
                    file_path=f"/path/to/doc_{i}.txt",
                    content=f"Document {i} content " * 100,  # ~2KB per doc
                    document_type="text",
                    processing_status=ProcessingStatus.PROCESSED
                )
                mock_documents.append(mock_doc)
            
            rag.document_parser.parse_document.side_effect = mock_documents
            rag.chunking_embedding.process_document.return_value = ["chunk1", "chunk2", "chunk3"]
            
            # Measure memory before and after processing
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process documents
            for i in range(doc_count):
                rag.add_document(f"/path/to/doc_{i}.txt")
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            memory_results.append({
                "document_count": doc_count,
                "memory_increase_mb": memory_increase,
                "memory_per_doc_kb": (memory_increase * 1024) / doc_count if doc_count > 0 else 0
            })
            
            print(f"\nMemory Usage - {doc_count} documents:")
            print(f"  Memory increase: {memory_increase:.1f}MB")
            print(f"  Memory per document: {(memory_increase * 1024) / doc_count:.1f}KB")
            
            # Memory usage should be reasonable
            assert memory_increase < doc_count * 0.1, f"Memory usage too high: {memory_increase}MB for {doc_count} docs"
            
            # Clean up
            del rag
        
        # Analyze scaling trend
        print(f"\nMemory Scaling Analysis:")
        for result in memory_results:
            print(f"  {result['document_count']} docs: {result['memory_increase_mb']:.1f}MB total, {result['memory_per_doc_kb']:.1f}KB/doc")
    
    def test_response_time_under_load(self, query_benchmark_rag_system):
        """Test response time degradation under increasing load."""
        rag = query_benchmark_rag_system
        
        # Simulate increasing load
        load_levels = [1, 5, 10, 25, 50]  # Concurrent queries
        
        for load_level in load_levels:
            queries = [f"Query under load test {i}" for i in range(load_level)]
            
            response_times = []
            start_time = time.time()
            
            def process_single_query(query):
                query_start = time.time()
                result = rag.query(query)
                query_time = time.time() - query_start
                return query_time
            
            # Process queries concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=load_level) as executor:
                response_times = list(executor.map(process_single_query, queries))
            
            total_time = time.time() - start_time
            
            # Calculate metrics
            avg_response_time = statistics.mean(response_times)
            p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]
            throughput = len(queries) / total_time
            
            print(f"\nLoad Level {load_level} concurrent queries:")
            print(f"  Average response time: {avg_response_time:.3f}s")
            print(f"  P95 response time: {p95_response_time:.3f}s")
            print(f"  Throughput: {throughput:.1f} queries/sec")
            
            # Performance should degrade gracefully
            max_acceptable_response_time = 0.1 + (load_level * 0.01)  # Allow some degradation
            assert avg_response_time < max_acceptable_response_time, \
                f"Response time degraded too much under load {load_level}: {avg_response_time:.3f}s"


def save_benchmark_results(results: Dict[str, Any], output_file: str = "benchmark_results.json"):
    """Save benchmark results to file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nBenchmark results saved to {output_file}")


@pytest.mark.benchmark
def test_comprehensive_performance_suite(mock_config, temp_dir):
    """Run comprehensive performance test suite."""
    print("\n" + "="*60)
    print("RAG AGENT SYSTEM PERFORMANCE BENCHMARK SUITE")
    print("="*60)
    
    # This would be the main entry point for running all benchmarks
    # Individual benchmark classes above would be called from here
    
    results = {
        "benchmark_timestamp": time.time(),
        "system_info": {
            "python_version": os.sys.version,
            "cpu_count": os.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3)
        },
        "test_summary": {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0
        }
    }
    
    print(f"System: {results['system_info']['cpu_count']} CPUs, {results['system_info']['memory_gb']:.1f}GB RAM")
    print("Starting performance benchmarks...")
    
    # Save results
    save_benchmark_results(results, Path(temp_dir) / "benchmark_results.json")
    
    print("\nPerformance benchmark suite completed!")
    print("="*60)