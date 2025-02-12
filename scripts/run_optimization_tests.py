import sys
from pathlib import Path
import logging
import asyncio
import json
from datetime import datetime
import pandas as pd
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizationTestRunner:
    def __init__(self):
        self.results_dir = project_root / "tests" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    async def run_chunking_tests(self) -> str:
        """Run chunking tests and return results file path"""
        logger.info("\n=== Running Chunking Tests ===")
        from scripts.test_chunking import main as chunking_main
        await chunking_main()
        # Return most recent results file
        return max(self.results_dir.glob("chunking_test_results_*.json"))
        
    async def run_retrieval_tests(self) -> str:
        """Run retrieval tests and return results file path"""
        logger.info("\n=== Running Retrieval Tests ===")
        from scripts.test_retrieval import main as retrieval_main
        await retrieval_main()
        return max(self.results_dir.glob("retrieval_test_results_*.json"))
        
    async def run_token_tests(self) -> str:
        """Run token capacity tests and return results file path"""
        logger.info("\n=== Running Token Capacity Tests ===")
        from scripts.test_token_capacity import main as token_main
        await token_main()
        return max(self.results_dir.glob("token_capacity_results_*.json"))
        
    def generate_enhanced_report(self, chunking_results: Dict, retrieval_results: Dict, token_results: Dict) -> str:
        """Generate comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"optimization_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Optimization Test Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Chunking Analysis
            f.write("## 1. Chunking Performance\n\n")
            successful = [r for r in chunking_results["results"] if r.get("success", False)]
            failed = [r for r in chunking_results["results"] if not r.get("success", False)]
            
            if successful:
                avg_time = sum(r["processing_time"] for r in successful) / len(successful)
                avg_chunks = sum(r["num_chunks"] for r in successful) / len(successful)
                complete_notes = sum(r.get("complete_notes", 0) for r in successful)
                chunked_notes = sum(r.get("chunked_notes", 0) for r in successful)
                
                f.write("### Overall Statistics\n")
                f.write(f"- Successfully processed files: {len(successful)}\n")
                f.write(f"- Failed files: {len(failed)}\n")
                f.write(f"- Average processing time: {avg_time:.2f} seconds\n")
                f.write(f"- Average chunks per file: {avg_chunks:.2f}\n")
                f.write(f"- Complete notes preserved: {complete_notes}\n")
                f.write(f"- Notes requiring chunking: {chunked_notes}\n\n")
            
            # Retrieval Analysis
            f.write("## 2. Retrieval Performance\n\n")
            for top_k_config, query_results in retrieval_results["results"].items():
                successful = [r for r in query_results if "error" not in r["metrics"]]
                
                if successful:
                    avg_time = sum(r["metrics"]["retrieval_time"] for r in successful) / len(successful)
                    avg_coherence = sum(r["metrics"]["coherence_score"] for r in successful) / len(successful)
                    avg_complete = sum(r["metrics"]["complete_notes"] for r in successful) / len(successful)
                    avg_chunked = sum(r["metrics"]["chunked_notes"] for r in successful) / len(successful)
                    
                    f.write(f"\n### {top_k_config}\n")
                    f.write(f"- Average retrieval time: {avg_time:.3f} seconds\n")
                    f.write(f"- Average coherence score: {avg_coherence:.3f}\n")
                    f.write(f"- Average complete notes: {avg_complete:.1f}\n")
                    f.write(f"- Average chunked notes: {avg_chunked:.1f}\n")
                    f.write(f"- Successful queries: {len(successful)}\n")
            
            # Token Capacity Analysis
            f.write("\n## 3. Token Capacity Analysis\n\n")
            for token_config, result in token_results["results"].items():
                if "error" not in result:
                    validation = result["validation"]
                    token_analysis = validation["token_analysis"]
                    
                    f.write(f"\n### {token_config}\n")
                    f.write(f"- Total tokens: {token_analysis['total_tokens']}\n")
                    f.write(f"- Window utilization: {validation['window_utilization']:.2%}\n")
                    f.write(f"- Complete notes: {token_analysis['complete_notes']}\n")
                    f.write(f"- Chunked notes: {token_analysis['chunked_notes']}\n")
                    f.write(f"- Average tokens per chunk: {token_analysis['avg_tokens_per_chunk']:.1f}\n")
            
            # Recommendations
            f.write("\n## 4. Recommendations\n\n")
            
            # Analyze chunking performance
            if successful:
                f.write("### Content Processing\n")
                f.write(f"- Processing success rate: {len(successful)/(len(successful) + len(failed)):.1%}\n")
                f.write(f"- Complete note ratio: {complete_notes/(complete_notes + chunked_notes):.1%}\n")
                f.write(f"- Average processing time per file: {avg_time:.2f} seconds\n\n")
            
            # Analyze retrieval performance
            f.write("### Retrieval Configuration\n")
            best_coherence = 0
            best_config = None
            for top_k_config, query_results in retrieval_results["results"].items():
                successful = [r for r in query_results if "error" not in r["metrics"]]
                if successful:
                    avg_coherence = sum(r["metrics"]["coherence_score"] for r in successful) / len(successful)
                    if avg_coherence > best_coherence:
                        best_coherence = avg_coherence
                        best_config = top_k_config
            
            if best_config:
                f.write(f"- Recommended configuration: {best_config}\n")
                f.write(f"- Best coherence score: {best_coherence:.3f}\n\n")
            
            # Token capacity recommendations
            f.write("### Token Window Utilization\n")
            for token_config, result in token_results["results"].items():
                if "error" not in result:
                    utilization = result["validation"]["window_utilization"]
                    f.write(f"- {token_config}: {utilization:.1%} window utilization\n")
            
            f.write("\n### Implementation Notes\n")
            f.write("1. Preserve complete notes when possible for better coherence\n")
            f.write("2. Use chunking with overlap for longer content\n")
            f.write("3. Monitor token window utilization to prevent context overflow\n")
            f.write("4. Consider content type when selecting chunk size\n")
            
        return report_file

async def main():
    """Run all optimization tests and generate report"""
    try:
        runner = OptimizationTestRunner()
        
        # Run all tests
        chunking_file = await runner.run_chunking_tests()
        retrieval_file = await runner.run_retrieval_tests()
        token_file = await runner.run_token_tests()
        
        # Load and analyze results
        with open(chunking_file) as f:
            chunking_results = json.load(f)
        with open(retrieval_file) as f:
            retrieval_results = json.load(f)
        with open(token_file) as f:
            token_results = json.load(f)
        
        # Generate enhanced report
        report_file = runner.generate_enhanced_report(
            chunking_results,
            retrieval_results,
            token_results
        )
        
        logger.info(f"\nEnhanced optimization report generated: {report_file}")
        
    except Exception as e:
        logger.error(f"Error running optimization tests: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
