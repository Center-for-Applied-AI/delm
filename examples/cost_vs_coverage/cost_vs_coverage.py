"""
Cost vs Coverage Analysis Example

This example demonstrates how to analyze the trade-off between cost and coverage
for LLM-based data extraction using keyword filtering.

The approach:
1. Load commodity data and split into train/test (80/20)
2. Discover keywords on training data using TF-IDF
3. Run baseline experiment (no filtering) on test data
4. Run keyword experiments with different keyword sizes on test data
5. Plot Pareto curve showing coverage vs cost trade-offs
6. Save top 100 keywords in TF-IDF order
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from delm import DELM, DELMConfig
from delm.core.data_processor import DataProcessor
from delm.strategies import KeywordScorer
from delm.utils.cost_estimation import estimate_input_token_cost


class CostVsCoverageAnalyzer:
    """Analyzer for cost vs coverage trade-offs in keyword filtering."""
    
    def __init__(self, data_path: Path, config_path: Path, schema_path: Path):
        """Initialize the analyzer.
        
        Args:
            data_path: Path to the commodity data CSV file
            config_path: Path to the DELM configuration file
            schema_path: Path to the schema specification file
        """
        self.data_path = data_path
        self.config_path = config_path
        self.schema_path = schema_path
        self.commodity_data_df = None
        self.train_df = None
        self.test_df = None
        self.config = DELMConfig.from_yaml(self.config_path)
        self.baseline_results = {}
        self.top_keywords = []
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the commodity data with train/test split."""
        print("Loading commodity data...")
        self.commodity_data_df = pd.read_csv(self.data_path)
        
        # Filter out rows with NaN text
        self.commodity_data_df = self.commodity_data_df.dropna(subset=["text"])
        
        print(f"Loaded {len(self.commodity_data_df)} total records")
        
        # Split into train/test (80/20) on ALL data first
        self.train_df, self.test_df = train_test_split(
            self.commodity_data_df, 
            test_size=0.2, 
            random_state=42
        )
        
        print(f"Train set: {len(self.train_df)} records")
        print(f"Test set: {len(self.test_df)} records")
        
        return self.commodity_data_df
    
    def discover_keywords(self, top_n: int = 100) -> List[str]:
        """Discover keywords using TF-IDF on training data."""
        print(f"Discovering top {top_n} keywords using TF-IDF on training data...")
        
        # Prepare data for keyword discovery on training data
        texts_with_extractions = []
        texts_without_extractions = []
        
        for _, row in self.train_df.iterrows():
            if pd.notna(row.get('good')) or pd.notna(row.get('price_expectation')):
                texts_with_extractions.append(row['text'])
            else:
                texts_without_extractions.append(row['text'])
        
        print(f"Texts with extractions: {len(texts_with_extractions)}")
        print(f"Texts without extractions: {len(texts_without_extractions)}")
        
        # Combine texts and create labels
        all_texts = texts_with_extractions + texts_without_extractions
        labels = [1] * len(texts_with_extractions) + [0] * len(texts_without_extractions)
        
        # Create TF-IDF features
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 1))
        X = vectorizer.fit_transform(all_texts)
        
        # Select best features - analyze more words to get more commodity-related ones
        selector = SelectKBest(chi2, k=min(500, len(vectorizer.get_feature_names_out())))
        X_selected = selector.fit_transform(X, labels)
        
        # Get selected feature names
        feature_names = vectorizer.get_feature_names_out()
        selected_features = feature_names[selector.get_support()]
        
        # Get feature scores for ordering
        feature_scores = selector.scores_[selector.get_support()]
        
        # Calculate which class each word is more associated with
        X_dense = X.toarray()
        word_class_associations = []
        
        for i, feature_idx in enumerate(np.where(selector.get_support())[0]):
            word = selected_features[i]
            score = feature_scores[i]
            
            # Calculate mean TF-IDF score for each class
            class_1_scores = X_dense[:len(texts_with_extractions), feature_idx]
            class_0_scores = X_dense[len(texts_with_extractions):, feature_idx]
            
            mean_class_1 = np.mean(class_1_scores)
            mean_class_0 = np.mean(class_0_scores)
            
            # Determine which class this word is more associated with
            if mean_class_1 > mean_class_0:
                association = "WITH extractions"
            else:
                association = "WITHOUT extractions"
            
            word_class_associations.append((word, score, association, mean_class_1, mean_class_0))
        
        # Sort by chi2 score (descending) but only keep words associated with texts WITH extractions
        word_class_associations.sort(key=lambda x: x[1], reverse=True)
        
        # Debug: Show top 100 most distinctive words regardless of class
        print(f"\nTop 100 most distinctive words (regardless of class):")
        for i, (word, score, association, mean_1, mean_0) in enumerate(word_class_associations[:100]):
            print(f"{i+1:3d}. {word:15s} - {association:20s} (score: {score:.2f})")
        
        # Filter to only include words more common in texts WITH extractions
        keywords_with_extractions = [word for word, score, association, _, _ in word_class_associations 
                                   if association == "WITH extractions"]
        
        # Take top N keywords that are associated with extractions
        self.top_keywords = keywords_with_extractions[:top_n]
        
        print(f"Discovered {len(self.top_keywords)} keywords associated with texts containing extractions")
        print(f"Top 10 keywords: {self.top_keywords[:10]}")
        
        # Debug: Show some examples of word associations
        print("\nSample word associations:")
        for i, (word, score, association, mean_1, mean_0) in enumerate(word_class_associations[:20]):
            print(f"{i+1:2d}. {word:15s} - {association:20s} (score: {score:.2f}, class1: {mean_1:.4f}, class0: {mean_0:.4f})")
        
        # Save keywords to file
        with open("top_250_keywords.txt", "w") as f:
            for i, keyword in enumerate(self.top_keywords, 1):
                f.write(f"{i:3d}. {keyword}\n")
        print("Keywords saved to top_250_keywords.txt")
        
        return self.top_keywords
    
    def run_baseline_experiment(self, sample_size: int = 200, dataset: str = "test") -> Dict[str, Any]:
        """Run baseline experiment (no filtering) on specified dataset."""
        print(f"Running baseline experiment (no filtering) on {dataset} data...")
        
        # Sample data from specified dataset
        if dataset == "test":
            sample_df = self.test_df.sample(n=min(sample_size, len(self.test_df)), random_state=42)
        else:  # train
            sample_df = self.train_df.sample(n=min(sample_size, len(self.train_df)), random_state=42)
        
        # Break text into paragraphs using the splitting strategy
        paragraphs, labels = self._split_text_into_paragraphs(sample_df, self.config)
        
        # Calculate metrics
        total_paragraphs = len(paragraphs)
        paragraphs_with_extractions = self._count_expectation_paragraphs(labels)
        
        # Calculate actual cost for all paragraphs
        from delm.utils.cost_estimation import estimate_input_token_cost
        # Create a DataFrame with all paragraphs
        all_df = self._create_dataframe_with_selected_paragraphs(sample_df, paragraphs, self.config)
        baseline_cost = estimate_input_token_cost(self.config, all_df)
        
        metrics = {
            "total_paragraphs": total_paragraphs,
            "paragraphs_with_extractions": paragraphs_with_extractions,
            "paragraphs_processed": total_paragraphs,
            "selected_with_extractions": paragraphs_with_extractions,
            "coverage": 1.0,  # 100% coverage
            "cost_savings": 0.0,  # No cost savings
            "filtered_cost": baseline_cost,
            "keywords": []
        }
        
        if dataset == "test":
            self.baseline_results = metrics
        else:
            self.train_baseline_results = metrics
            
        print(f"Baseline ({dataset}): {total_paragraphs} paragraphs, {paragraphs_with_extractions} with extractions, cost=${baseline_cost:.2f}")
        
        return metrics
    
    def run_keyword_experiment(self, keywords: List[str], sample_size: int = 200, dataset: str = "test") -> Dict[str, Any]:
        """Run experiment with specific keywords on specified dataset."""
        print(f"Running keyword experiment with {len(keywords)} keywords on {dataset} data...")
        
        # Sample data from specified dataset
        if dataset == "test":
            sample_df = self.test_df.sample(n=min(sample_size, len(self.test_df)), random_state=42)
        else:  # train
            sample_df = self.train_df.sample(n=min(sample_size, len(self.train_df)), random_state=42)
        
        # Break text into paragraphs using the splitting strategy
        paragraphs, labels = self._split_text_into_paragraphs(sample_df, self.config)
        
        # Test keyword scorer on each paragraph
        keyword_scorer = KeywordScorer(keywords)
        selected_paragraphs = []
        selected_labels = []
        
        for paragraph, label in zip(paragraphs, labels):
            score = keyword_scorer.score(paragraph)
            if score > 0:  # Paragraph contains keywords
                selected_paragraphs.append(paragraph)
                selected_labels.append(label)
        
        # Calculate metrics
        total_paragraphs = len(paragraphs)
        paragraphs_processed = len(selected_paragraphs)
        total_expectation = self._count_expectation_paragraphs(labels)
        selected_with_extractions = self._count_expectation_paragraphs(selected_labels)
        
        # Calculate coverage
        coverage = selected_with_extractions / total_expectation if total_expectation > 0 else 0
        
        # Calculate actual cost for selected paragraphs
        from delm.utils.cost_estimation import estimate_input_token_cost
        # Create a DataFrame with only the selected paragraphs
        selected_df = self._create_dataframe_with_selected_paragraphs(sample_df, selected_paragraphs, self.config)
        filtered_cost = estimate_input_token_cost(self.config, selected_df)
        
        # Calculate cost savings relative to baseline
        if dataset == "test":
            baseline_cost = self.baseline_results["filtered_cost"]
        else:
            baseline_cost = self.train_baseline_results["filtered_cost"]
        cost_savings = (baseline_cost - filtered_cost) / baseline_cost if baseline_cost > 0 else 0
        
        metrics = {
            "total_paragraphs": total_paragraphs,
            "paragraphs_with_extractions": total_expectation,
            "paragraphs_processed": paragraphs_processed,
            "selected_with_extractions": selected_with_extractions,
            "coverage": coverage,
            "cost_savings": cost_savings,
            "filtered_cost": filtered_cost,
            "keywords": keywords,
            "dataset": dataset
        }
        
        return metrics
    
    def _split_text_into_paragraphs(self, df: pd.DataFrame, config: DELMConfig) -> tuple[list[str], list[bool]]:
        """Split text into paragraphs using the configured splitting strategy."""
        from delm.core.data_processor import DataProcessor
        
        # Create a temporary data processor to use the splitting strategy
        data_processor = DataProcessor(config.data_preprocessing)
        
        # Process the data to get chunks
        processed_df = data_processor.process_dataframe(df)
        
        # Extract the chunked text and create expectation labels
        processed_df["has_expectation"] = (
            processed_df["price_expectation"].notna() | processed_df["good"].notna()
        )
        paragraphs = processed_df["delm_text_chunk"].dropna().tolist()
        labels = processed_df["has_expectation"].tolist()
        return paragraphs, labels
    
    @staticmethod
    def _count_expectation_paragraphs(labels: list[bool]) -> int:
        """Count how many paragraphs contain actual price expectations."""
        return sum(labels)
    
    def _create_dataframe_with_selected_paragraphs(self, original_df: pd.DataFrame, selected_paragraphs: List[str], config: DELMConfig) -> pd.DataFrame:
        """Create a DataFrame containing only the selected paragraphs for cost estimation."""
        # Create a new DataFrame with only the selected paragraphs
        # We need to reconstruct the text column with only the selected paragraphs
        
        # Get the target column name from config
        target_column = config.data_preprocessing.target_column
        
        # Create a new DataFrame with the selected paragraphs as the text
        selected_df = pd.DataFrame({
            target_column: selected_paragraphs
        })
        
        # Add any other columns that might be needed (like record_id)
        if "record_id" in original_df.columns:
            selected_df["record_id"] = range(len(selected_df))
        
        return selected_df
    
    def run_pareto_analysis(self, keyword_sizes: List[int] = None, sample_size: int = 200) -> pd.DataFrame:
        """Run Pareto analysis with different keyword sizes on both train and test data."""
        if keyword_sizes is None:
            keyword_sizes = list(range(1, 251))  # 1 to 250 keywords with step=1
        
        print(f"Running Pareto analysis with keyword sizes: {keyword_sizes}")
        
        results = []
        
        # Run experiments for both train and test datasets
        for dataset in ["train", "test"]:
            # Add baseline result
            baseline_result = self.run_baseline_experiment(sample_size, dataset)
            baseline_result["keyword_size"] = 0
            baseline_result["dataset"] = dataset
            results.append(baseline_result)
            
            # Run experiments for each keyword size
            for keyword_size in keyword_sizes:
                if keyword_size <= len(self.top_keywords):
                    keywords = self.top_keywords[:keyword_size]
                    result = self.run_keyword_experiment(keywords, sample_size, dataset)
                    result["keyword_size"] = keyword_size
                    results.append(result)
                    
                    print(f"Keywords {keyword_size} ({dataset}): Coverage={result['coverage']:.3f}, Cost=${result['filtered_cost']:.2f}")
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def plot_results(self, results_df: pd.DataFrame, save_path: Path = None):
        """Plot the Pareto curves for both train and test data."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Separate train and test data
        train_df = results_df[(results_df["keyword_size"] > 0) & (results_df["dataset"] == "train")]
        test_df = results_df[(results_df["keyword_size"] > 0) & (results_df["dataset"] == "test")]
        
        # Get baseline costs for each dataset
        train_baseline = results_df[(results_df["keyword_size"] == 0) & (results_df["dataset"] == "train")]["filtered_cost"].iloc[0]
        test_baseline = results_df[(results_df["keyword_size"] == 0) & (results_df["dataset"] == "test")]["filtered_cost"].iloc[0]
        
        # Calculate cost percentages
        train_cost_pct = (train_df["filtered_cost"] / train_baseline) * 100
        test_cost_pct = (test_df["filtered_cost"] / test_baseline) * 100
        
        # Plot both curves with different colors
        ax.plot(train_cost_pct, train_df["coverage"] * 100, 
                marker="o", linewidth=2.5, markersize=6, 
                color="#2E86AB", alpha=0.8, markeredgecolor="white", markeredgewidth=1,
                label="Train")
        
        ax.plot(test_cost_pct, test_df["coverage"] * 100, 
                marker="s", linewidth=2.5, markersize=6, 
                color="#A23B72", alpha=0.8, markeredgecolor="white", markeredgewidth=1,
                label="Test")
        
        # Add annotations for key sizes
        key_sizes = [5, 10, 20, 100, 200, 250]
        for size in key_sizes:
            # Train annotation
            if size in train_df["keyword_size"].values:
                row = train_df[train_df["keyword_size"] == size].iloc[0]
                cost_pct = (row["filtered_cost"] / train_baseline) * 100
                ax.annotate(f"k={size}", 
                            (cost_pct, row["coverage"] * 100), 
                            xytext=(10, 10), textcoords="offset points", 
                            fontsize=9, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2E86AB", alpha=0.8, edgecolor="white"))
            
            # Test annotation
            if size in test_df["keyword_size"].values:
                row = test_df[test_df["keyword_size"] == size].iloc[0]
                cost_pct = (row["filtered_cost"] / test_baseline) * 100
                ax.annotate(f"k={size}", 
                            (cost_pct, row["coverage"] * 100), 
                            xytext=(-10, -15), textcoords="offset points", 
                            fontsize=9, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="#A23B72", alpha=0.8, edgecolor="white"))
        
        ax.set_xlabel("Cost (% of Total)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Coverage (%)", fontsize=14, fontweight="bold")
        ax.set_title("Train vs Test Pareto Curves: Coverage vs Cost for Keyword-Based Filtering", 
                    fontsize=16, fontweight="bold", pad=20)
        
        ax.legend(fontsize=12, loc="lower right")
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        ax.tick_params(axis="both", which="major", labelsize=12)
        
        plt.tight_layout()
        plt.savefig("cost_vs_coverage_results.pdf", dpi=300, bbox_inches="tight", format="pdf")
        plt.savefig("cost_vs_coverage_results.svg", dpi=300, bbox_inches="tight", format="svg")
        plt.close()
        
        print("Results saved to cost_vs_coverage_results.pdf and cost_vs_coverage_results.svg")
    
    def save_results(self, results_df: pd.DataFrame, output_path: Path):
        """Save the results to CSV."""
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


def main():
    """Main function to run the cost vs coverage analysis."""
    # Setup paths
    data_path = Path("../commodity_data_large.csv")
    config_path = Path("config.yaml")
    schema_path = Path("../commodity_schema.yaml")
    
    # Create analyzer
    analyzer = CostVsCoverageAnalyzer(data_path, config_path, schema_path)
    
    # Load data with train/test split
    analyzer.load_data()
    
    # Discover keywords on training data
    analyzer.discover_keywords(top_n=250)
    
    # Run Pareto analysis on test data
    results_df = analyzer.run_pareto_analysis(keyword_sizes=list(range(1, 251)))
    
    # Plot results
    analyzer.plot_results(results_df)
    
    # Save results
    analyzer.save_results(results_df, Path("cost_vs_coverage_results.csv"))
    
    print("\nAnalysis complete! Check the generated files:")
    print("- cost_vs_coverage_results.csv: Numerical results")
    print("- cost_vs_coverage_results.pdf/.svg: Pareto curve visualization")
    print("- top_250_keywords.txt: Top 250 keywords in TF-IDF order")


if __name__ == "__main__":
    main()
