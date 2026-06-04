#!/usr/bin/env python3
"""
Example benchmark run script.

Usage:
    python example_run.py --input-video test_data/Ncam/video.mp4
    python example_run.py --input-bag test_data/Realsense/recording.bag
    python example_run.py --benchmark-all
"""

import argparse
import logging
from pathlib import Path
from common import setup_logging, set_seed, BenchmarkConfig
from benchmark import BenchmarkPipeline, CSVExporter, ReportGenerator, ModelComparator


def run_single_video_benchmark(video_path: str, output_dir: str = './outputs'):
    """Run benchmark on single video."""
    setup_logging(log_level='INFO')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("DEPTH ESTIMATION BENCHMARK - SINGLE VIDEO")
    logger.info("=" * 70)
    
    # Load config
    config = BenchmarkConfig.from_yaml('config.yaml')
    config.data.video_path = video_path
    config.output_dir = output_dir
    
    # Initialize pipeline
    pipeline = BenchmarkPipeline(config)
    pipeline.load_models()
    
    logger.info(f"Processing: {video_path}")
    logger.info(f"Models: {list(pipeline.models.keys())}")
    logger.info(f"Output: {output_dir}")
    
    # Process video
    results, rgb_frames = pipeline.process_video(video_path)
    
    # Compute metrics
    all_results = {}
    for model_name, depths in results['models'].items():
        logger.info(f"Computing metrics for {model_name}...")
        metrics = pipeline.compute_metrics(depths['depths'], depths['depths'])
        all_results[model_name] = metrics
    
    # Run pose estimation if enabled
    if config.pose.enable_pose:
        logger.info("Running pose estimation...")
        first_model = list(results['models'].keys())[0]
        pose_results = pipeline.run_pose_estimation(
            rgb_frames,
            results['models'][first_model]['depths']
        )
        all_results['pose'] = pose_results
    
    # Save results
    logger.info("Saving results...")
    pipeline.save_results(all_results, 'benchmark_results')
    
    # Export CSV
    csv_path = Path(output_dir) / 'metrics' / 'model_comparison.csv'
    CSVExporter.export_model_comparison(all_results, str(csv_path))
    
    # Generate reports
    logger.info("Generating reports...")
    report_gen = ReportGenerator(str(Path(output_dir) / 'reports'))
    report_gen.generate_all_reports(all_results)
    
    logger.info("=" * 70)
    logger.info("✓ Benchmark completed!")
    logger.info(f"Results: {output_dir}")
    logger.info("=" * 70)


def run_realsense_benchmark(bag_path: str, output_dir: str = './outputs'):
    """Run benchmark on RealSense .bag file."""
    setup_logging(log_level='INFO')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("DEPTH ESTIMATION BENCHMARK - REALSENSE .BAG")
    logger.info("=" * 70)
    
    # Load config
    config = BenchmarkConfig.from_yaml('config.yaml')
    config.data.bag_path = bag_path
    config.output_dir = output_dir
    
    # Initialize pipeline
    pipeline = BenchmarkPipeline(config)
    pipeline.load_models()
    
    logger.info(f"Processing: {bag_path}")
    logger.info(f"Models: {list(pipeline.models.keys())}")
    logger.info(f"Output: {output_dir}")
    
    # Process RealSense .bag
    results, rgb_frames, ref_depths = pipeline.process_realsense_bag(bag_path)
    
    # Compute metrics against RealSense reference
    all_results = {}
    for model_name, depths in results['models'].items():
        logger.info(f"Computing metrics for {model_name}...")
        metrics = pipeline.compute_metrics(depths['depths'], ref_depths)
        all_results[model_name] = metrics
    
    # Run pose estimation if enabled
    if config.pose.enable_pose:
        logger.info("Running pose estimation...")
        pose_results = pipeline.run_pose_estimation(rgb_frames, ref_depths)
        all_results['pose'] = pose_results
    
    # Save results
    logger.info("Saving results...")
    pipeline.save_results(all_results, 'benchmark_results')
    
    # Export CSV
    csv_path = Path(output_dir) / 'metrics' / 'model_comparison.csv'
    CSVExporter.export_model_comparison(all_results, str(csv_path))
    
    # Generate reports
    logger.info("Generating reports...")
    report_gen = ReportGenerator(str(Path(output_dir) / 'reports'))
    report_gen.generate_all_reports(all_results)
    
    logger.info("=" * 70)
    logger.info("✓ Benchmark completed!")
    logger.info(f"Results: {output_dir}")
    logger.info("=" * 70)


def run_comparative_benchmark(output_dir: str = './outputs'):
    """Run comparative benchmark on all test data."""
    setup_logging(log_level='INFO')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("DEPTH ESTIMATION BENCHMARK - COMPARATIVE")
    logger.info("=" * 70)
    
    test_data_dir = Path('../test_data')
    
    if not test_data_dir.exists():
        logger.error(f"Test data directory not found: {test_data_dir}")
        logger.error("Please create test_data/ folder with Ncam/ and Realsense/ subfolders")
        return
    
    # Find test files
    ncam_videos = list((test_data_dir / 'Ncam').glob('*.mp4')) if (test_data_dir / 'Ncam').exists() else []
    realsense_bags = list((test_data_dir / 'Realsense').glob('*.bag')) if (test_data_dir / 'Realsense').exists() else []
    
    logger.info(f"Found {len(ncam_videos)} Ncam videos")
    logger.info(f"Found {len(realsense_bags)} RealSense .bag files")
    
    # Process each file
    all_results = {}
    
    for video_path in ncam_videos:
        logger.info(f"\nProcessing: {video_path.name}")
        output_subdir = Path(output_dir) / video_path.stem
        run_single_video_benchmark(str(video_path), str(output_subdir))
    
    for bag_path in realsense_bags:
        logger.info(f"\nProcessing: {bag_path.name}")
        output_subdir = Path(output_dir) / bag_path.stem
        run_realsense_benchmark(str(bag_path), str(output_subdir))
    
    logger.info("=" * 70)
    logger.info("✓ All benchmarks completed!")
    logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Depth Estimation Benchmark - Example Run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on single video
  python example_run.py --input-video test_data/Ncam/video.mp4
  
  # Run on RealSense .bag
  python example_run.py --input-bag test_data/Realsense/recording.bag
  
  # Comparative benchmark on all files
  python example_run.py --benchmark-all
  
  # Custom output directory
  python example_run.py --input-video video.mp4 --output outputs/custom
        """
    )
    
    parser.add_argument('--input-video', type=str, default=None,
                       help='Path to input video file')
    parser.add_argument('--input-bag', type=str, default=None,
                       help='Path to RealSense .bag file')
    parser.add_argument('--output', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--benchmark-all', action='store_true',
                       help='Run benchmark on all test data')
    
    args = parser.parse_args()
    
    try:
        if args.benchmark_all:
            run_comparative_benchmark(args.output)
        elif args.input_video:
            run_single_video_benchmark(args.input_video, args.output)
        elif args.input_bag:
            run_realsense_benchmark(args.input_bag, args.output)
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\n⚠ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        raise


if __name__ == '__main__':
    main()
