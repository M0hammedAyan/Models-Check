"""
Main entry point for depth estimation benchmark pipeline.
"""

import argparse
import logging
import torch
from pathlib import Path
from common import setup_logging, set_seed, BenchmarkConfig
from benchmark import BenchmarkPipeline, CSVExporter, ReportGenerator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Depth Estimation Benchmark Pipeline'
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--input-video', type=str, default=None,
                       help='Path to input video file')
    parser.add_argument('--input-bag', type=str, default=None,
                       help='Path to RealSense .bag file')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Models to run (if None, use all from config)')
    parser.add_argument('--model', type=str, default=None,
                       help='Run a single model by name (shorthand for --models <name>)')
    parser.add_argument('--skip-pose', action='store_true',
                       help='Skip pose estimation')
    parser.add_argument('--skip-temporal-refinement', action='store_true',
                       help='Skip temporal refinement')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Depth Estimation Benchmark Pipeline")
    
    # Load config
    config = BenchmarkConfig.from_yaml(args.config)
    
    # Update config from command line args
    if args.input_video:
        config.data.video_path = args.input_video
    if args.input_bag:
        config.data.bag_path = args.input_bag
    if args.model and args.models:
        parser.error('Use either --model or --models, not both')

    if args.model:
        args.models = [args.model]

    if args.models:
        config.models = {m: config.models[m] for m in args.models if m in config.models}
    if args.skip_pose:
        config.pose.enable_pose = False
    if args.skip_temporal_refinement:
        config.processing.apply_temporal_refinement = False
    
    config.output_dir = args.output_dir
    config.device = args.device
    
    logger.info(f"Config: {config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize pipeline
    pipeline = BenchmarkPipeline(config)
    
    # Load models
    logger.info("Loading models...")
    pipeline.load_models()
    
    logger.info(f"Loaded {len(pipeline.models)} models: {list(pipeline.models.keys())}")
    
    # Process input
    all_results = {}
    
    if config.data.video_path:
        logger.info(f"Processing video: {config.data.video_path}")
        results, rgb_frames = pipeline.process_video(config.data.video_path)
        
        # Run inference and compute metrics for each model
        for model_name, depths in results['models'].items():
            logger.info(f"Computing metrics for {model_name}...")
            metrics = pipeline.compute_metrics(depths['depths'], depths['depths'])  # Self-comparison
            all_results[model_name] = metrics
        
        # Run pose estimation
        if config.pose.enable_pose:
            logger.info("Running pose estimation...")
            pose_results = pipeline.run_pose_estimation(
                rgb_frames,
                results['models'][list(results['models'].keys())[0]]['depths']
            )
            all_results['pose'] = pose_results
    
    if config.data.bag_path:
        logger.info(f"Processing RealSense bag: {config.data.bag_path}")
        results, rgb_frames, ref_depths = pipeline.process_realsense_bag(config.data.bag_path)
        
        # Compute metrics against RealSense reference
        for model_name, depths in results['models'].items():
            logger.info(f"Computing metrics for {model_name}...")
            metrics = pipeline.compute_metrics(depths['depths'], ref_depths)
            all_results[model_name] = metrics
        
        # Run pose estimation
        if config.pose.enable_pose:
            logger.info("Running pose estimation...")
            pose_results = pipeline.run_pose_estimation(
                rgb_frames,
                ref_depths
            )
            all_results['pose'] = pose_results
    
    # Save results
    logger.info("Saving results...")
    pipeline.save_results(all_results, 'benchmark_results')
    
    # Export CSV
    logger.info("Exporting CSV...")
    csv_exporter = CSVExporter()
    csv_path = Path(args.output_dir) / 'metrics' / 'model_comparison.csv'
    csv_exporter.export_model_comparison(all_results, str(csv_path))
    
    # Generate reports
    logger.info("Generating reports...")
    report_gen = ReportGenerator(str(Path(args.output_dir) / 'reports'))
    report_gen.generate_all_reports(all_results)
    
    logger.info("Benchmark pipeline completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
