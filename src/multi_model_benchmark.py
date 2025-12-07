import os
import argparse
import json
import time
import numpy as np
import pandas as pd
from detection3d.mmdet3d_inference2 import *
from pathlib import Path
from collections import defaultdict

try:
    from mmdet3d.apis import (
        LidarDet3DInferencer,
        MonoDet3DInferencer,
        MultiModalityDet3DInferencer
    )
except ImportError:
    print("Error: This script requires 'mmdetection3d' and its dependencies.")
    exit()

try:
    import open3d as o3d
except ImportError:
    print("Error: This script requires 'open3d' for visualization.")
    exit()

try:
    import cv2
except ImportError:
    print("Error: This script requires 'opencv-python' for 2D visualization.")
    exit()

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: This script requires 'matplotlib'.")
    exit()
    
class MultiModelBenchmark:
    
    def __init__(self, config):
        self.config = config
        self.results = defaultdict(lambda: defaultdict(list))
        self.metric_summary = []
        
    def calculate_iou_3d(self, box1, box2):
        """Calculates the 3D IoU between 2 bounding boxes"""
        
        try:
            obb1 = self._create_o3d_box(box1)
            obb2 = self._create_o3d_box(box2)
            
            # Compute volume of each box
            vol1 = np.prod(box1[3:6])
            vol2 = np.prod(box2[3:6])
            
            # Approximate intersection using center distance
            center_dist = np.linalg.norm(np.array(box1[:3]) - np.array(box2[:3]))
            max_dim = max(np.max(box1[3:6]), np.max(box2[3:6]))
            
            if center_dist > max_dim * 2:
                return 0.0
            
            # Overlap approximation
            overlap_ratio = max(0, 1 - center_dist / max_dim)
            intersection = min(vol1, vol2) * overlap_ratio
            union = vol1 + vol2 - intersection
            
            return intersection / union if union > 0 else 0.0   
            
        except:
            return 0
    
    def _create_o3d_box(self, bbox):
        """Create Open3D oriented bounding boxes"""
        
        center = np.array(bbox[:3], dtype=float)
        extent = np.array(bbox[3:6], dtype=float)
        yaw = float(bbox[6])
        center[2] = center[2] + extent[2] / 2.0
        R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
        return o3d.geometry.OrientedBoundingBox(center, R, extent)
    
    def match_predictions_to_gt(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """Match predicted boxes to ground truth using Hungarian algorithm approximation."""
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = self.calculate_iou_3d(pred_box, gt_box)
        
        # Greedy matching
        matched_pairs = []
        used_pred = set()
        used_gt = set()
        
        while True:
            max_iou = np.max(iou_matrix)
            if max_iou < iou_threshold:
                break
            
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matched_pairs.append((i, j, iou_matrix[i, j]))
            used_pred.add(i)
            used_gt.add(j)
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
        
        false_positives = [i for i in range(len(pred_boxes)) if i not in used_pred]
        false_negatives = [j for j in range(len(gt_boxes)) if j not in used_gt]
        
        return matched_pairs, false_positives, false_negatives
    
    def compute_metrics(self, pred_boxes, gt_boxes, pred_scores=None, iou_threshold=0.5):
        """Compute precision, recall, and AP metrics."""
        
        matched, fp, fn = self.match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold)
        
        tp = len(matched)
        fp_count = len(fp)
        fn_count = len(fn)
        
        precision = tp / (tp + fp_count) if (tp + fp_count) > 0 else 0.0
        recall = tp / (tp + fn_count) if (tp + fn_count) > 0 else 0.0
        
        # Average IoU for matched boxes
        avg_iou = np.mean([iou for _, _, iou in matched]) if matched else 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'true_positives': tp,
            'false_positives': fp_count,
            'false_negatives': fn_count,
            'avg_iou': avg_iou,
            'num_predictions': len(pred_boxes),
            'num_ground_truth': len(gt_boxes)
        }
        
        return metrics
    
    def run_inference_single_model(self, model_config, inputs_list):
        """Run inference for a single model on a list of inputs."""
        
        model_name = model_config['name']
        model_path = model_config['config']
        checkpoint = model_config['checkpoint']
        modality = model_config.get('modality', 'lidar')
        
        print(f"\n{'='*80}")
        print(f"Running Model: {model_name}")
        print(f"{'='*80}")
        
        # Initialize inferencer
        if modality == 'lidar':
            InferencerClass = LidarDet3DInferencer
        elif modality == 'mono':
            InferencerClass = MonoDet3DInferencer
        else:
            InferencerClass = MultiModalityDet3DInferencer
        
        inferencer = InferencerClass(
            model_path,
            checkpoint,
            device=self.config['device']
        )
        
        model_results = []
        
        for idx, single_input in enumerate(inputs_list):
            frame_id = single_input.get('frame_id', f'frame_{idx}')
            print(f"\n[{idx+1}/{len(inputs_list)}] Processing: {frame_id}")
            
            # Load GT if available
            gt_bboxes = []
            if 'gt_label' in single_input and single_input['gt_label']:
                try:
                    gt_bboxes = load_kitti_gt_labels(single_input['gt_label'])
                except Exception as e:
                    print(f"  Warning: Could not load GT labels: {e}")
            
            # Run inference with timing
            start_time = time.time()
            results_dict = inferencer(
                single_input,
                show=False,
                out_dir=self.config['out_dir'],
                pred_score_thr=self.config['score_threshold']
            )
            inference_time = time.time() - start_time
            
            pred_dict = results_dict['predictions'][0]
            pred_bboxes = np.array(pred_dict['bboxes_3d'])
            pred_scores = pred_dict.get('scores_3d', [])
            pred_labels = pred_dict.get('labels_3d', [])
            
            # Compute metrics
            metrics = self.compute_metrics(
                pred_bboxes, 
                gt_bboxes, 
                pred_scores,
                iou_threshold=0.5
            )
            metrics['inference_time'] = inference_time
            metrics['fps'] = 1.0 / inference_time if inference_time > 0 else 0
            
            # Save results
            frame_result = {
                'model_name': model_name,
                'frame_id': frame_id,
                'metrics': metrics,
                'pred_bboxes': pred_bboxes.tolist(),
                'pred_scores': pred_scores if isinstance(pred_scores, list) else pred_scores.tolist(),
                'pred_labels': pred_labels if isinstance(pred_labels, list) else pred_labels.tolist(),
                'gt_bboxes': [box.tolist() for box in gt_bboxes],
                'input_files': single_input
            }
            
            model_results.append(frame_result)
            
            # Visualization
            self.save_visualizations(
                single_input,
                pred_dict,
                gt_bboxes,
                model_name,
                frame_id,
                modality
            )
            
            print(f"  Metrics: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
                  f"IoU={metrics['avg_iou']:.3f}, FPS={metrics['fps']:.1f}")
        
        return model_results
    
    def save_visualizations(self, input_dict, pred_dict, gt_bboxes, model_name, frame_id, modality):
        """Save PNG and PLY visualizations."""
        output_dir = Path(self.config['out_dir']) / model_name / frame_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pred_bboxes = np.array(pred_dict['bboxes_3d'])
        pred_labels = pred_dict.get('labels_3d', [])
        pred_scores = pred_dict.get('scores_3d', [])
        
        # Get class names from metainfo
        metainfo = pred_dict.get('metainfo', {})
        class_names = metainfo.get('classes', ['Car', 'Pedestrian', 'Cyclist'])
        
        # Save JSON metadata
        metadata = {
            'model': model_name,
            'frame_id': frame_id,
            'num_predictions': len(pred_bboxes),
            'num_ground_truth': len(gt_bboxes),
            'predictions': {
                'bboxes_3d': pred_bboxes.tolist(),
                'scores_3d': pred_scores if isinstance(pred_scores, list) else pred_scores.tolist(),
                'labels_3d': pred_labels if isinstance(pred_labels, list) else pred_labels.tolist(),
            }
        }
        with open(output_dir / f"{frame_id}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 2D visualization
        if 'img' in input_dict and input_dict['img'] and 'calib' in input_dict and input_dict['calib']:
            try:
                img_2d_path = output_dir / f"{frame_id}_2d_vis.png"
                draw_projected_boxes_on_image(
                    input_dict['img'],
                    input_dict['calib'],
                    pred_bboxes,
                    gt_bboxes,
                    str(img_2d_path),
                    pred_labels=pred_labels,
                    class_names=class_names
                )
            except Exception as e:
                print(f"  Warning: 2D visualization failed: {e}")
        
        # 3D visualization (PLY files)
        if modality != 'mono' and 'points' in input_dict:
            try:
                self.save_3d_ply_files(
                    input_dict['points'],
                    pred_dict,
                    gt_bboxes,
                    output_dir,
                    frame_id
                )
            except Exception as e:
                print(f"  Warning: 3D PLY export failed: {e}")
    
    def save_3d_ply_files(self, lidar_file, pred_dict, gt_bboxes, output_dir, frame_id):
        """Save separate PLY files for points, bboxes, and labels."""
        
        points = load_lidar_file(lidar_file)
        
        # Save point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd_colors = color_points_by_height(points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        o3d.io.write_point_cloud(str(output_dir / f"{frame_id}_points.ply"), pcd)
        
        # Save coordinate axes
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        o3d.io.write_triangle_mesh(str(output_dir / f"{frame_id}_axes.ply"), axes)
        
        # Save predicted bboxes
        pred_bboxes = np.array(pred_dict['bboxes_3d'])
        pred_labels = pred_dict.get('labels_3d', [])
        metainfo = pred_dict.get('metainfo', {})
        class_names = metainfo.get('classes', ['Car', 'Pedestrian', 'Cyclist'])
        
        if len(pred_bboxes) > 0:
            pred_line_sets = []
            pred_label_line_sets = []
            
            for i, bbox in enumerate(pred_bboxes):
                # Bounding box
                bbox_lines = create_open3d_bbox(bbox, color=[0.0, 1.0, 0.0])
                pred_line_sets.append(bbox_lines)
                
                # Class label text
                cls_id = None
                if isinstance(pred_labels, (list, np.ndarray)) and i < len(pred_labels):
                    try:
                        cls_id = int(pred_labels[i])
                    except:
                        pass
                cls_name = class_names[cls_id] if (cls_id is not None and 0 <= cls_id < len(class_names)) else 'OBJ'
                top_pos = get_bbox_top_center(bbox)
                text_ls = create_text_stroke_label(cls_name, top_pos, color=[1.0, 1.0, 1.0], scale=0.6)
                pred_label_line_sets.append(text_ls)
            
            combined_pred = combine_line_sets(pred_line_sets, color=[0.0, 1.0, 0.0])
            o3d.io.write_line_set(str(output_dir / f"{frame_id}_pred_bboxes.ply"), combined_pred)
            
            if pred_label_line_sets:
                combined_labels = combine_line_sets(pred_label_line_sets, color=[1.0, 1.0, 1.0])
                o3d.io.write_line_set(str(output_dir / f"{frame_id}_pred_labels.ply"), combined_labels)
        
        # Save ground truth bboxes
        if len(gt_bboxes) > 0:
            gt_line_sets = []
            for bbox in gt_bboxes:
                bbox_lines = create_open3d_bbox(bbox, color=[1.0, 0.0, 0.0])
                gt_line_sets.append(bbox_lines)
            
            combined_gt = combine_line_sets(gt_line_sets, color=[1.0, 0.0, 0.0])
            o3d.io.write_line_set(str(output_dir / f"{frame_id}_gt_bboxes.ply"), combined_gt)
    
    def create_video_from_frames(self, model_name, fps=5):
        """Create a video from saved 2D visualization frames."""
        
        try:
            import cv2
        except ImportError:
            print("Warning: opencv-python not available for video creation")
            return
        
        model_dir = Path(self.config['out_dir']) / model_name
        if not model_dir.exists():
            print(f"Warning: No output directory for {model_name}")
            return
        
        # Find all 2D visualization frames
        frame_files = sorted(model_dir.glob("*/*_2d_vis.png"))
        if not frame_files:
            print(f"Warning: No 2D frames found for {model_name}")
            return
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            return
        
        height, width = first_frame.shape[:2]
        
        # Create video writer
        video_path = model_dir / f"{model_name}_demo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        print(f"\nCreating video for {model_name}: {len(frame_files)} frames")
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        print(f"  Saved video: {video_path}")
    
    def generate_comparison_report(self, all_results):
        """Generate comparison tables and analysis."""
        
        print("\n" + "="*80)
        print("GENERATING COMPARISON REPORT")
        print("="*80)
        
        # Aggregate metrics by model
        model_metrics = defaultdict(lambda: defaultdict(list))
        
        for model_results in all_results:
            for frame_result in model_results:
                model_name = frame_result['model_name']
                metrics = frame_result['metrics']
                
                for key, value in metrics.items():
                    model_metrics[model_name][key].append(value)
        
        # Compute averages
        summary_data = []
        for model_name, metrics_dict in model_metrics.items():
            row = {'Model': model_name}
            for metric_name, values in metrics_dict.items():
                if values:
                    row[metric_name] = np.mean(values)
            summary_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Select key metrics for display
        display_columns = ['Model', 'precision', 'recall', 'avg_iou', 'fps', 
                          'true_positives', 'false_positives', 'false_negatives']
        display_columns = [col for col in display_columns if col in df.columns]
        df_display = df[display_columns]
        
        # Format for readability
        for col in ['precision', 'recall', 'avg_iou']:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(3)
        if 'fps' in df_display.columns:
            df_display['fps'] = df_display['fps'].round(1)
        
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        print(df_display.to_string(index=False))
        
        # Save to CSV
        output_path = Path(self.config['out_dir']) / "comparison_metrics.csv"
        df.to_csv(output_path, index=False)
        print(f"\nFull metrics saved to: {output_path}")
        
        # Generate key takeaways
        self.generate_takeaways(df)
        
        return df
    
    def generate_takeaways(self, df):
        """Generate 3-5 key takeaways from comparison."""
        
        print("\n" + "="*80)
        print("KEY TAKEAWAYS")
        print("="*80)
        
        takeaways = []
        
        # 1. Best overall model
        if 'precision' in df.columns and 'recall' in df.columns:
            df['f1'] = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'] + 1e-8)
            best_idx = df['f1'].idxmax()
            best_model = df.loc[best_idx, 'Model']
            best_f1 = df.loc[best_idx, 'f1']
            takeaways.append(
                f"1. Best Overall: {best_model} achieves highest F1-score ({best_f1:.3f}), "
                f"balancing precision ({df.loc[best_idx, 'precision']:.3f}) and recall ({df.loc[best_idx, 'recall']:.3f})"
            )
        
        # 2. Speed champion
        if 'fps' in df.columns:
            fastest_idx = df['fps'].idxmax()
            fastest_model = df.loc[fastest_idx, 'Model']
            fastest_fps = df.loc[fastest_idx, 'fps']
            takeaways.append(
                f"2. Speed Champion: {fastest_model} runs fastest at {fastest_fps:.1f} FPS, "
                f"making it suitable for real-time applications"
            )
        
        # 3. Accuracy vs Speed tradeoff
        if 'precision' in df.columns and 'fps' in df.columns:
            df['efficiency'] = df['precision'] * df['fps']
            efficient_idx = df['efficiency'].idxmax()
            efficient_model = df.loc[efficient_idx, 'Model']
            takeaways.append(
                f"3. Best Accuracy/Speed Tradeoff: {efficient_model} offers optimal balance "
                f"(precision={df.loc[efficient_idx, 'precision']:.3f}, fps={df.loc[efficient_idx, 'fps']:.1f})"
            )
        
        # 4. Localization accuracy
        if 'avg_iou' in df.columns:
            best_iou_idx = df['avg_iou'].idxmax()
            best_iou_model = df.loc[best_iou_idx, 'Model']
            best_iou = df.loc[best_iou_idx, 'avg_iou']
            takeaways.append(
                f"4. Best Localization: {best_iou_model} has highest IoU ({best_iou:.3f}), "
                f"indicating superior bounding box accuracy"
            )
        
        # 5. False positive/negative analysis
        if 'false_positives' in df.columns and 'false_negatives' in df.columns:
            df['fp_rate'] = df['false_positives'] / (df['true_positives'] + df['false_positives'] + 1e-8)
            df['fn_rate'] = df['false_negatives'] / (df['true_positives'] + df['false_negatives'] + 1e-8)
            
            conservative_idx = df['fp_rate'].idxmin()
            conservative_model = df.loc[conservative_idx, 'Model']
            
            aggressive_idx = df['fn_rate'].idxmin()
            aggressive_model = df.loc[aggressive_idx, 'Model']
            
            takeaways.append(
                f"5. Error Patterns: {conservative_model} minimizes false positives (conservative), "
                f"while {aggressive_model} minimizes false negatives (aggressive detection)"
            )
        
        for takeaway in takeaways[:5]:  # Limit to 5
            print(f"\n{takeaway}")
        
        print("\n" + "="*80)
    
    def run(self):
        """Main execution method."""
        
        all_results = []
        
        # Load dataset inputs once
        print(f"\nLoading dataset: {self.config['dataset_name']}")
        inputs_list = self.load_dataset()
        print(f"Loaded {len(inputs_list)} samples")
        
        # Run each model
        for model_config in self.config['models']:
            model_results = self.run_inference_single_model(model_config, inputs_list)
            all_results.append(model_results)
            
            # Create video for this model
            if self.config.get('create_video', True):
                self.create_video_from_frames(model_config['name'], fps=self.config.get('video_fps', 5))
        
        # Generate comparison report
        if len(all_results) > 1:
            self.generate_comparison_report(all_results)
        
        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print("="*80)
        print(f"Results saved to: {self.config['out_dir']}")
    
    def load_dataset(self):
        """Load dataset based on configuration."""
        
        dataset_type = self.config.get('dataset_type', 'kitti')
        dataset_path = self.config['dataset_path']
        frame_number = self.config.get('frame_number', '-1')
        
        if dataset_type == 'kitti':
            return build_kitti_input_list(dataset_path, frame_number)
        elif dataset_type == 'waymokitti':
            return build_waymokitti_input_list(dataset_path, frame_number)
        else:
            # Single file or custom
            return [build_input_dict(
                dataset_path,
                self.config['modality'],
                self.config.get('img_dir'),
                self.config.get('calib_dir'),
                self.config.get('gt_label_dir')
            )]
            
def main():
    parser = argparse.ArgumentParser(description="Multi-Model MMDetection3D Benchmark")
    parser.add_argument('--config', type=str, required=True,
                       help="Path to benchmark configuration JSON file")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Run benchmark
    benchmark = MultiModelBenchmark(config)
    benchmark.run()


if __name__ == "__main__":
    main()