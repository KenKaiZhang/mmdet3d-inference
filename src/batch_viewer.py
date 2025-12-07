#!/usr/bin/env python3
"""
Enhanced Batch Viewer for Multiple Models and Frames
Headless mode using matplotlib for Docker compatibility.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is not installed. Install with `pip install open3d`.")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Set backend before importing pyplot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("Error: matplotlib is not installed. Install with `pip install matplotlib`.")
    sys.exit(1)

import json
import numpy as np


class BatchViewer:
    def __init__(self, base_dir, models):
        self.base_dir = Path(base_dir)
        self.models = models
        self.current_model_idx = 0
        self.current_frame_idx = 0
        self.frames = []
        
        # Discover available frames
        self.discover_frames()
        
        if not self.frames:
            print("Error: No frames found in directory")
            sys.exit(1)
        
        print(f"Found {len(self.frames)} frames across {len(self.models)} models")
    
    def discover_frames(self):
        """Discover all available frames."""
        
        if not self.models:
            # Auto-discover models
            self.models = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        
        if not self.models:
            return
        
        # Use first model to find frame list
        first_model_dir = self.base_dir / self.models[0]
        if first_model_dir.exists():
            frame_dirs = sorted([d.name for d in first_model_dir.iterdir() if d.is_dir()])
            self.frames = frame_dirs
    
    def load_geometries(self, model_name, frame_id, verbose=True):
        """Load all geometries for a specific model and frame."""
        
        frame_dir = self.base_dir / model_name / frame_id
        
        if not frame_dir.exists():
            if verbose:
                print(f"Warning: Frame directory does not exist: {frame_dir}")
            return []
        
        geometries = []
        
        # Load point cloud
        pcd_path = frame_dir / f"{frame_id}_points.ply"
        if pcd_path.exists():
            try:
                pcd = o3d.io.read_point_cloud(str(pcd_path))
                geometries.append(("Point Cloud", pcd))
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load point cloud: {e}")
        
        # Load axes
        axes_path = frame_dir / f"{frame_id}_axes.ply"
        if axes_path.exists():
            try:
                axes = o3d.io.read_triangle_mesh(str(axes_path))
                geometries.append(("Axes", axes))
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load axes: {e}")
        
        # Load predicted bboxes
        pred_bbox_path = frame_dir / f"{frame_id}_pred_bboxes.ply"
        if pred_bbox_path.exists():
            try:
                pred_bboxes = o3d.io.read_line_set(str(pred_bbox_path))
                geometries.append(("Predicted Boxes", pred_bboxes))
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load predicted boxes: {e}")
        
        # Load predicted labels
        pred_label_path = frame_dir / f"{frame_id}_pred_labels.ply"
        if pred_label_path.exists():
            try:
                pred_labels = o3d.io.read_line_set(str(pred_label_path))
                geometries.append(("Predicted Labels", pred_labels))
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load predicted labels: {e}")
        
        # Load ground truth bboxes
        gt_bbox_path = frame_dir / f"{frame_id}_gt_bboxes.ply"
        if gt_bbox_path.exists():
            try:
                gt_bboxes = o3d.io.read_line_set(str(gt_bbox_path))
                geometries.append(("Ground Truth Boxes", gt_bboxes))
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load GT boxes: {e}")
        
        # Load metadata
        metadata = {}
        metadata_path = frame_dir / f"{frame_id}_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if verbose:
                        print(f"  Predictions: {metadata.get('num_predictions', 'N/A')}, "
                              f"Ground Truth: {metadata.get('num_ground_truth', 'N/A')}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load metadata: {e}")
        
        return geometries, metadata
    
    def render_with_matplotlib(self, geometries, output_path, title, width=1920, height=1080, dpi=100):
        """Render geometries using matplotlib."""
        
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Track bounds for setting axis limits
        all_points = []
        
        for name, geom in geometries:
            if isinstance(geom, o3d.geometry.PointCloud):
                # Render point cloud
                points = np.asarray(geom.points)
                if len(points) > 0:
                    all_points.append(points)
                    colors = np.asarray(geom.colors) if geom.has_colors() else None
                    
                    # Downsample for faster rendering if too many points
                    if len(points) > 100000:
                        indices = np.random.choice(len(points), 100000, replace=False)
                        points = points[indices]
                        if colors is not None:
                            colors = colors[indices]
                    
                    if colors is not None:
                        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                 c=colors, s=0.1, alpha=0.6)
                    else:
                        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                 c='gray', s=0.1, alpha=0.6)
            
            elif isinstance(geom, o3d.geometry.LineSet):
                # Render bounding boxes
                points = np.asarray(geom.points)
                lines = np.asarray(geom.lines)
                colors = np.asarray(geom.colors) if geom.has_colors() else None
                
                if len(points) > 0:
                    all_points.append(points)
                    
                    for i, line in enumerate(lines):
                        p1, p2 = points[line[0]], points[line[1]]
                        color = colors[i] if colors is not None else [1, 0, 0]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                               color=color, linewidth=2)
            
            elif isinstance(geom, o3d.geometry.TriangleMesh):
                # Render mesh (axes)
                vertices = np.asarray(geom.vertices)
                if len(vertices) > 0:
                    all_points.append(vertices)
                    triangles = np.asarray(geom.triangles)
                    colors = np.asarray(geom.vertex_colors) if geom.has_vertex_colors() else None
                    
                    # Simple mesh rendering
                    for tri in triangles[:min(len(triangles), 1000)]:  # Limit triangles
                        pts = vertices[tri]
                        color = colors[tri[0]] if colors is not None else [0.5, 0.5, 0.5]
                        ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], 
                                      color=color, alpha=0.7)
        
        # Set axis limits based on all points
        if all_points:
            all_points = np.vstack(all_points)
            margin = 5
            ax.set_xlim([all_points[:, 0].min() - margin, all_points[:, 0].max() + margin])
            ax.set_ylim([all_points[:, 1].min() - margin, all_points[:, 1].max() + margin])
            ax.set_zlim([all_points[:, 2].min() - margin, all_points[:, 2].max() + margin])
        
        # Set viewing angle (bird's eye view for autonomous driving)
        ax.view_init(elev=30, azim=45)
        
        # Labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title, fontsize=16, pad=20)
        
        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 0.5])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    def save_screenshots_headless(self, output_dir="./screenshots", max_frames=None, 
                                   width=1920, height=1080):
        """Save screenshots in headless mode using matplotlib."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nRunning in headless mode (matplotlib)")
        print(f"Saving screenshots to: {output_path}")
        print(f"Resolution: {width}x{height}")
        if max_frames:
            print(f"Max frames per model: {max_frames}")
        print("="*80)
        
        total_saved = 0
        
        for model_name in self.models:
            print(f"\nModel: {model_name}")
            print("-"*80)
            
            frames_to_process = self.frames[:max_frames] if max_frames else self.frames
            
            for frame_id in frames_to_process:
                geometries, metadata = self.load_geometries(model_name, frame_id, verbose=True)
                
                if not geometries:
                    print(f"  Skipping {frame_id} (no geometries)")
                    continue
                
                print(f"  Processing {frame_id}... ", end="", flush=True)
                
                try:
                    screenshot_path = output_path / f"{model_name}_{frame_id}.png"
                    title = f"{model_name} - {frame_id}"
                    if metadata:
                        title += f" | Pred: {metadata.get('num_predictions', 'N/A')}, GT: {metadata.get('num_ground_truth', 'N/A')}"
                    
                    self.render_with_matplotlib(geometries, screenshot_path, title, width, height)
                    
                    total_saved += 1
                    print(f"✓ Saved")
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
                    import traceback
                    traceback.print_exc()
        
        print("\n" + "="*80)
        print(f"Complete! Saved {total_saved} screenshots to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Batch Viewer with Matplotlib Backend")
    parser.add_argument("--dir", default="./benchmark_results",
                       help="Base directory containing model results")
    parser.add_argument("--models", nargs="*", default=None,
                       help="List of model names to view (default: auto-discover)")
    parser.add_argument("--screenshot-dir", default="./screenshots",
                       help="Directory for saving screenshots")
    parser.add_argument("--width", type=int, default=1920,
                       help="Screenshot width in pixels (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                       help="Screenshot height in pixels (default: 1080)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to process per model (default: all)")
    
    args = parser.parse_args()
    
    viewer = BatchViewer(args.dir, args.models or [])
    viewer.save_screenshots_headless(
        args.screenshot_dir, 
        args.max_frames,
        args.width,
        args.height
    )


if __name__ == "__main__":
    main()