"""
Analyze sample videos and provide quality feedback.
This script reviews sample videos and suggests improvements for final dataset.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from pathlib import Path


SAMPLE_DIR = "data/videos/sample_videos"


class VideoAnalyzer:
    """Analyze sample videos for quality assessment."""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
    
    def analyze_video(self, video_path):
        """
        Analyze a single video and provide detailed feedback.
        
        Returns:
            dict with analysis results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Could not open video"}
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Analysis metrics
        frames_with_hands = 0
        frames_with_both_hands = 0
        brightness_values = []
        hand_sizes = []
        hands_off_screen_count = 0
        
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Brightness analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Hand detection
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    num_hands = len(results.multi_hand_landmarks)
                    frames_with_hands += 1
                    
                    if num_hands == 2:
                        frames_with_both_hands += 1
                    
                    # Check if hands are within frame bounds
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        
                        # Check if any landmarks are near edges (potential off-screen)
                        if (min(x_coords) < 0.05 or max(x_coords) > 0.95 or
                            min(y_coords) < 0.05 or max(y_coords) > 0.95):
                            hands_off_screen_count += 1
                        
                        # Hand size (bounding box area)
                        hand_width = max(x_coords) - min(x_coords)
                        hand_height = max(y_coords) - min(y_coords)
                        hand_area = hand_width * hand_height
                        hand_sizes.append(hand_area)
                
                frame_count += 1
        
        cap.release()
        
        # Calculate metrics
        hand_detection_rate = frames_with_hands / max(frame_count, 1)
        both_hands_rate = frames_with_both_hands / max(frame_count, 1)
        avg_brightness = np.mean(brightness_values) if brightness_values else 0
        avg_hand_size = np.mean(hand_sizes) if hand_sizes else 0
        
        # Quality assessment
        quality_score = 0
        issues = []
        suggestions = []
        
        # Duration check (3-10 seconds ideal for multi-word ISL phrases)
        if duration < 2:
            issues.append(f"Video too short ({duration:.1f}s) - should be 3-10 seconds")
            suggestions.append("Record longer videos to capture complete phrase")
        elif duration > 15:
            issues.append(f"Video too long ({duration:.1f}s) - should be 3-10 seconds")
            suggestions.append("Trim video or remove unnecessary pauses")
        else:
            quality_score += 25
        
        # Hand detection rate
        if hand_detection_rate < 0.5:
            issues.append(f"Low hand detection ({hand_detection_rate:.0%}) - hands not visible enough")
            suggestions.append("Ensure hands are clearly visible throughout")
            suggestions.append("Improve lighting on hands")
        elif hand_detection_rate < 0.8:
            issues.append(f"Moderate hand detection ({hand_detection_rate:.0%}) - could be better")
            suggestions.append("Keep hands in frame more consistently")
            quality_score += 15
        else:
            quality_score += 25
        
        # Both hands detection
        if both_hands_rate < 0.3:
            issues.append(f"Rarely shows both hands ({both_hands_rate:.0%})")
            suggestions.append("If phrase requires both hands, keep both visible")
        else:
            quality_score += 15
        
        # Brightness
        if avg_brightness < 80:
            issues.append(f"Video too dark (brightness: {avg_brightness:.0f}/255)")
            suggestions.append("Record in brighter environment")
            suggestions.append("Add more lighting or move closer to window")
        elif avg_brightness > 200:
            issues.append(f"Video too bright (brightness: {avg_brightness:.0f}/255)")
            suggestions.append("Reduce lighting or avoid direct sunlight")
        else:
            quality_score += 20
        
        # Hands off-screen check
        if hands_off_screen_count > frame_count * 0.2:
            issues.append("Hands frequently near edge or off-screen")
            suggestions.append("Position yourself so hands stay in center of frame")
            suggestions.append("Use a wider camera angle or step back")
        else:
            quality_score += 15
        
        # Overall assessment
        if quality_score >= 80:
            quality = "EXCELLENT ✅"
        elif quality_score >= 60:
            quality = "GOOD ✓"
        elif quality_score >= 40:
            quality = "ACCEPTABLE ⚠️"
        else:
            quality = "NEEDS IMPROVEMENT ❌"
        
        return {
            "filename": Path(video_path).name,
            "duration": duration,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "total_frames": total_frames,
            "hand_detection_rate": hand_detection_rate,
            "both_hands_rate": both_hands_rate,
            "avg_brightness": avg_brightness,
            "avg_hand_size": avg_hand_size,
            "hands_off_screen": hands_off_screen_count,
            "quality_score": quality_score,
            "quality": quality,
            "issues": issues,
            "suggestions": suggestions
        }
    
    def analyze_all_samples(self):
        """Analyze all sample videos in the directory."""
        if not os.path.exists(SAMPLE_DIR):
            print(f"❌ Sample directory not found: {SAMPLE_DIR}")
            print(f"\nPlease place your sample videos in: {SAMPLE_DIR}/")
            return
        
        video_extensions = ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(SAMPLE_DIR, ext)))
        
        if not video_files:
            print(f"❌ No sample videos found in: {SAMPLE_DIR}")
            print(f"\nPlease add sample MP4/MOV/AVI files to this directory")
            print(f"\nExample naming:")
            print(f"  - phrase0_sample.mp4")
            print(f"  - phrase1_person1.mov")
            print(f"  - test_video.mp4")
            return
        
        print("="*80)
        print("SAMPLE VIDEO ANALYSIS - Quality Assessment")
        print("="*80)
        print(f"\nFound {len(video_files)} sample video(s)\n")
        
        results = []
        for video_path in video_files:
            print(f"\n{'='*80}")
            print(f"Analyzing: {Path(video_path).name}")
            print(f"{'='*80}")
            
            result = self.analyze_video(video_path)
            results.append(result)
            
            # Print detailed analysis
            print(f"\n📹 Video Properties:")
            print(f"   Duration: {result['duration']:.2f} seconds")
            print(f"   FPS: {result['fps']:.1f}")
            print(f"   Resolution: {result['resolution']}")
            print(f"   Total Frames: {result['total_frames']}")
            
            print(f"\n🖐️  Hand Detection:")
            print(f"   Hand Detection Rate: {result['hand_detection_rate']:.0%}")
            print(f"   Both Hands Visible: {result['both_hands_rate']:.0%}")
            print(f"   Hands Off-screen Frames: {result['hands_off_screen']}")
            
            print(f"\n💡 Lighting:")
            print(f"   Average Brightness: {result['avg_brightness']:.0f}/255")
            
            print(f"\n⭐ Quality Score: {result['quality_score']}/100 - {result['quality']}")
            
            if result['issues']:
                print(f"\n⚠️  Issues Found:")
                for issue in result['issues']:
                    print(f"   - {issue}")
            
            if result['suggestions']:
                print(f"\n💡 Suggestions for Improvement:")
                for suggestion in result['suggestions']:
                    print(f"   - {suggestion}")
            
            if result['quality_score'] >= 80:
                print(f"\n✅ This video quality is excellent for final dataset!")
            elif result['quality_score'] >= 60:
                print(f"\n✓ This video quality is good - minor improvements would help")
            else:
                print(f"\n⚠️  Please address issues before recording final dataset")
        
        # Overall summary
        print(f"\n\n{'='*80}")
        print("SUMMARY & RECOMMENDATIONS")
        print(f"{'='*80}")
        
        avg_quality = np.mean([r['quality_score'] for r in results])
        excellent_count = sum(1 for r in results if r['quality_score'] >= 80)
        good_count = sum(1 for r in results if 60 <= r['quality_score'] < 80)
        needs_work = sum(1 for r in results if r['quality_score'] < 60)
        
        print(f"\nTotal Videos Analyzed: {len(results)}")
        print(f"Average Quality Score: {avg_quality:.0f}/100")
        print(f"\n  ✅ Excellent Quality: {excellent_count}")
        print(f"  ✓  Good Quality: {good_count}")
        print(f"  ⚠️  Needs Improvement: {needs_work}")
        
        # Consolidated recommendations
        all_issues = []
        all_suggestions = []
        for r in results:
            all_issues.extend(r['issues'])
            all_suggestions.extend(r['suggestions'])
        
        common_issues = set(all_issues)
        common_suggestions = set(all_suggestions)
        
        if common_issues:
            print(f"\n📋 Common Issues Across Videos:")
            for issue in list(common_issues)[:5]:  # Top 5
                print(f"   - {issue}")
        
        if common_suggestions:
            print(f"\n💡 Key Recommendations for Final Dataset:")
            for suggestion in list(common_suggestions)[:5]:  # Top 5
                print(f"   - {suggestion}")
        
        print(f"\n{'='*80}")
        print("CHECKLIST FOR FINAL DATASET VIDEOS")
        print(f"{'='*80}")
        print("""
✓ Duration: 3-10 seconds per video (longer for multi-word phrases)
✓ Hands visible: >80% of frames
✓ Both hands shown when needed
✓ Good lighting (brightness: 80-200)
✓ Hands stay in frame (not touching edges)
✓ Plain background
✓ Resolution: 720p or higher
✓ Stable camera (no shaking)
✓ Clear, deliberate hand movements
        """)
        
        print(f"\n📁 Next Steps:")
        if avg_quality >= 70:
            print(f"   ✅ Sample quality is good! Apply same setup for final dataset")
            print(f"   ✅ Record 50 videos per phrase (5 people × 10 videos)")
            print(f"   ✅ Maintain this quality consistently")
        else:
            print(f"   ⚠️  Improve sample quality before recording full dataset")
            print(f"   ⚠️  Address issues mentioned above")
            print(f"   ⚠️  Test with a few more samples until quality is >70/100")


def main():
    analyzer = VideoAnalyzer()
    
    print("\n" + "="*80)
    print("SAMPLE VIDEO QUALITY ANALYZER")
    print("="*80)
    print(f"\nPlace your sample videos in: {SAMPLE_DIR}/")
    print("Supported formats: MP4, MOV, AVI")
    print("\nThis will analyze:")
    print("  - Video duration and frame rate")
    print("  - Hand detection quality")
    print("  - Lighting conditions")
    print("  - Frame composition")
    print("  - Overall quality score")
    print("\n" + "="*80 + "\n")
    
    analyzer.analyze_all_samples()


if __name__ == "__main__":
    main()
