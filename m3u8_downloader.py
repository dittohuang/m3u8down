#!/usr/bin/env python3

import argparse
import os
import requests
import urllib.parse
import cv2
from skimage.metrics import structural_similarity as ssim
import shutil
import subprocess
import logging
import time

# Global constants
DEFAULT_TIMEOUT = 15  # seconds
MAX_DOWNLOAD_RETRIES = 10 # Max retries for downloads
RETRY_SLEEP_DURATION = 5 # seconds

def parse_input_file(file_path):
    pairs = []
    try:
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                if '$' not in line:
                    logging.warning(f"Line {line_number} in {file_path} does not contain '$' delimiter: '{line}'")
                    continue
                parts = line.split('$', 1)
                if len(parts) == 2:
                    name, url = parts[0].strip(), parts[1].strip()
                    if not name:
                        logging.warning(f"Line {line_number} in {file_path} has missing name: '{line}'")
                        continue
                    if not url:
                        logging.warning(f"Line {line_number} in {file_path} has missing URL: '{line}'")
                        continue
                    pairs.append((name, url))
                else:
                    logging.warning(f"Line {line_number} in {file_path} could not be split correctly: '{line}'")
    except FileNotFoundError:
        logging.error(f"Input file not found at {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while parsing {file_path}: {e}", exc_info=True)
    return pairs

def extract_first_frame(video_path, output_image_path):
    cap = None
    try:
        if not os.path.exists(video_path):
            logging.warning(f"Video file not found for frame extraction: {video_path}")
            return False
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"Could not open video file for frame extraction: {video_path}")
            return False

        success, frame = cap.read()
        if success:
            try:
                logging.debug(f"Original frame size: {frame.shape[1]}x{frame.shape[0]} for {video_path}")
                resized_frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                logging.debug(f"Resized frame from {video_path} to 320x180 before saving to {output_image_path}")

                if cv2.imwrite(output_image_path, resized_frame):
                    logging.debug(f"Successfully extracted and saved resized first frame from {video_path} to {output_image_path}")
                    return True
                else:
                    logging.error(f"Failed to save resized frame to {output_image_path} (cv2.imwrite might have failed silently).")
                    return False
            except Exception as e:
                logging.error(f"Error resizing or writing frame from {video_path}: {e}", exc_info=True)
                return False
        else:
            logging.warning(f"Could not read first frame from video: {video_path}")
            return False

    except cv2.error as e: # Should be caught by the general Exception below if specific handling isn't needed
        logging.error(f"OpenCV error during frame extraction setup from {video_path}: {e}", exc_info=True)
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during frame extraction from {video_path}: {e}", exc_info=True)
        return False
    finally:
        if cap:
            cap.release()

def get_ssim_score(image_path1, image_path2):
    try:
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        if img1 is None:
            logging.warning(f"Could not read image {image_path1} for SSIM.")
            return 0.0
        if img2 is None:
            logging.warning(f"Could not read image {image_path2} for SSIM.")
            return 0.0

        # SSIM expects grayscale images of the same size. Frames are now resized to 320x180.
        # If not, resizing here would be another option:
        # if img1.shape != img2.shape:
        #    logging.warning(f"Image dimensions mismatch for SSIM: {img1.shape} vs {img2.shape}. Resizing {os.path.basename(image_path2)} to match {os.path.basename(image_path1)}.")
        #    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if gray1.shape != gray2.shape: # This should ideally not happen if all frames are resized to a standard size.
            logging.warning(f"Grayscale image dimensions mismatch for SSIM: {gray1.shape} vs {gray2.shape} between {os.path.basename(image_path1)} and {os.path.basename(image_path2)}. This might indicate an issue with frame resizing. Returning 0.0.")
            return 0.0

        score, _ = ssim(gray1, gray2, full=True)
        logging.debug(f"SSIM score between {os.path.basename(image_path1)} and {os.path.basename(image_path2)}: {score:.4f}")
        return score
    except Exception as e:
        logging.error(f"Error calculating SSIM between {os.path.basename(image_path1)} and {os.path.basename(image_path2)}: {e}", exc_info=True)
        return 0.0

def process_advertisements(ads_dir_path, temp_ads_frames_path):
    if not os.path.exists(ads_dir_path):
        logging.warning(f"Ads directory '{ads_dir_path}' not found. SSIM detection will be affected.")
        return []
    os.makedirs(temp_ads_frames_path, exist_ok=True)
    ad_frame_paths = []
    logging.info(f"Processing advertisement videos from: {ads_dir_path}")
    for filename in os.listdir(ads_dir_path):
        if filename.startswith('.'):
            continue
        ad_video_path = os.path.join(ads_dir_path, filename)
        if not os.path.isfile(ad_video_path):
            continue
        frame_filename = f"{os.path.splitext(filename)[0]}_frame.jpg"
        output_image_path = os.path.join(temp_ads_frames_path, frame_filename)
        if extract_first_frame(ad_video_path, output_image_path): # Frames are now resized by extract_first_frame
            logging.info(f"Extracted and resized first frame for ad '{filename}' to '{output_image_path}'")
            ad_frame_paths.append(output_image_path)
        else:
            logging.warning(f"Failed to extract first frame for ad '{filename}' from '{ad_video_path}'")
    if not ad_frame_paths:
        logging.warning(f"No advertisement frames processed from '{ads_dir_path}'.")
    return ad_frame_paths

def get_video_resolution(video_path):
    cap = None
    try:
        if not os.path.exists(video_path):
            logging.warning(f"Video file not found at {video_path} for resolution check.")
            return None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"Could not open video file {video_path} for resolution check.")
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 0 and height > 0:
            return (width, height)
        else:
            logging.warning(f"Video file {video_path} has invalid dimensions (0x0).")
            return None
    except Exception as e:
        logging.error(f"Error getting resolution for {video_path}: {e}", exc_info=True)
        return None
    finally:
        if cap:
            cap.release()

def download_segments(name, m3u8_url, output_subdir_path):
    logging.info(f"Fetching M3U8 playlist for '{name}' from {m3u8_url}")
    playlist_content = None
    for attempt in range(MAX_DOWNLOAD_RETRIES):
        try:
            response = requests.get(m3u8_url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            playlist_content = response.text
            logging.info(f"Successfully fetched M3U8 playlist for '{name}'.")
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logging.warning(f"Timeout/ConnectionError fetching playlist {m3u8_url} for '{name}' (attempt {attempt + 1}/{MAX_DOWNLOAD_RETRIES}). Retrying in {RETRY_SLEEP_DURATION}s... Error: {e}")
            if attempt == MAX_DOWNLOAD_RETRIES - 1:
                logging.error(f"Failed to fetch playlist {m3u8_url} for '{name}' after {MAX_DOWNLOAD_RETRIES} attempts.")
                return []
            time.sleep(RETRY_SLEEP_DURATION)
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching M3U8 playlist for '{name}': {e}", exc_info=True)
            return []

    if not playlist_content:
        logging.error(f"Playlist content is empty for '{name}' after attempting downloads.")
        return []

    segment_urls = []
    lines = playlist_content.splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#EXT-X-ENDLIST'):
            continue
        if line.startswith('#'):
            continue
        if not line.startswith('http'):
            segment_url = urllib.parse.urljoin(m3u8_url, line)
        else:
            segment_url = line
        segment_urls.append(segment_url)

    if not segment_urls:
        logging.warning(f"No segment URLs found in M3U8 playlist for '{name}'.")
        return []
    logging.info(f"Found {len(segment_urls)} segments for '{name}'.")

    first_segment_resolution = None
    advertisement_segments_by_resolution = []

    for i, segment_url in enumerate(segment_urls, 1):
        segment_filename = f"{i:04d}.ts"
        segment_filepath = os.path.join(output_subdir_path, segment_filename)

        if os.path.exists(segment_filepath):
            logging.info(f"Segment {segment_filename} for '{name}' already exists. Skipping download.")
        else:
            segment_downloaded_successfully = False
            for attempt in range(MAX_DOWNLOAD_RETRIES):
                try:
                    logging.info(f"Downloading segment {i}/{len(segment_urls)} for '{name}' (attempt {attempt + 1}/{MAX_DOWNLOAD_RETRIES}): {segment_url}")
                    segment_response = requests.get(segment_url, stream=True, timeout=DEFAULT_TIMEOUT)
                    segment_response.raise_for_status()
                    with open(segment_filepath, 'wb') as f:
                        for chunk in segment_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logging.info(f"Successfully downloaded {segment_filename} for '{name}'.")
                    segment_downloaded_successfully = True
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    logging.warning(f"Timeout/ConnectionError downloading segment {segment_url} for '{name}' (attempt {attempt + 1}/{MAX_DOWNLOAD_RETRIES}). Retrying in {RETRY_SLEEP_DURATION}s... Error: {e}")
                    if attempt == MAX_DOWNLOAD_RETRIES - 1:
                        logging.error(f"Failed to download segment {segment_url} for '{name}' after {MAX_DOWNLOAD_RETRIES} attempts. Skipping segment.")
                    time.sleep(RETRY_SLEEP_DURATION)
                except requests.exceptions.RequestException as e:
                    logging.error(f"Error downloading segment {segment_url} for '{name}': {e}. Skipping segment.", exc_info=True)
                    break
                except Exception as e:
                    logging.error(f"An unexpected error occurred while downloading segment {segment_url} for '{name}': {e}. Skipping segment.", exc_info=True)
                    break

            if not segment_downloaded_successfully:
                continue

        if os.path.exists(segment_filepath):
            current_segment_resolution = get_video_resolution(segment_filepath)
            if current_segment_resolution:
                if first_segment_resolution is None:
                    first_segment_resolution = current_segment_resolution
                    logging.info(f"First segment resolution for '{name}' ({segment_filename}): {first_segment_resolution[0]}x{first_segment_resolution[1]}")
                elif current_segment_resolution != first_segment_resolution:
                    logging.info(f"Resolution mismatch for '{name}' segment {segment_filename}: {current_segment_resolution[0]}x{current_segment_resolution[1]}, expected {first_segment_resolution[0]}x{first_segment_resolution[1]}. Marking as advertisement.")
                    advertisement_segments_by_resolution.append(segment_filepath)
            else:
                logging.warning(f"Could not get resolution for segment {segment_filename} of '{name}'. Skipping resolution check.")
    return advertisement_segments_by_resolution

def merge_segments(stream_name, segments_dir, all_downloaded_segment_files,
                   ads_by_resolution, ads_by_ssim, output_dir_for_merged_file, ffmpeg_path):
    ad_segment_paths = set(ads_by_resolution + ads_by_ssim)
    non_ad_segments_to_merge_relative_paths = []
    logging.info(f"Identifying non-advertisement segments for '{stream_name}'...")
    for segment_filename in all_downloaded_segment_files:
        full_segment_path = os.path.join(segments_dir, segment_filename)
        if full_segment_path not in ad_segment_paths:
            non_ad_segments_to_merge_relative_paths.append(segment_filename)
        else:
            logging.debug(f"Excluding ad segment: {segment_filename} for stream {stream_name}")
    if not non_ad_segments_to_merge_relative_paths:
        logging.info(f"No non-advertisement segments found to merge for '{stream_name}'.")
        return None
    logging.info(f"Found {len(non_ad_segments_to_merge_relative_paths)} non-ad segments to merge for '{stream_name}'.")
    ffmpeg_input_list_path = os.path.join(segments_dir, f"{stream_name}_ffmpeg_input.txt")
    merged_ts_output_path = os.path.join(output_dir_for_merged_file, f"{stream_name}_merged.ts")
    try:
        with open(ffmpeg_input_list_path, 'w') as list_file:
            for rel_path in non_ad_segments_to_merge_relative_paths:
                list_file.write(f"file '{rel_path}'\n")
        logging.debug(f"Created FFmpeg input list: {ffmpeg_input_list_path}")
        ffmpeg_cmd = [
            ffmpeg_path, "-f", "concat", "-safe", "0", "-i", ffmpeg_input_list_path,
            "-c", "copy", "-y", merged_ts_output_path
        ]
        logging.info(f"Running FFmpeg command for '{stream_name}': {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, cwd=segments_dir)
        logging.info(f"Successfully merged non-ad segments for '{stream_name}' to '{merged_ts_output_path}'.")
        logging.debug(f"FFmpeg merge output for {stream_name}:\n{result.stdout}")
        return merged_ts_output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during FFmpeg merge execution for '{stream_name}':")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Return code: {e.returncode}")
        if e.stdout: logging.error(f"Stdout: {e.stdout}")
        if e.stderr: logging.error(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during segment merging for '{stream_name}': {e}", exc_info=True)
        return None
    finally:
        if os.path.exists(ffmpeg_input_list_path):
            try:
                os.remove(ffmpeg_input_list_path)
            except OSError as e:
                logging.warning(f"Error cleaning up FFmpeg input list {ffmpeg_input_list_path}: {e}")

def convert_to_mp4(merged_ts_path, stream_name, output_dir, ffmpeg_path):
    if not os.path.exists(merged_ts_path):
        logging.error(f"Merged TS file not found for MP4 conversion: {merged_ts_path}")
        return None
    mp4_output_path = os.path.join(output_dir, f"{stream_name}.mp4")
    logging.info(f"Converting '{merged_ts_path}' to '{mp4_output_path}'...")
    ffmpeg_cmd = [
        ffmpeg_path, "-i", merged_ts_path, "-c:v", "copy", "-c:a", "aac",
        "-strict", "experimental", "-b:a", "192k", "-y", mp4_output_path
    ]
    try:
        logging.info(f"Running FFmpeg MP4 conversion for '{stream_name}': {' '.join(ffmpeg_cmd)}")
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        logging.info(f"Successfully converted '{stream_name}' to '{mp4_output_path}'.")
        logging.debug(f"FFmpeg convert output for {stream_name}:\n{result.stdout}")
        return mp4_output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"Error during FFmpeg MP4 conversion for '{stream_name}':")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Return code: {e.returncode}")
        if e.stdout: logging.error(f"Stdout: {e.stdout}")
        if e.stderr: logging.error(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during MP4 conversion for '{stream_name}': {e}", exc_info=True)
        return None

def cleanup_files(segments_dir, merged_ts_path, temp_stream_frames_path):
    logging.info(f"Initiating cleanup for stream resources related to '{os.path.basename(segments_dir)}'...")
    if segments_dir and os.path.exists(segments_dir):
        try:
            shutil.rmtree(segments_dir)
            logging.info(f"Cleaned up segment directory: {segments_dir}")
        except Exception as e:
            logging.error(f"Error cleaning up segment directory {segments_dir}: {e}", exc_info=True)
    if merged_ts_path and os.path.exists(merged_ts_path):
        try:
            os.remove(merged_ts_path)
            logging.info(f"Cleaned up merged .ts file: {merged_ts_path}")
        except OSError as e:
            logging.error(f"Error cleaning up merged .ts file {merged_ts_path}: {e}", exc_info=True)
    if temp_stream_frames_path and os.path.exists(temp_stream_frames_path):
        try:
            shutil.rmtree(temp_stream_frames_path)
            logging.info(f"Cleaned up temporary stream frames directory: {temp_stream_frames_path}")
        except Exception as e:
            logging.error(f"Error cleaning up temporary stream frames directory {temp_stream_frames_path}: {e}", exc_info=True)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description="M3U8 Downloader")
    parser.add_argument(
        "--input_file", "-i", required=True, help="Path to the input TXT file")
    parser.add_argument(
        "--output_dir", "-o", required=True, help="Path to the directory where downloaded files will be saved")
    parser.add_argument(
        "--ads_dir", "-a", required=True, help="Path to the directory containing advertisement video segments")
    parser.add_argument(
        "--ssim_threshold", "-s", type=float, default=0.8, help="Float value for SSIM comparison (default: 0.8)")
    parser.add_argument(
        "--ffmpeg_path", default="ffmpeg", help="Path to ffmpeg executable (default: ffmpeg)")
    parser.add_argument(
        "--cleanup", action="store_true", default=False,
        help="Remove all downloaded segment files, the intermediate merged .ts file, and temporary frames after successful MP4 conversion."
    )
    parser.add_argument(
        "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        logging.error(f"Invalid log level: {args.log_level}. Defaulting to INFO.")
        numeric_level = logging.INFO
    logging.getLogger().setLevel(numeric_level)

    logging.info("Starting M3U8 Downloader script.")
    logging.info(f"Input File: {args.input_file}")
    logging.info(f"Output Directory: {args.output_dir}")
    logging.info(f"Ads Directory: {args.ads_dir}")
    logging.info(f"SSIM Threshold: {args.ssim_threshold}")
    logging.info(f"FFmpeg Path: {args.ffmpeg_path}")
    logging.info(f"Cleanup Enabled: {args.cleanup}")
    logging.info(f"Log Level: {args.log_level.upper()}")

    temp_frames_main_dir = os.path.join(args.output_dir, ".temp_frames")
    temp_ads_frames_path = os.path.join(temp_frames_main_dir, "ads")

    if os.path.exists(temp_ads_frames_path):
        logging.info(f"Cleaning up old temporary ad frames from: {temp_ads_frames_path}")
        shutil.rmtree(temp_ads_frames_path, ignore_errors=True)

    ad_frame_paths = process_advertisements(args.ads_dir, temp_ads_frames_path)
    if not ad_frame_paths:
        logging.warning("No ad frames processed. SSIM-based ad detection will be skipped.")

    parsed_data = parse_input_file(args.input_file)
    if not parsed_data:
        logging.critical("No data parsed from input file. Exiting.")
        return
    logging.info(f"Parsed {len(parsed_data)} stream(s) from input file.")

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Main output directory '{args.output_dir}' ensured.")
    except OSError as e:
        logging.critical(f"Error creating main output directory {args.output_dir}: {e}. Exiting.", exc_info=True)
        return

    for name, url in parsed_data:
        logging.info(f"--- Processing stream: {name} ---")
        logging.debug(f"URL for {name}: {url}")
        sub_dir_path = os.path.join(args.output_dir, name)
        current_temp_segment_frames_path = None

        try:
            os.makedirs(sub_dir_path, exist_ok=True)
            logging.debug(f"Ensured stream segment directory exists: {sub_dir_path}")

            ads_by_resolution = download_segments(name, url, sub_dir_path)

            ads_by_ssim = []
            if ad_frame_paths:
                logging.info(f"Starting SSIM based ad detection for '{name}' using {args.ssim_threshold} threshold...")
                current_temp_segment_frames_path = os.path.join(temp_frames_main_dir, name)
                if os.path.exists(current_temp_segment_frames_path):
                    shutil.rmtree(current_temp_segment_frames_path, ignore_errors=True)
                os.makedirs(current_temp_segment_frames_path, exist_ok=True)

                downloaded_segment_files = sorted([
                    f for f in os.listdir(sub_dir_path)
                    if os.path.isfile(os.path.join(sub_dir_path, f)) and f.endswith('.ts')
                ])

                for segment_filename in downloaded_segment_files:
                    segment_file_path = os.path.join(sub_dir_path, segment_filename)
                    if segment_file_path in ads_by_resolution:
                        logging.debug(f"Segment {segment_filename} already marked as ad by resolution. Skipping SSIM check for {name}.")
                        continue
                    segment_frame_basename = f"{os.path.splitext(segment_filename)[0]}.jpg"
                    segment_frame_image_path = os.path.join(current_temp_segment_frames_path, segment_frame_basename)
                    if not extract_first_frame(segment_file_path, segment_frame_image_path): # Frames are resized here
                        logging.warning(f"Could not extract frame from {segment_filename} for SSIM for stream {name}. Skipping.")
                        continue
                    for ad_frame_path in ad_frame_paths: # ad_frame_paths also contain resized frames
                        score = get_ssim_score(segment_frame_image_path, ad_frame_path)
                        if score > args.ssim_threshold:
                            logging.info(f"Segment {segment_filename} for stream '{name}' matches ad {os.path.basename(ad_frame_path)} (SSIM: {score:.4f}). Marking as ad.")
                            ads_by_ssim.append(segment_file_path)
                            break
                if ads_by_ssim:
                    logging.info(f"Segments detected as ads by SSIM for '{name}': {len(ads_by_ssim)}")
                else:
                    logging.info(f"No new segments detected as ads by SSIM for '{name}'.")
            else:
                 current_temp_segment_frames_path = None

            all_segment_filenames = sorted([
                f for f in os.listdir(sub_dir_path)
                if os.path.isfile(os.path.join(sub_dir_path, f)) and f.endswith(".ts")
            ])

            if not all_segment_filenames:
                logging.warning(f"No .ts segments found in {sub_dir_path} to consider for merging for '{name}'.")
            else:
                merged_file_path = merge_segments(
                    name, sub_dir_path, all_segment_filenames,
                    ads_by_resolution, ads_by_ssim,
                    args.output_dir, args.ffmpeg_path
                )
                if merged_file_path:
                    logging.info(f"Successfully merged non-ad segments for '{name}' to: {merged_file_path}")
                    final_mp4_path = convert_to_mp4(merged_file_path, name, args.output_dir, args.ffmpeg_path)
                    if final_mp4_path:
                        logging.info(f"Successfully converted '{name}' to MP4: {final_mp4_path}")
                        if args.cleanup:
                            logging.info(f"Cleanup requested for '{name}'.")
                            cleanup_files(sub_dir_path, merged_file_path, current_temp_segment_frames_path)
                    else:
                        logging.error(f"MP4 conversion failed for '{name}'.")
                else:
                    logging.warning(f"Merging failed or was skipped for '{name}'.")
        except Exception as e:
            logging.critical(f"FATAL ERROR processing stream {name}: {e}", exc_info=True)

    if ad_frame_paths and os.path.exists(temp_ads_frames_path):
        logging.info(f"Cleaning up temporary ad frames directory: {temp_ads_frames_path}")
        try:
            shutil.rmtree(temp_ads_frames_path)
        except Exception as e:
            logging.error(f"Error cleaning up temporary ad frames directory {temp_ads_frames_path}: {e}", exc_info=True)

    logging.info("All processing finished.")

if __name__ == "__main__":
    main()
