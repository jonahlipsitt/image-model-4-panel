#!/usr/bin/env python3
"""
Four Panel List Generator (Slow Edit Fixed Order) - Batch editing with fixed panel placement

This script uses a fixed, non-randomized panel order for image editing:
- Position 1 (top-left): Qwen Image Edit Plus
- Position 2 (top-right): Nano Banana
- Position 3 (bottom-left): Seedream 4
- Position 4 (bottom-right): GPT Image 1

Usage:
    python3 four_panel_list_slow_edit_fixed.py prompts.txt --image "image_url" --start-from 1 --max-prompts 10 --continue-on-error
"""

import sys
import time
import argparse
from pathlib import Path

# Import everything from the main module
sys.path.insert(0, str(Path(__file__).parent / "src"))
from four_panel import *

def parse_list_args():
    """Parse command line arguments for list processing."""
    parser = argparse.ArgumentParser(description="Batch generate four-panel editing collages with fixed panel order")
    parser.add_argument("prompts_file", help="Text file containing prompts (one per line)")
    parser.add_argument("--image", help="Image URL or local file path to edit (optional for text-only prompts)")
    parser.add_argument("--start-from", type=int, default=1, help="Line number to start from (1-based)")
    parser.add_argument("--max-prompts", type=int, help="Maximum number of prompts to process")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing if a prompt fails")
    parser.add_argument("--delay", type=float, default=0, help="Delay between prompts in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output-dir", type=Path, help="Output directory for generated images")
    return parser.parse_args()

def generate_collage_fixed_order_edit(
    prompt: str,
    all_images: List[str],
    config: Dict[str, Any],
    output_dir: Path,
    collage_name: str,
    tile_size: int,
    margin: int,
    background_color: str,
) -> Tuple[Path, Path]:
    """Generate editing collage with fixed model order using the existing generate_collage function."""
    
    # Use the existing generate_collage function but override model selection
    # This ensures proper .env loading and API handling
    
    # Fixed model order for editing
    fixed_models = [
        "qwen_edit_plus",    # Position 1 (top-left)
        "nano_latest",       # Position 2 (top-right) 
        "seedream_latest",   # Position 3 (bottom-left)
        "gpt_image_1"        # Position 4 (bottom-right)
    ]
    
    catalog = config.get("catalog", [])
    catalog_by_id = {entry.get("id"): entry for entry in catalog}
    
    # Get the specific model specs in our fixed order
    model_specs = []
    for model_id in fixed_models:
        if model_id in catalog_by_id:
            model_specs.append(catalog_by_id[model_id])
        else:
            LOGGER.error(f"Model {model_id} not found in catalog")
            raise ValueError(f"Model {model_id} not found")
    
    LOGGER.info("Fixed edit model order: %s", [spec.get("name", spec.get("id")) for spec in model_specs])
    
    # Use the existing generate_collage function but inject our fixed model specs
    # Temporarily modify the function to skip randomization
    
    import four_panel
    original_random_shuffle = getattr(four_panel.random, 'shuffle', None)
    
    # Disable shuffling temporarily
    def no_shuffle(x):
        pass
    
    if hasattr(four_panel.random, 'shuffle'):
        four_panel.random.shuffle = no_shuffle
    
    try:
        # Call the existing function with our model specs and multiple images
        # Convert list back to comma-separated string for the generate_collage function
        image_ref_string = ','.join(all_images) if len(all_images) > 1 else (all_images[0] if all_images else None)
        
        collage_path, numbered_path, panels = generate_collage(
            prompt,
            model_specs,
            config=config,
            allow_input=False,
            editing=bool(all_images),
            image_reference=image_ref_string,
            output_dir=output_dir,
            collage_name=collage_name,
            tile_size=tile_size,
            margin=margin,
            background_color=background_color,
            panel_count=4,
        )
        return collage_path, numbered_path
    finally:
        # Restore original shuffle function
        if original_random_shuffle:
            four_panel.random.shuffle = original_random_shuffle

def get_category_info(test_id):
    """Get category folder name and description from test ID."""
    category_map = {
        'FE': ('Face_Editing', 'FACE EDITING & IDENTITY'),
        'TX': ('Text_Signage', 'TEXT & SIGNAGE EDITING'),
        'RM': ('Object_Removal', 'OBJECT REMOVAL & INPAINTING'),
        'BG': ('Background_Replacement', 'BACKGROUND REPLACEMENT & SCENE COMPOSITION'),
        'CM': ('Color_Material', 'COLOR & MATERIAL TRANSFORMATION'),
        'LS': ('Lighting_Atmosphere', 'LIGHTING & ATMOSPHERE'),
        'ST': ('Style_Transfer', 'STYLE TRANSFER & ARTISTIC TRANSFORMATION'),
        'OA': ('Object_Addition', 'OBJECT ADDITION & ENHANCEMENT'),
        'SP': ('Scale_Perspective', 'SCALE & PERSPECTIVE MANIPULATION'),
        'MD': ('Motion_Dynamics', 'MOTION & DYNAMICS'),
        'WE': ('Weather_Environmental', 'WEATHER & ENVIRONMENTAL EFFECTS'),
        'CO': ('Complex_Composition', 'COMPLEX MULTI-OBJECT COMPOSITION'),
        'BE': ('Bleeding_Edge', 'BLEEDING-EDGE PRECISION TESTS'),
        'TD': ('Temporal_Paradox', 'TEMPORAL & DIMENSIONAL PARADOX'),
    }
    
    prefix = test_id.split('-')[0] if '-' in test_id else test_id[:2]
    return category_map.get(prefix, ('Other', 'OTHER TESTS'))

def main_list_slow_edit_fixed():
    """Main function for batch editing processing with fixed model order."""
    args = parse_list_args()
    configure_logging(args.verbose)
    
    # Use Slow Edit configuration
    config_path = Path("model_config_slow_edit.json")
    if not config_path.exists():
        LOGGER.error(f'Slow Edit config file not found: {config_path}')
        return 1

    config = load_config(config_path)
    config["_config_path"] = str(config_path.resolve())
    
    catalog = config.get("catalog", [])
    if not catalog:
        LOGGER.error("Configuration catalog is empty; add at least one model entry")
        return 1
    
    # Create main outputs directory
    base_output_dir = args.output_dir or Path("outputs_categorized")
    base_output_dir.mkdir(exist_ok=True)
    
    # Read prompts from file
    prompts_file = Path(args.prompts_file)
    if not prompts_file.exists():
        LOGGER.error(f"Prompts file not found: {prompts_file}")
        return 1
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    # Parse category_prompts_C.txt format or fallback to simple format
    prompts = []
    assets = {}
    
    # First pass: collect asset definitions
    for line in lines:
        if ' -> ' in line and line.count('http') == 1:
            # Asset definition: ALIAS -> URL
            alias, url = line.split(' -> ', 1)
            assets[alias.strip()] = url.strip()
    
    # Second pass: collect test prompts
    for line in lines:
        if line.startswith('[') and ' edit: ' in line:
            # Category C format: [TEST-ID] [n-image] images: ALIASES edit: PROMPT
            try:
                parts = line.split(' edit: ', 1)
                if len(parts) == 2:
                    test_part = parts[0]  # [FE-01] [1-image] images: PORTRAIT_WOMAN
                    clean_prompt = parts[1].strip()  # Just the actual editing instruction
                    
                    # Extract image aliases
                    if 'images: ' in test_part:
                        image_part = test_part.split('images: ')[1]
                        aliases = [alias.strip() for alias in image_part.split(' | ')]
                        
                        # Convert aliases to URLs and create prompt with image references
                        image_urls = []
                        for alias in aliases:
                            if alias in assets:
                                image_urls.append(assets[alias])
                            else:
                                LOGGER.warning(f"Unknown asset alias: {alias}")
                        
                        # Create structured prompt data
                        prompt_data = {
                            'text': clean_prompt,
                            'images': image_urls,
                            'test_id': test_part.split(']')[0][1:] if ']' in test_part else 'Unknown'
                        }
                        prompts.append(prompt_data)
                
            except Exception as e:
                LOGGER.warning(f"Failed to parse line: {line[:50]}... Error: {e}")
                
        elif '\t' in line and not ' -> ' in line:
            # Tab-separated format: prompt<TAB>use_image<TAB>image_url (but not asset definitions)
            parts = line.split('\t')
            if len(parts) >= 1:
                prompt_data = {
                    'text': parts[0].strip(),
                    'images': [parts[2].strip()] if len(parts) > 2 and parts[2].strip() else [],
                    'test_id': 'TAB'
                }
                prompts.append(prompt_data)
    
    if not prompts:
        LOGGER.error("No prompts found in file")
        return 1
    
    # Calculate range
    start_idx = args.start_from - 1  # Convert to 0-based
    if start_idx < 0 or start_idx >= len(prompts):
        LOGGER.error(f"Start line {args.start_from} is out of range (1-{len(prompts)})")
        return 1
    
    end_idx = len(prompts)
    if args.max_prompts:
        end_idx = min(start_idx + args.max_prompts, len(prompts))
    
    prompts_to_process = prompts[start_idx:end_idx]
    
    LOGGER.info(f"Processing {len(prompts_to_process)} editing prompts (items {start_idx + 1}-{end_idx}) from {prompts_file}")
    LOGGER.info(f"Found {len(assets)} asset definitions")
    if args.image:
        LOGGER.info(f"Override image reference: {args.image}")
    LOGGER.info("Fixed edit model order: Qwen Edit Plus (1), Nano Banana (2), Seedream 4 (3), GPT Image 1 (4)")
    
    composition_cfg = config.get("composition", {})
    tile_size = composition_cfg.get("tile_size", 1024)
    margin = composition_cfg.get("margin", 8)
    background_color = composition_cfg.get("background_color", "#101012")
    
    success_count = 0
    error_count = 0
    
    for i, prompt_data in enumerate(prompts_to_process, 1):
        current_item = start_idx + i
        
        # Extract prompt text and images
        if isinstance(prompt_data, dict):
            prompt_text = prompt_data['text']
            prompt_images = prompt_data['images']
            test_id = prompt_data['test_id']
        else:
            # Fallback for old format
            prompt_text = str(prompt_data)
            prompt_images = []
            test_id = 'LEGACY'
        
        # Use override image if provided, otherwise use all images from prompt
        if args.image:
            image_reference = args.image
            all_images = [args.image]
        else:
            image_reference = prompt_images[0] if prompt_images else None
            all_images = prompt_images
        
        LOGGER.info(f"Processing edit prompt {i}/{len(prompts_to_process)} (item {current_item}) [{test_id}]: {prompt_text[:80]}...")
        if all_images:
            LOGGER.info(f"Using {len(all_images)} image(s): {', '.join(img[:50] + '...' for img in all_images)}")
        else:
            LOGGER.info("No image reference - using text-only mode")
        
        try:
            # Get category info and create category folder
            category_folder, category_desc = get_category_info(test_id)
            category_dir = base_output_dir / category_folder
            category_dir.mkdir(exist_ok=True)
            
            # Create category prompt list file if it doesn't exist
            prompt_list_file = category_dir / "prompts.txt"
            if not prompt_list_file.exists():
                with open(prompt_list_file, 'w') as f:
                    f.write(f"# {category_desc}\n")
                    f.write(f"# Generated from {args.prompts_file}\n\n")
            
            # Add this prompt to the category file
            with open(prompt_list_file, 'a') as f:
                f.write(f"{test_id}: {prompt_text}\n")
            
            # Create clean filenames with new naming scheme
            if test_id != 'LEGACY':
                base_name = test_id
            else:
                base_name = f"LEGACY-{i:03d}"
            
            ref_name = f"{base_name}_raw.png"
            collage_name = f"{base_name}_panel_cap.png"
            numbered_name = f"{base_name}_panel_num.png"
            
            # Generate reference slide first
            if prompt_images:
                ref_path = category_dir / ref_name
                compose_reference_slide(
                    prompt_images,
                    prompt_text,
                    ref_path,
                    tile_size,
                    margin,
                    background_color,
                )
                LOGGER.info(f"Generated reference slide: {ref_path}")
            
            # Generate four-panel results in category folder
            temp_collage_name = f"temp_{base_name}_panel_cap.png"
            collage_path, numbered_path = generate_collage_fixed_order_edit(
                prompt_text,  # Use clean prompt text
                all_images,   # Pass all images for multi-image support
                config,
                category_dir,  # Save to category folder
                temp_collage_name,
                tile_size,
                margin,
                background_color,
            )
            
            # Rename files to proper naming scheme
            final_collage_path = category_dir / collage_name
            final_numbered_path = category_dir / numbered_name
            
            if collage_path.exists():
                collage_path.rename(final_collage_path)
            if numbered_path.exists():
                numbered_path.rename(final_numbered_path)
            
            LOGGER.info(f"Generated in {category_folder}/: {base_name}_raw.png, {base_name}_panel_cap.png, {base_name}_panel_num.png")
            success_count += 1
            
        except Exception as exc:
            error_count += 1
            LOGGER.error(f"Error processing edit prompt {i} (item {current_item}) [{test_id}]: {exc}")
            if not args.continue_on_error:
                return 1
        
        # Delay between prompts if specified
        if args.delay > 0 and i < len(prompts_to_process):
            time.sleep(args.delay)
    
    LOGGER.info(f"Batch edit processing complete: {success_count} successful, {error_count} errors")
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main_list_slow_edit_fixed())
