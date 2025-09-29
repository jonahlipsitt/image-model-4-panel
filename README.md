# Four Panel AI Image Editing Benchmark

A comprehensive testing suite for evaluating AI image editing models across multiple capabilities. This tool generates four-panel comparison grids showing how different models handle the same editing tasks.

## 🎯 What This Does

Tests 4 AI models simultaneously on image editing tasks:
- **Qwen Image Edit Plus** (Position 1: top-left)
- **Nano Banana** (Position 2: top-right) 
- **Seedream 4** (Position 3: bottom-left)
- **GPT Image 1** (Position 4: bottom-right)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Token
Create a `.env` file with your Replicate API token:
```
REPLICATE_API_TOKEN=your_token_here
```

### 3. Run Tests
```bash
# Run all 150 comprehensive tests (~$52.50)
python3 four_panel_list_slow_edit_fixed.py category_prompts_E.txt --continue-on-error

# Run just 3 tests for quick verification (~$1.05)
python3 four_panel_list_slow_edit_fixed.py category_prompts_E.txt --start-from 1 --max-prompts 3 --continue-on-error
```

## 📊 Test Categories (150 Total Tests)

- **Face Editing** (15 tests) - Hair, eyes, aging, expressions
- **Text & Signage** (20 tests) - Typography, logos, perspective text
- **Object Removal** (15 tests) - Inpainting, seamless removal
- **Background Replacement** (20 tests) - Environmental compositing
- **Color & Material** (15 tests) - Surface transformation
- **Lighting & Atmosphere** (20 tests) - Illumination mastery
- **Style Transfer** (15 tests) - Artistic transformation
- **Object Addition** (15 tests) - Creative object integration
- **Scale & Perspective** (15 tests) - Spatial reasoning
- **Motion & Dynamics** (10 tests) - Movement effects
- **Weather & Environmental** (10 tests) - Atmospheric conditions
- **Bleeding-Edge** (20 tests) - Impossible scenarios & extreme precision

## 📁 Output Structure

Tests create organized folders by category:
```
outputs_categorized/
├── Face_Editing/
│   ├── FE-01_raw.png (reference image + prompt)
│   ├── FE-01_panel_cap.png (4-panel with model names)
│   ├── FE-01_panel_num.png (4-panel with position numbers)
│   └── prompts.txt (list of all prompts in this category)
├── Text_Signage/
├── Object_Removal/
└── ... (12 total categories)
```

## 🔗 Image Assets

All test images are verified working Pexels photos. The `images.txt` file contains the complete asset library with direct links.

## 💰 Cost Estimate

- **Per test**: $0.35 (all 4 models)
- **Full suite**: 150 tests × $0.35 = $52.50
- **Single category**: ~15-20 tests = $5.25-7.00

## 🎛️ Advanced Usage

### Test Specific Categories
```bash
# Face editing only (tests 1-15)
python3 four_panel_list_slow_edit_fixed.py category_prompts_E.txt --start-from 1 --max-prompts 15 --continue-on-error

# Bleeding-edge tests only (tests 131-150) 
python3 four_panel_list_slow_edit_fixed.py category_prompts_E.txt --start-from 131 --max-prompts 20 --continue-on-error
```

### Test Individual Prompts
Use the formula: **Item Number = (Section Number - 1) × 15-20 + Test Number**

Examples:
- **[FE-05]** = Item 5
- **[TX-09]** = Item 24 (15 face + 9)
- **[BE-11]** = Item 141 (15+20+15+20+15+20+15+15+15+10+10 + 11)

## 🧪 Bleeding-Edge Tests

The benchmark includes 20 extreme challenges that push models to their absolute limits:
- Microscopic precision editing
- Impossible physics scenarios  
- Paradoxical visual effects
- Cellular-level detail preservation
- Mathematical accuracy requirements

## 🛠️ Technical Details

- **Models tested**: Qwen Edit Plus, Nano Banana, Seedream 4, GPT Image 1
- **Output resolution**: 1536px tiles (high quality)
- **Fixed panel order**: Consistent positioning for easy comparison
- **Multi-image support**: Handles 1, 2, or 3 reference images per test
- **Error handling**: Continues on failures, logs all issues

## 📋 File Structure

- `category_prompts_E.txt` - 150 comprehensive test cases
- `images.txt` - Verified working image asset library  
- `four_panel_list_slow_edit_fixed.py` - Main test runner
- `model_config_slow_edit.json` - Model configurations
- `src/four_panel.py` - Core image processing engine

## 🤝 Contributing

This benchmark is designed to evaluate and compare AI image editing capabilities. Feel free to:
- Add new test categories
- Extend image asset library
- Improve prompt engineering
- Add new model configurations

## 📄 License

Open source - feel free to use and modify for research and evaluation purposes.
