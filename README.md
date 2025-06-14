# 🍍 Ananas AI Nutrition Coach

An AI-powered nutrition coach that analyzes food images, provides nutritional insights, and matches meals with your mood. Built with Streamlit, Imagga Vision API, USDA FoodData Central API, and Ollama for AI-powered explanations.

## Features

- 📸 Food Image Recognition using Imagga Vision API
- 📊 Nutritional Analysis with USDA FoodData Central API
- 🧠 AI-powered explanations using Ollama (llama3 model)
- ⚡ Real-time streaming responses
- 🎯 Mood-based Food Recommendations
- 🛒 Smart Grocery List Generation
- 🚀 Mock Instacart Integration

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables in `.env`:
   - Imagga credentials (IMAGGA_API_KEY, IMAGGA_API_SECRET)
   - USDA API key (USDA_API_KEY) - Get one at [data.gov](https://api.data.gov/signup/)
   - Ollama (install locally from [ollama.ai](https://ollama.ai/))

## Running the App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Usage

1. Select your current mood from the Food Mood Matching section
2. Upload a photo of your meal
3. View detected foods and nutritional analysis
4. Get mood-based meal recommendations
5. Generate and manage your grocery list

## API Dependencies

- [Imagga Vision API](https://imagga.com/) for food detection
- [USDA FoodData Central API](https://fdc.nal.usda.gov/) for nutritional analysis - Free, government-verified nutrition data

## Docker Support

Set `DOCKER_MODE=true` in `.env` when running in a container.

## Note

This is a prototype developed for the lablab.ai Hackathon. Some features use mock data for demonstration purposes.