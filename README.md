AI Floor Plan Analyzer

Live Demo: https://ai-floorplan-analyzer-wzzlz2o4pd3gc22vmcgcjn.streamlit.app/

Last Updated: March 2026

Author: Janu Dhanya Vinoth Singh

Overview

AI Floor Plan Analyzer is an interactive web application that leverages computer vision techniques to automatically detect and classify rooms from architectural floor plan images. This tool provides a quick and automated way to understand floor plan layouts by identifying rooms, drawing bounding boxes, and generating textual summaries — eliminating the need for manual labeling.

The project is implemented using Python, OpenCV, and Streamlit, making it easy to run locally or access directly via a browser.

Try the App Online

No installation required — upload a floor plan image and view the results instantly:
https://ai-floorplan-analyzer-wzzlz2o4pd3gc22vmcgcjn.streamlit.app/

Key Features
Upload floor plan images (PNG, JPG, JPEG)
Automatic room detection using image segmentation
Room classification based on area (Hall, Bedroom, Kitchen, Bathroom)
Visual output with labeled bounding boxes
Textual summary of detected rooms

How It Works
Preprocess Image
Resize and convert images to RGB for consistent processing.
Room Detection
Analyze unique colors to generate masks and extract contours representing room areas.
Area Analysis
Measure detected regions and classify rooms based on size.
Output Visualization
Draw bounding boxes and labels on the image for clear visualization.
Summary Description
Generate a textual description summarizing the number and type of rooms detected.
Local Installation Guide

Clone the repository
git clone https://github.com/janudhanyavinothsingh-bit/AI-FloorPlan-Analyzer.git
cd AI-FloorPlan-Analyzer
Set up Python environment
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
Install dependencies
pip install -r requirements.txt
Run the app
streamlit run app.py

Your browser will open the interactive interface.

Dependencies
Package	Purpose
streamlit	Web app interface
opencv-python-headless	Image processing
numpy	Numerical computing
pandas	Data handling
matplotlib	Visualization
scikit-learn	Basic ML utilities
Pillow	Image file handling
altair & vega-datasets	Optional charting
pytesseract	Optional OCR support

Screenshots
<img width="1920" height="924" alt="Screenshot (194)" src="https://github.com/user-attachments/assets/bc6eedea-56a0-4965-b7bb-a5640a830938" />
<img width="1920" height="903" alt="Screenshot (195)" src="https://github.com/user-attachments/assets/1add4224-bee8-478f-b427-879f3115f10d" />



Future Enhancements
Deep learning-based room segmentation
Integration with OCR text extraction for labels
Support for multi-floor plans
Export results in JSON or PDF formats
 
Acknowledgments

This project uses open-source tools and libraries for rapid prototyping of computer vision solutions. Special thanks to the Python and Streamlit communities for their support and resources.
