## AI Floor Plan Analyzer
Professional Internship Project | March 2026

## Live Demo:
https://ai-floorplan-analyzer-wzzlz2o4pd3gc22vmcgcjn.streamlit.app/


## Project Overview
AI Floor Plan Analyzer is a computer vision–based web app that automatically detects and classifies rooms from floor plan images.
	•	Identifies room regions
	•	Classifies spaces based on size
	•	Generates visual and analytical outputs


##  Objective
	•	Automate floor plan interpretation
	•	Reduce manual layout analysis
	•	Apply computer vision to real-world problems


##  Key Features
	•	Upload floor plan images (PNG, JPG, JPEG)
	•	AI-enhanced room detection
	•	Automatic room classification: Living Room, Hall, Bedroom, Kitchen, Bathroom
	•	Real-time statistics & insights
	•	Visual output with bounding boxes
	•	Export results (CSV, Image, TXT)
	•	Analysis history tracking


##  System Workflow

1️⃣ Image Preprocessing
	•	Resize images for performance
	•	Color space conversion (BGR → RGB → LAB)
	•	Histogram equalization

2️⃣ Room Detection
	•	Color quantization & segmentation
	•	Mask generation & morphological operations
	•	Contour detection for room areas

3️⃣ Feature Extraction
	•	Area, Perimeter, Aspect ratio, Circularity

4️⃣ Room Classification
    • Very Large → Living Room
	•	Large → Hall
	•	Medium → Bedroom
	•	Small → Kitchen
	•	Smaller → Bathroom
	•	Minimal → Closet

##  Live Demo
Try it instantly: https://ai-floorplan-analyzer-wzzlz2o4pd3gc22vmcgcjn.streamlit.app/

## 📸 Screenshots 
<img width="1920" height="912" alt="Screenshot (196)" src="https://github.com/user-attachments/assets/c89f8ea9-8eda-461e-9e88-2fcc3e329cdd" />
<img width="1920" height="919" alt="Screenshot (197)" src="https://github.com/user-attachments/assets/9c1ab646-d16d-4e1d-ae07-f009d42288cb" />
<img width="1920" height="921" alt="Screenshot (198)" src="https://github.com/user-attachments/assets/4ba21a65-c3f2-4a0f-9da0-029f320b034d" />
<img width="1920" height="911" alt="Screenshot (199)" src="https://github.com/user-attachments/assets/86fbc16d-a9e5-46f2-856a-dbafbda6b513" />
<img width="1920" height="925" alt="Screenshot (200)" src="https://github.com/user-attachments/assets/0433c42b-aafa-41f7-98d0-ecb763010e6c" />


##  Tech Stack
Python | Streamlit | OpenCV | NumPy | Pandas | Matplotlib | Altair | Pillow | Scikit-learn


## Installation Guide
1. Clone repo  
2. Create virtual environment  
3. Install dependencies  
4. Run the app

   
## Future Enhancements
- Deep learning-based room segmentation  
- OCR integration for room labels  
- Multi-floor support  
- PDF report generation  
- Real-world dimension estimation  
- Cloud API deployment


##  Author
Janu Dhanya Vinoth Singh | B.Com ( CA ) – Data Science with AI


## Acknowledgments
Open-source libraries, Streamlit community, OpenCV


## Internship Note
Professional internship project demonstrating applied computer vision and end-to-end development.
