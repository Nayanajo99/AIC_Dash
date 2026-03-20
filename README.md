This project is a web-based audio processing application that enhances speech quality by reducing noise. It provides an interactive interface to visualize and compare original and processed audio in real time.




Frontend: Dash (Plotly)
	•	Backend: Python (Flask via Dash)
	•	Audio Processing: Librosa, NumPy, SciPy
	•	Visualization: Plotly
	•	Deployment: Render (Gunicorn)

.
├── app.py                # Main Dash application
├── requirements.txt     # Python dependencies
├── render.yaml          # Deployment configuration
├── assets/
│   └── style.css        # Custom styling
└── data/
    └── clean.wav        # Sample audio file

Installation (Local Setup)

Clone the repositorygit clone : https://github.com/](https://github.com/Nayanajo99/AIC_Dash.git)

Requirements
pip install -r requirements.txt
python app.py
