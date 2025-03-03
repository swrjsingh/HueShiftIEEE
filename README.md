# Hue Shift - IEEE Executive Project (C2)

A Flask-based web application for colorizing grayscale images and videos! We use two approaches - A diffusion based approach, and a GAN based approach

### Reference Papers
 - [1] NoGAN Based Implementation - [DeOldify](https://github.com/jantic/DeOldify?tab=readme-ov-file#what-is-nogan)
 - [2] Colorization of black-and-white images using deep neural networks (PDF) - [Link](https://core.ac.uk/download/pdf/151072499.pdf)
 - [3] Video Colorization with Pre-trained Text-to-Image Diffusion (PDF) - [Link](https://arxiv.org/pdf/2306.01732v1)
 - [4] Deep Exemplar-Based Video Coloring (PDF) - [Link](https://arxiv.org/pdf/1906.09909v1)
 - [5] Diffusion based image coloring using Piggybacked models (PDF) - [Link](https://arxiv.org/pdf/2304.11105)
 - [6] SwinTExCo: Exemplar-based video colorization using Swin Transformer (PDF) - [Link](https://www.sciencedirect.com/science/article/pii/S0957417424023042)

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the application:
   ```
   python app.py
   ```
6. Open your browser and navigate to `http://127.0.0.1:5000/`

## Team

### Mentors
- Aditya Ubaradka
- Aishini Bhattacharjee
- Hemang Jamadagni
- Sree Dakshinya

### Mentees
- Akhil Sakhtieswaran
- Swaraj Singh
- Vanshika Mittal
