RF Target Position and Velocity Estimation using Neural Networks

Overview
========
This project presents a deep learning approach for estimating the position and velocity of 
a moving RF (Radio Frequency) target using Doppler signal data.
A neural network model is trained on simulated Doppler datasets and predicts the target's 
position and velocity. A Streamlit-based graphical user interface (GUI) is provided for 
easy interaction and visualization of the complete prediction workflow.

Features
========
- Doppler signal dataset generation
- Neural network-based target localization
- Position estimation
- Velocity estimation
- Interactive Streamlit GUI
- Pre-trained model included
- Complete training pipeline

Repository Contents
===================
├── DataGen.py                  # Generates Doppler signal datasets
├── TrainingModelGen.py         # Neural network training script
├── app.py                      # Streamlit GUI application
├── inputt_file500_50hDSGUI.mat # Generated Doppler dataset(this can be generated uisng DataGen.py)
├── output_file500_50hDSGUI.mat # Generated transmitter data for generated Doppler dataset
├── peaknet_bestGUI.pth         # Trained PyTorch model
├── train_lossesPeakExt.npy     # Training loss values
├── val_lossesPeakExt.npy       # Validation loss values
├── demo_IntroNew.mp4           # Use this video for GUI
├── DemoVideo.mp4               # Project demonstration
└── README.md

 Dataset
 =======
The project uses simulated Doppler signal data.
- Number of datasets: 500
- Training Split: 70%
- Validation Split: 15%
- Testing Split: 15%

Technologies Used
=================
- Python
- PyTorch
- NumPy
- SciPy
- Streamlit
- Matplotlib
- MATLAB (.mat Dataset)

Project Workflow
=================
1. Generate Doppler signal dataset.
2. Preprocess the generated data.
3. Split data into training, validation, and testing sets.
4. Train the neural network.
5. Save the trained model.
6. Load the model using the GUI.
7. Predict the target position and velocity.

Running the Project
===================
### Clone the repository
```bash
git clone https://github.com/yourusername/yourrepository.git
```
### Install dependencies
```bash
pip install torch numpy scipy matplotlib streamlit
```
### Launch the application
```bash
streamlit run app.py
```

Results
=======
The trained neural network successfully estimates:

- Target Position
- Target Velocity

The GUI provides an easy interface to generate predictions using the trained model.

Future Improvements
========================

- Real-time RF target tracking
- Multi-target localization
- CNN/LSTM-based architectures
- Improved localization accuracy
- Deployment on embedded hardware

Demo
====
A complete demonstration video is included in this repository.
**Demo Video:** `DemoVideo.mp4`

Author
======
Koyel Das
Chaithra D

- MATLAB
- Python
- Wireless Communications
- Machine Learning
- Deep Learning


## License

This project is intended for educational and research purposes.
