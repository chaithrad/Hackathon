import streamlit as st
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import json


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Position and Velocity Estimation of moving RF Targets Dashboard", layout="wide")

st.title("📡 Position and Velocity Estimation of Moving RF Targets")
st.markdown(" RF target motion and position detection using peak feature extraction")

#st.sidebar.button("🔄 Reset App", on_click=st.session_state.clear) 


if st.sidebar.button("🔄 Reset App", key="reset_sidebar"):
    version = st.session_state.get("video_version", 0) + 1
    
    st.session_state.clear()
    
    # ✅ restore version AFTER clearing
    st.session_state.video_version = version


st.sidebar.markdown("---")

# ✅ Title in sidebar
st.sidebar.markdown("## 📡 RF Dashboard")

# ✅ Project Info
st.sidebar.markdown("### 📌 Project Info")
st.sidebar.write("RF Target Localization using TDoA, Doppler and Neural Networks")

# ✅ System Flow
st.sidebar.markdown("### ⚙️ System Flow")
st.sidebar.write("""
TX → 3 Receivers  
Signal Processing  
Doppler  
Neural Network  
Position & Velocity Output
""")

# ✅ Instructions
st.sidebar.markdown("### ▶️ How to Use")
st.sidebar.write("""
1. Start Pipeline  
2. Generate Data  
3. Train Model  
4. Load Model  
5. Run Prediction  
""")

# ✅ Footer
st.sidebar.markdown("---")
st.sidebar.caption("Demo Prototype • RF Localization System")


# ---------------- INTRO SECTION ----------------
st.subheader("🎬 System Overview: RF Signal Localization")

# ✅ Ensure video version exists
if "video_version" not in st.session_state:
    st.session_state.video_version = 0

# ✅ Read video once
with open("demo_introNew.mp4", "rb") as video_file:
    video_bytes = video_file.read()

# ✅ Force Streamlit to reload video
video_bytes = video_bytes + bytes([st.session_state.video_version % 256])

# ✅ Display video
st.video(video_bytes)

st.markdown("---")

# ✅ Start button (with key for safety)
if st.button("▶️ Start Pipeline", key="start_pipeline_btn"):
    st.session_state.start_pipeline = True
    
if st.session_state.get("start_pipeline"):

    #----------------- Datagen------------
    import subprocess
    
    st.subheader("⚙️ Pipeline Control")
    
    if st.button("🧪 Generate Data"):
        st.caption("Demo mode: generates a single synthetic RF sample for quick visualization")
        st.session_state.datagen_output = ""
    
        with st.spinner("Generating data... please wait"):
            result = subprocess.run(
                ["python", "DataGen.py"],
                capture_output=True,
                text=True
            )
    
        st.success("✅ Data Generation Completed")
    
        # ✅ Show output in GUI
        st.subheader("📊 Data Generation Summary")
        st.text(result.stdout)
    
    # ------------- Tranning --------------
    import subprocess
    import re
    
    if st.button("🚀 Train Model"):
      #  st.caption("Demo mode: Lightweight training for demonstration; actual model trained offline with 200 epochs")
    
        st.subheader("📊 Training Progress")
    
        # ✅ Progress bar
        progress = st.progress(0)
    
        # ✅ Live log area
        log_area = st.empty()
    
        # ✅ Summary area
        summary_area = st.empty()
    
        process = subprocess.Popen(
            ["python", "TrainingModelGen.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
    
        output_text = ""
    
        total_epochs = 200   # ✅ change if different
    
        for line in process.stdout:
    
            output_text += line
    
            # ✅ Show ONLY last few important lines (clean logging)
            if ("Epoch" in line or "Loss" in line):
                log_area.text(line.strip())
    
            # ✅ Update progress bar
            if "Epoch" in line:
                try:
                    epoch_num = int(line.strip().split()[1])
                    progress.progress(int(epoch_num / 10 * 100))
                except:
                    pass
    
        process.wait()
    
        st.success("✅ Training Completed")
    
        # ✅ ✅ Extract final summary
        final_text = process.stdout.read() if process.stdout else ""
        full_output = output_text + final_text
    
        # ✅ Extract values using regex
        time_match = re.search(r"Total Training Time: ([\d.]+)", full_output)
        train_loss_match = re.search(r"Final Train Loss: ([\d.]+)", full_output)
        val_loss_match = re.search(r"Final Val Loss: ([\d.]+)", full_output)
        gpu_match = re.search(r"GPU Memory \(Allocated\): ([\d.]+)", full_output)
        epoch_match = re.search(r"Total Epochs: (\d+)", full_output)
        #epoch_match = total_epochs
    
        st.subheader("📊 Training Summary")
    
        col1, col2, col3, col4, col5 = st.columns(5)
        
        if time_match:
            col1.markdown(f"""
            **⏱ Time**  
            ### {float(time_match.group(1)):.2f} sec
            """)
        
        if train_loss_match:
            col2.markdown(f"""
            **📉 Train Loss**  
            ### {train_loss_match.group(1)}
            """)
        
        if val_loss_match:
            col3.markdown(f"""
            **📊 Val Loss**  
            ### {val_loss_match.group(1)}
            """)
        
        if gpu_match:
            col4.markdown(f"""
            **💾 GPU**  
            ### {float(gpu_match.group(1)):.2f} MB
            """)
        
        if epoch_match:
            col5.markdown(f"""
            **🔁 Epochs**  
            ### {epoch_match.group(1)}
            """)
    ### {epoch_match.group(1)}
         ### {epoch_match}
    # ---------------- LOAD DATA ----------------
    @st.cache_data
    def load_data():
        input_data = sio.loadmat('input_file500_50hDSGUI.mat')
        output_data = sio.loadmat('output_file500_50hDSGUI.mat')
    
        X1 = input_data['x1']
        X2 = input_data['x2']
        Y  = output_data['Y']
    
        def extract_peak(x):
            row, col = np.unravel_index(np.argmax(x), x.shape)
            return row, col
    
        features = []
    
        for i in range(len(X1)):
            r1, c1 = extract_peak(X1[i])
            r2, c2 = extract_peak(X2[i])
            features.append([c1, r1, c2, r2])
    
        features = np.array(features).astype(np.float32)
    
        # Normalize features
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
    
        # Normalize outputs
        Y = Y.astype(np.float32)
        Y[:, 0:2] /= 1000
        Y[:, 2:4] /= 100
    
        return features, Y
    
    X, Y = load_data()
    
    # ---------------- DATASET ----------------
    class PeakDataset(torch.utils.data.Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y
    
        def __len__(self):
            return len(self.X)
    
        def __getitem__(self, idx):
            return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])
    
    dataset = PeakDataset(X, Y)
    
    # Use last 20% as test set
    #test_size = int(0.2 * len(dataset))
    #test_ds = torch.utils.data.Subset(dataset, range(len(dataset) - test_size, len(dataset)))
    
    
    total_size = len(dataset)
    
    train_size = int(0.7 * total_size)
    val_size   = int(0.15 * total_size)
    test_size  = total_size - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    
    _, _, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=generator
    )
    
    
    # ---------------- MODEL ----------------
    class PeakNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(4, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 4)
            )
    
        def forward(self, x):
            return self.net(x)
    
    # ---------------- LOAD MODEL ----------------
    @st.cache_resource
    def load_model():
        model = PeakNet()
        model.load_state_dict(torch.load("peaknet_bestGUI.pth", map_location="cpu"))
        model.eval()
        return model
    
    # ---------------- MODEL CONTROL ----------------
    st.subheader("📂 Model Control")
    
    # ✅ Initialize model state
    if "model" not in st.session_state:
        st.session_state.model = None
    
    # ✅ Load model button
    if st.button("📥 Load Model"):
      #  st.caption("Loads pre-trained model (trained on full dataset with 200 epochs)")
        try:
            model = PeakNet()
            model.load_state_dict(torch.load("peaknet_bestGUI.pth", map_location="cpu"))
            model.eval()
    
            st.session_state.model = model
            st.session_state.model_loaded = True
            
            st.success("✅ Model Loaded Successfully")
    
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
    
    
    #model = load_model()
    #st.success("✅ Model Loaded Successfully")
    
    # ---------------- LOSS CURVE ----------------
    if "model_loaded" in st.session_state and st.session_state.model_loaded:
    
        st.subheader("📈 Training Curve")
    
        try:
            train_losses = np.load("train_lossesPeakExt.npy")
            val_losses = np.load("val_lossesPeakExt.npy")
    
            fig, ax = plt.subplots()
            ax.plot(train_losses, label="Train")
            ax.plot(val_losses, label="Validation")
            ax.legend()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True)
    
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.pyplot(fig)
    
        except:
            st.info("Loss curve not available")
    
    
    # ---------------- PREDICTION ----------------
    import time
    
    st.subheader("🔍 Sample Prediction")
    
    idx = st.slider("Select Test Sample", 0, len(test_ds)-1, 0)
    
    if st.button("▶️ Run Prediction"):
    
        model = st.session_state.model
    
        if model is None:
            st.warning("⚠️ Please load the model first")
    
        else:
            # ✅ START TIME
            start_time = time.time()
    
            x_sample, y_true = test_ds[idx]
    
            with torch.no_grad():
                pred = model(x_sample.unsqueeze(0)).numpy()[0]
    
            # ✅ END TIME
            latency = (time.time() - start_time) * 1000  # ms
            throughput = 1000 / latency
    
            # ✅ Display latency + throughput (SAFE)
            st.write(f"⚡ End-to-End Latency: {latency:.2f} ms")
            st.write(f"🚀 Throughput: {throughput:.1f} samples/sec")
    
            # ✅ Real-time badge
            if latency < 10:
                st.success("✅ Real-time capable (<10 ms)")
            else:
                st.warning("⚠️ Not real-time")
    
            # De-normalize
            pred[0:2] *= 1000
            pred[2:4] *= 100
    
            y_true = y_true.numpy()
            y_true[0:2] *= 1000
            y_true[2:4] *= 100
    
            # Split
            true_pos = y_true[0:2]
            true_vel = y_true[2:4]
    
            pred_pos = pred[0:2]
            pred_vel = pred[2:4]
    
            # Table
            data = {
                "Parameter": ["X (m)", "Y (m)", "Vx (m/s)", "Vy (m/s)"],
                "Ground Truth": [
                    round(true_pos[0], 2),
                    round(true_pos[1], 2),
                    round(true_vel[0], 2),
                    round(true_vel[1], 2)
                ],
                "Prediction": [
                    round(pred_pos[0], 2),
                    round(pred_pos[1], 2),
                    round(pred_vel[0], 2),
                    round(pred_vel[1], 2)
                ]
            }
    
            st.subheader("📊 Prediction Comparison")
            st.table(data)
    
            # ✅ Geometry plot (UNCHANGED)
            st.subheader("📡 RF Geometry with Signal Paths")
    
            rx_positions = [(0, 0), (1000, 0), (500, 800)]
    
            fig, ax = plt.subplots()
    
            for i, (rx, ry) in enumerate(rx_positions):
                ax.scatter(rx, ry, color='green')
                ax.text(rx, ry, f"RX{i+1}")
    
            ax.scatter(true_pos[0], true_pos[1], color='blue', label='True TX')
            ax.scatter(pred_pos[0], pred_pos[1], color='red', label='Predicted TX')
    
            for (rx, ry) in rx_positions:
                ax.plot([true_pos[0], rx], [true_pos[1], ry], 'b--', alpha=0.5)
    
            for (rx, ry) in rx_positions:
                ax.plot([pred_pos[0], rx], [pred_pos[1], ry], 'r-', alpha=0.5)
    
            ax.plot(
                [true_pos[0], pred_pos[0]],
                [true_pos[1], pred_pos[1]],
                'k--',
                label='Prediction Error'
            )
    
            ax.set_title("RF Signal Geometry")
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            ax.legend()
            ax.grid(True)
    
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.pyplot(fig)
    
    # ---------------- PERFORMANCE ----------------
    st.subheader("📊 Model Performance")
    
    #if st.button("Compute Performance"):
    model = st.session_state.model
    
    if model is None:
        st.warning("⚠️ Please load the model first")
    
    elif st.button("Compute Performance"):
    
        errors_pos = []
        errors_vel = []
    
        with torch.no_grad():
            for x_sample, y_true in test_ds:
    
                pred = model(x_sample.unsqueeze(0)).numpy()[0]
    
                pred[0:2] *= 1000
                pred[2:4] *= 100
    
                y_true = y_true.numpy()
                y_true[0:2] *= 1000
                y_true[2:4] *= 100
    
                pos_error = np.linalg.norm(pred[0:2] - y_true[0:2])
                vel_error = np.linalg.norm(pred[2:4] - y_true[2:4])
    
                errors_pos.append(pos_error)
                errors_vel.append(vel_error)
    
        st.write(f"📍 Position Error: {np.mean(errors_pos):.2f} m")
        st.write(f"🚀 Velocity Error: {np.mean(errors_vel):.2f} m/s")
    
    # ---------------- ERROR DISTRIBUTION ----------------
    st.subheader("📊 Error Distribution")
    
    #if st.button("Show Error Distribution"):
    model = st.session_state.model
    
    if model is None:
        st.warning("⚠️ Please load the model first")
    
    elif st.button("Show Error Distribution"):
    
        errors_pos = []
        errors_vel = []
    
        with torch.no_grad():
            for x_sample, y_true in test_ds:
    
                pred = model(x_sample.unsqueeze(0)).numpy()[0]
    
                pred[0:2] *= 1000
                pred[2:4] *= 100
    
                y_true = y_true.numpy()
                y_true[0:2] *= 1000
                y_true[2:4] *= 100
    
                pos_error = np.linalg.norm(pred[0:2] - y_true[0:2])
                vel_error = np.linalg.norm(pred[2:4] - y_true[2:4])
    
                errors_pos.append(pos_error)
                errors_vel.append(vel_error)
    
        fig, ax = plt.subplots(1, 2, figsize=(10,4))
    
        # Position error
        ax[0].hist(errors_pos, bins=20, color='blue')
        ax[0].axvline(np.mean(errors_pos), color='red', label='Mean')
        ax[0].set_title("Position Error (m)")
        ax[0].set_title("Position Error (m)")
        ax[0].set_xlabel("Position Error (meters)")
        ax[0].set_ylabel("Number of Samples")
        ax[0].legend()
    
        # Velocity error
        ax[1].hist(errors_vel, bins=20, color='green')
        ax[1].axvline(np.mean(errors_vel), color='red', label='Mean')
        ax[1].set_title("Velocity Error (m/s)")
        ax[1].set_title("Velocity Error (m/s)")
        ax[1].set_xlabel("Velocity Error (m/s)")
        ax[1].set_ylabel("Number of Samples")
        ax[1].legend()
    
        plt.tight_layout()
    
        st.pyplot(fig)
    
    
        
