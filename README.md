# Cloud Type Recognition using CNNs

A deep learning project for semantic segmentation of cloud types from satellite imagery using Convolutional Neural Networks.

## 📌 Overview

Weather forecasting has been crucial to human civilization for millennia. While traditional physics-based simulations have dominated meteorological analysis, machine learning has recently emerged as a powerful complementary approach. This project focuses on one specific application: classifying cloud types from satellite imagery using semantic segmentation.

Since weather patterns are inherently spatial, Convolutional Neural Networks (CNNs) are particularly well-suited for this task. This project serves as a learning playground for:

- **Dataset creation** – Building image and mask datasets from raw data
- **Image augmentation** – Applying transformations to improve model robustness
- **CNN architectures** – Experimenting with different network structures for semantic segmentation

The primary goal is to develop a complete data pipeline and explore various CNN architectures for cloud type classification.

---

## 🗂️ Dataset

This project uses the [Understanding Cloud Organization](https://www.kaggle.com/competitions/understanding_cloud_organization/data) dataset from a 2019 Kaggle competition. 

**Dataset Details:**
- Satellite images of clouds with hand-labeled segmentation masks
- Four cloud types: **Flower**, **Gravel**, **Fish**, and **Sugar**
- Task: Predict segmentation masks for each cloud type given an input image
- Evaluation metric: Dice coefficient (measures pixel-wise prediction accuracy)

---

## ⚙️ Installation

### Prerequisites
- Python 3.7+
- Kaggle account and API credentials
- (Optional) Access to a GPU cluster for training

### Setup Instructions

**1. Clone the repository**
```bash
git clone git@github.com:SATheinen/cloud-type-recognition.git
cd cloud-type-recognition
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Configure Kaggle API**

First, create a Kaggle account and generate API credentials from your [account settings](https://www.kaggle.com/account).

```bash
pip install kaggle
mkdir -p ~/.kaggle
touch ~/.kaggle/kaggle.json
```

Edit `~/.kaggle/kaggle.json` and paste your API credentials:
```json
{
  "username": "your_username",
  "key": "your_api_key"
}
```

Set proper permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

**4. Download the dataset**
```bash
kaggle competitions download -c understanding_cloud_organization
unzip understanding_cloud_organization.zip
```

---

## 🚀 Usage

### Local Execution

**Option 1: Jupyter Notebook**
```bash
jupyter notebook main.ipynb
```

**Option 2: Visual Studio Code**
```bash
code main.ipynb
```

### Cluster Execution

For training on a compute cluster (configurations may need adjustment based on your environment):

**1. Connect to cluster and submit job**
```bash
ssh your_cluster
cd cloud-type-recognition
sbatch job.slurm
```

**2. Create SSH tunnel** (from your local machine)
```bash
ssh -L 8889:localhost:8889 -J user_name@login user_name@node_address
```

**3. Access notebook**

Open `http://localhost:8889` in your browser to access the Jupyter notebook running on cluster resources.

### Configuration

All hyperparameters and settings can be configured in the second cell of `main.ipynb`. To run a full training session:

1. Set your desired parameters
2. Click **Kernel → Restart Kernel and Run All Cells**

### Example Output

```
Epoch: 9
Train loss: 0.7080
Val loss: 0.6830
Dice coefficient: 0.3156
```

---

## ⭐ Features

- 🎭 **Complete data pipeline** – End-to-end image and mask loading
- 🔁 **Heavy augmentation** – Extensive image transformations for robustness
- 💬 **Dynamic loss functions** – Flexible loss calculation for different scenarios
- 🧱 **Debugging tools** – Built-in utilities for development and troubleshooting

---

## 📁 Project Structure

```
cloud-type-recognition/
├── job.slurm                  # SLURM batch script for cluster execution
├── main.ipynb                 # Main training notebook
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── train.csv                  # Training labels and metadata
├── sample_submission.csv      # Example submission format
├── train_images/              # Training images directory
├── train_images_broken/       # Corrupted/invalid images
├── train_images.txt           # List of training image files
└── test_images/               # Test images directory
```

---

## 🔧 Technologies Used

- **Python** – Primary programming language
- **PyTorch** – Deep learning framework
- **Segmentation Models PyTorch** – Pre-built architectures ([GitHub](https://github.com/qubvel-org/segmentation_models.pytorch))
- **Albumentations** – Image augmentation library
- **Jupyter Notebook** – Interactive development environment

---

## 📊 Results

The model successfully learns to segment different cloud types from satellite imagery. Below are examples comparing the original images, ground truth masks, and model predictions:

<table>
  <tr>
    <td align="center"><b>Original Image</b></td>
    <td align="center"><b>Ground Truth Mask</b></td>
    <td align="center"><b>Model Prediction</b></td>
  </tr>
  <tr>
    <td><img src="./example_images/cloud_image.png" alt="Original Satellite Image" width="300"/></td>
    <td><img src="./example_images/cloud_mask.png" alt="Ground Truth Mask" width="300"/></td>
    <td><img src="./example_images/cloud_pred.png" alt="Predicted Mask" width="300"/></td>
  </tr>
  <tr>
    <td align="center"><i>Raw satellite imagery</i></td>
    <td align="center"><i>Hand-labeled segmentation</i></td>
    <td align="center"><i>CNN output</i></td>
  </tr>
</table>

### Model Performance

| Metric | Value |
|--------|-------|
| **Final Train Loss** | 0.7080 |
| **Final Val Loss** | 0.6830 |
| **Dice Coefficient** | 0.3156 |

The model shows promising segmentation capabilities, with room for improvement through hyperparameter tuning and architecture optimization.

---

## 🤝 Contributing

This is primarily a personal learning project, but suggestions and improvements are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is available under the MIT License. See LICENSE file for details.

---

## 🙏 Acknowledgments

- [Kaggle Understanding Cloud Organization Competition](https://www.kaggle.com/competitions/understanding_cloud_organization) for providing the dataset
- [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch) for pre-built architectures (MIT License)

---

## 👤 Author

**Silas Theinen**

- 🔗 GitHub: [@SATheinen](https://github.com/SATheinen)
- 💼 LinkedIn: [Silas Theinen](https://www.linkedin.com/in/silas-theinen-058977358)