# Open-EMMA 

**Open-EMMA** is an open-source end-to-end framework for motion palnning in autonomous vehicles. **Open-EMMA** leverages the pretrained world knowledge of Multimodal Large Language Models, integrating text and front camera inputs to enable accurate future ego waypoint predictions and decision rationales. Our goal is to provide accessible tools for researchers and developers to advance autonomous driving research and applications.

### Installation  
To get started with Open-EMMA, follow these steps to set up your environment and dependencies.

1. **Environment Setup**  
   Set up a Conda environment for Open-EMMA with Python 3.8:
   ```bash
   conda create -n openemma-env python=3.8
   conda activate openemma-env
   ```

2. **Clone Open-EMMA Repository**   
    Clone the Open-EMMA repository and navigate to the root directory:
    ```bash
    git clone git@github.com:dummy4submission/OpenEMMA.git
    cd Open-EMMA
    ```

3. **Install Dependencies**  
    Ensure you have cudatoolkit installed. If not, use the following command:
    ```bash
    conda install nvidia/label/cuda-12.4.0::cuda-toolkit
    ```
    To install the core packages required for Open-EMMA, run the following command:
    ```bash
    pip install -r requirements.txt
    ```
    This will install all dependencies, including those for YOLO-3D, an external tool used for critical object detection. The weights needed to run YOLO-3D will be automatically downloaded during the first execution.

4. **Set up GPT-4 API Access**  
    To enable GPT-4’s reasoning capabilities, obtain an API key from OpenAI. You can add your API key directly in the code where prompted or set it up as an environment variable:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```
    This allows Open-EMMA to access GPT-4 for generating future waypoints and decision rationales.



### Usage  
After setting up the environment, you can start using Open-EMMA with the following instructions:

1. **Prepare Input Data**   
    Download and extract the [nuScenes dataset](https://www.nuscenes.org/nuscenes#download)
    
2. **Run Open-EMMA**  
    Use the following command to execute Open-EMMA's main script:
    ```bash
    python main.py \
        --model-path qwen \
        --dataroot [dir-of-nuscnse-dataset] \
        --version [vesion-of-nuscnse-dataset] \
        --method openemma
    ```

3. **Output Interpretation**   
    After running the model, Open-EMMA generates the following output in the `./qwen-reults` location:

    - **Waypoints**: A list of future waypoints predicting the ego vehicle’s trajectory.

    - **Decision Rationales**: Text explanations of the model’s reasoning, including scene context, critical objects, and behavior decisions.

    - **Annotated Images**: Visualizations of the planned trajectory and detected critical objects overlaid on the original images.

    - **Compiled Video**: A video (e.g., `output_video.mp4`) created from the annotated images, showing the predicted path over time.

