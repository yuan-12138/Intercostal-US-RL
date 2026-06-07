# Reinforcement Learning Based Acoustic Window Planning for Intercostal Liver Ultrasound Scan

This repository contains the code for the paper:

**Reinforcement Learning Based Acoustic Window Planning for Intercostal Liver Ultrasound Scan** 

## Data download
The trained policies and the blender context file can be downloaded from https://drive.google.com/file/d/1fj0ZFA3JapaNW1-o54-QwUCnHgStwZMr/view?usp=drive_link

## Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/sNiper-Qian/robus.git
   cd robus
    ```
2. Create virtual environment and install the dependencies:
    ```bash
    source setup.sh
    ```
3. For training the model:
    ```bash
    python main.py --log_path='experiment/train_folder/'
    ```
4. For testing the model without visualization:
    ```bash
    python main.py --mode='test' --checkpoint_path='experiment/train_folder/checkpoint.pth'
    ```
5. For testing the model with visualization:

    Firstly, change the value of 'RENDER' to True in config.yaml file 
    
    Secondly, change the checkpoint path to the path of the checkpoint file and the video folder id in the main function of tester.py file
    
    Finally, open context.blender in Blender and run the following code in the blender console:
    ```bash
    python tester.py
    ```

## Demo Video

To see our project in action, watch the demo video:

[![Demo Video](http://img.youtube.com/vi/VIDEO_ID/0.jpg)](http://www.youtube.com/watch?v=VIDEO_ID "Demo Video Title")


## Acknowledgements
We would like to thank the authors of the following repositories:
- [Tianshou](https://github.com/thu-ml/tianshou)
    
## Contact
For any questions, feedback, or discussions related to this project, please reach out to:

- **Cheng Qian:** cheng.qian@tum.de

- **Yuan Bi:** yuan.bi@tum.de

