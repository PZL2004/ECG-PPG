**Problem Statement**

Those who do not have access to nearby hospitals should have easy access to ECG and PPG signals in order to diagnose disease, heart problems, deficiency in O2 concentrations, and [other related things] so they can begin to treat themselves and recover from sickness.

**Application Background**

Photoplethysmography, or PPG, (“light volume recording”) is a method used to measure minute changes in blood volume using absorption, scattering, and transmission of the human body under a light. These changes are caused by blood flow from the heart, making it a valuable tool to analyze heart rate data and blood concentration in the human body. PPG is often used on the ends of the body such as fingers because the volume of the vascular bed is high, making for an easy signal to read. PPG is often used to measure blood oxygen saturation (PulseOx), peripheral blood flow, and peripheral vascular tone and so it can be used to address hypertension, coronary artery disease, respiratory distress, heart failure and evaluation, and arterial blood estimation. All of these make PPG a valuable asset to any person’s health [1].
Electrocardiography, or ECG, is used extensively in coronary care and diagnosis of cardiovascular health. Surface electrodes are used to detect an electrical pulse associated with each heartbeat, characterized by muscular contraction and pushing of blood. With these measurements, each patient’s physiological status over time can be monitored for any unusual or detrimental cardiac activity [2].

**Methods**

*Python Libraries:*

NumPy: Mathematical Computation

SciPy: Optimizations, Integration, Statistics

Pandas: Data Analysis, Data Structures

HeartPy: PPG, ECG, and noisy ECG analysis [3]

Matplotlib: graphs

datetime: datetime conversions

sys: error handling

time: compute time

*Hardware:*

Maxim Integrated MAX86150 Evaluation Kit; PPG/ECG data gathering

MAX32625 PICO Board Developer (PICOBD); burning code to Arduino on Maxim Integrated MAX86150 Evaluation Kit

2 USB-C to Micro-USB Power Cables; power MAX86150 and MAX32625 PICO Board Developer

LiPo 2200 mAh Battery; powers MAX86150 when not plugged in

10-pin 2X5 Socket-Socket Cable; for connecting PICOBD to the Arduino on the Maxim Integrated MAX86150 Evaluation Kit

*Software:*

Maxim DeviceStudio5: collection of data and exporting to .csv file for analysis

Visual Studio Code (VSCode): Python, C++

Jupyter Notebook: Python

Mbed Studio: C++

**Data and Resource Overview**

The primary source of the data used  will be the primary investigators themselves. They will use the Maxim Integrated Evaluation Kit to measure their own ECG and PPG data. Use of data and code from computing class projects may be used for additional testing and development. Reference the Hardware and Software sections of the Methods for more information.

**Expected Results**

The end goal of this project, and, therefore, what is expected to be the final result, is the creation of a handheld, pocket-size PPG and ECG monitor that can be used outside of the hospital by patients unfamiliar with the software in order to diagnose health issues. Additionally, a simple code for a tool using C++, as well as an analytical Python code should be developed.

**References**

[1] J. Park, H. Seok Seok, S.-S. Kim, and H. Shin, “Photoplethysmogram analysis and applications: An integrative review,” Frontiers in Physiology, https://pubmed.ncbi.nlm.nih.gov/35300400/ (accessed Sep. 27, 2023). 

[2] P. R. E. Harris, “The normal electrocardiogram: Resting 12-lead and electrocardiogram monitoring in the hospital,” Science Direct, https://www.sciencedirect.com/science/article/pii/S0899588516300284?via%3Dihub (accessed Sep. 27, 2023).

[3] P. Van Gent, “Welcome to HeartPy,” Python Heart Rate Analysis Toolkit, https://python-heart-rate-analysis-toolkit.readthedocs.io/en/latest/ (accessed Sep. 27, 2023).

[4] C. Gonzalez, E. Meza Vega, and Hakim, “find-peaks,” GitHub, Mar. 12, 2021. https://github.com/claydergc/find-peaks (accessed Dec. 01, 2023).

If any other references were missed here, they are referenced in the code or in files related to the code.
