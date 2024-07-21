# Object Detection for Blind with MobileNet V3
A research project made for the Indonesian Student Research Olympiad / Olimpiade Penelitian Siswa Indonesia (OPSI)
\n
A revision of the [Vizhat Object Detection](https://github.com/hmsgobrr/vizhat-objectdetection) project, utilizes the new MobileNet V3 model.
## Setup on RaspberryPi 4B
Install required dependencies
```sh
sudo apt update
sudo apt install python3 python3-pip git
sudo apt install espeak # for text-to-speech
```
Create virtual environment and install required python libraries
```sh
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install torch opencv-python-headless
pip install pyttsx3 # for text-to-speech
```
Setup bluetooth earphone for text-to-speech output
```sh
sudo apt install bluetooth pi-bluetooth bluez pulseaudio-module-bluetooth
sudo usermod -a -G bluetooth $USER
sudo reboot
bluetoothctl
	# Connect to earphone
	scan on # Scan for available bluetooth devices to connect
	scan off # Turn off scanning once you see the earphone device name, save the address of the earphone for later
	pair <THE ADDRESS>
	trust <THE ADDRESS>
	connect <THE ADDRESS>
	exit # Exit bluetoothctl
```
Clone repository & run program (do not exit the virtual environment yet)
```sh
git clone https://github.com/hmsgobrr/object-detection-opsi.git
cd object-detection-opsi
python detect.py
```
