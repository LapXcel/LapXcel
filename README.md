# LapXcel: Sim Racing Telemetry Optimizer

<img src="./logo.png" alt="Project Logo" width="400">

## About
This project aims to develop a software solution to optimize sim racing by analyzing race telemetry data. By ingesting data such as speed, acceleration, braking, and steering inputs, the software will identify patterns to aid racers in getting the theoretical best lap time on any given track. The system will continuously learn from telemetry inputs across various tracks and conditions, allowing sim racers to refine their performance and understand the fastest possible lap dynamics. This tool is intended for professional e-sports teams and individual enthusiasts.

## Getting started
1. Purchase and install [Assetto Corsa](https://store.steampowered.com/app/244210/Assetto_Corsa/) on Steam, and download the free [Content Manager](https://assettocorsa.club/content-manager.html) extension software.
2. Clone this repository to your local machine. (See [here](https://help.github.com/en/articles/cloning-a-repository) for instructions on how to clone a repository.
3. Getting the Assetto Corsa app working:
    - Install the [Python 3.3.5](https://legacy.python.org/download/releases/3.3.5/) interpreter. This is what Assetto Corsa uses and we need this locally for the socket import to not throw errors in AC.
    - Copy the `ACRL` folder to the `apps/python` folder in your Assetto Corsa installation directory. (e.g. `C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\apps\python`)
    - Run Assetto Corsa and enable the `ACRL` app in the `General` tab of the `Settings` menu. (You can also enable it through the `Content Manager` settings, or in the `Custom Shaders Patch` tab if you have CSP installed).

## Benefits and Outcomes
- Improve sim racers lap times
- Analyze telemetry data

## Team
| Name   | Student Number   | Email  |
|------------|------------|------------|
| [Colby Todd](https://www.linkedin.com/in/colbytodd/) | 300241178| ctodd083@uottawa.ca|
| [Engy Elsayed](https://www.linkedin.com/in/engy-els) | 300228400| eelsa005@uottawa.ca|
| [Sarah Siage](https://www.linkedin.com/in/sarah-siage-167144224)| 300228396| ssiag101@uottawa.ca |
| [Samuel Braun](https://www.linkedin.com/in/samuel-braun-5a1435221/)| 300238833| sbrau038@uottawa.ca|

## Customer
Sim Racer

Alfred Genadri

alfredgenadri@gmail.com

## License
This project is licensed under the MIT License.
