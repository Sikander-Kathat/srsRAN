# RL-Driven MAC Scheduling with EdgeRIC and srsRAN

This repository provides an integrated real-time framework for MAC layer scheduling using Reinforcement Learning (RL) within an srsRAN + EdgeRIC setup. The architecture supports deployment of Q-Learning, PPO, and Actor-Critic agents as μApps interacting with a simulated or real RAN stack via ZMQ and Redis interfaces.

## 🔗 Repository Link
https://github.com/Sikander-Kathat/srsRAN.git

---

## Setup Instructions

### Step 1: Clone and Run EdgeRIC Base
```bash
git clone https://github.com/ushasigh/EdgeRIC-A-real-time-RIC.git
cd EdgeRIC-A-real-time-RIC
git checkout oaic-workshop
sudo docker pull nlpurnhyun/edgeric_base_oaic
sudo ./dockerrun_edgeric_oaic.sh host 0

**Step 2: Build srsRAN Inside Docker**
./make_ran.sh

**Running the System
Multi-UE Setup**

EPC: ./run_epc.sh
eNB: ./run_enb.sh
UEs: ./run_srsran_2ue.sh

**Traffic Generation
iperf for UDP Traffic**

cd traffic-generator
./iperf_server_2ues.sh       # Terminal 5
./iperf_client_2ues.sh 21M 5M 10000   # Terminal 6

**Running RL-Based Scheduling
muApp1 – Inference Execution (Pre-trained Models)**

cd edgeric/muApp1
redis-cli set scheduling_algorithm "RL"
python3 muApp1_run_DL_scheduling.py

**To Run Q_learning or Actor critic Based RL model**
cd edgeric/muApp1
python3 q_learning_MAc_scheduling.py
**OR**
python3 actor_critic_MAc_scheduling.py

**Training RL Models
1. Proximal Policy Optimization (PPO)**

cd edgeric/muApp2
python3 muApp2_train_RL_DL_scheduling.py

**2. Tabular Q-Learning**

cd edgeric/muApp2/q_learning
python3 muApp2_train_RL_DL_q_learning_scheduling.py

**3. Actor-Critic**
cd edgeric/muApp2/actor_critic
python3 actor_critic_mac_scheduling.py

**Monitoring and Metrics
muApp3 – Real-Time Monitoring**
cd edgeric/muApp3
python3 muApp3_monitor_terminal.py






