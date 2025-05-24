
# EdgeRIC-Enabled RL-Based MAC Scheduler for srsRAN-Based Open RAN Systems

This repository provides an integrated real-time framework for MAC layer scheduling using Reinforcement Learning (RL) within an srsRAN + EdgeRIC setup. The architecture supports deployment of Q-Learning, PPO, and Actor-Critic agents as μApps interacting with a simulated or real RAN stack via ZMQ and Redis interfaces.

## 🔗 Repository Link
[GitHub: Sikander-Kathat/srsRAN](https://github.com/Sikander-Kathat/srsRAN.git)

---

## **🧱 Setup Instructions**

### **Step 1: Clone and Run EdgeRIC Base**
```bash
git clone https://github.com/ushasigh/EdgeRIC-A-real-time-RIC.git
cd EdgeRIC-A-real-time-RIC
git checkout oaic-workshop
sudo docker pull nlpurnhyun/edgeric_base_oaic
sudo ./dockerrun_edgeric_oaic.sh host 0
```

### **Step 2: Build srsRAN Inside Docker**
```bash
./make_ran.sh
```

---

### **Running in Docker Container: Run the following on every terminal**
```bash
cd EdgeRIC-A-real-time-RIC
sudo ./dockerexec_edgeric_oaic.sh 0
```

---

## **🚀 Running the System (Multi-UE Setup)**

### **Run the GRC Broker (2UE Scenario)**
```bash
# Terminal 1
python3 top_block_2ue_no_gui.py
```

### **EPC**
```bash
# Terminal 2
./run_epc.sh
```

### **eNB**
```bash
# Terminal 3
./run_enb.sh
```

### **UEs**
```bash
# Terminal 4
./run_srsran_2ue.sh
```

---

## **📡 Traffic Generation**

### **Terminal 5**
```bash
cd traffic-generator
./iperf_server_2ues.sh
```

### **Terminal 6**
```bash
cd traffic-generator
./iperf_client_2ues.sh 21M 5M 10000
```

---

## **🧠 Running EdgeRIC Core and Scheduling**

### **Start Redis Server**
```bash
# Terminal 7
cd edgeric
redis-server
```

---

## **⚙️ muApp1 – Downlink Scheduler (Weight-Based)**

The scheduling logic in srsenb is updated to support a weight-based abstraction. This allows dynamic scheduling where a weight `w_i` is given to each UE, allocating `[w_i × available_RBGs]`.

```bash
# Terminal 8
cd edgeric/muApp1
redis-cli set scheduling_algorithm "Max CQI"  # Initial scheduler
python3 muApp1_run_DL_scheduling.py
```

### **Switch to RL-Based Scheduling (Pre-trained)**
```bash
redis-cli set scheduling_algorithm "RL"
python3 muApp1_run_DL_scheduling.py
```

### **Run Specific Models**
```bash
python3 q_learning_MAc_scheduling.py
# OR
python3 actor_critic_MAc_scheduling.py
```

---

## **🎯 Training RL Models**

### **1. PPO**
```bash
cd edgeric/muApp2
python3 muApp2_train_RL_DL_scheduling.py
```

### **2. Q-Learning**
```bash
cd edgeric/muApp2/q_learning
python3 muApp2_train_RL_DL_q_learning_scheduling.py
```

### **3. Actor-Critic**
```bash
cd edgeric/muApp2/actor_critic
python3 actor_critic_mac_scheduling.py
```

---

## **📊 Real-Time Monitoring (muApp3)**
```bash
# Terminal 9
cd edgeric/muApp3
python3 muApp3_monitor_terminal.py
```

---

## 📡 Real-World Deployment Note

To validate over-the-air MAC scheduling, you may replace the GNU Radio emulation with **SDR hardware** such as the **USRP X410**.

---
