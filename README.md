# Project Title: Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning

**Team Members:**
* Boyang Xia - Student ID: 2023533073
* Jiawen Dai - Student ID: 2023533132
* Zhichen Zhong - Student ID: 2023533131

## Project Overview

This project undertakes the implementation and evaluation of the Diffusion Q-Learning (Diffusion-QL) algorithm, introduced in the ICLR 2023 paper "Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning" by Wang et al. Building upon a thorough understanding of this work, we further aim to explore potential innovations.

The core objective is to leverage the high expressiveness of **diffusion models** to represent policies in the context of **offline reinforcement learning (RL)**. Offline RL aims to learn effective policies from a pre-collected static dataset without further interaction with the environment, which is crucial for applications where online exploration is costly or risky. This project will delve into how Diffusion-QL addresses the common challenges in offline RL, such as function approximation errors on out-of-distribution actions and the limited expressiveness of traditional policy classes.

## Background and Motivation

Offline Reinforcement Learning (Offline RL or Batch RL) presents a significant paradigm for learning decision-making policies in a wide array of real-world scenarios, including autonomous driving, healthcare, and robotics, by utilizing previously gathered data. A primary challenge in offline RL is **distributional shift**: policies trained naively on a static dataset may learn to exploit errors in the value function estimates for out-of-distribution (OOD) actions, leading to poor performance when deployed.

Existing approaches to mitigate this issue often involve:
1.  **Policy Regularization**: Constraining the learned policy to stay close to the behavior policy (the policy that generated the offline data).
2.  **Value Function Constraints**: Penalizing OOD actions by assigning them low values.
3.  **Model-Based Methods**: Learning a model of the environment dynamics.
4.  **Sequence Modeling**: Treating offline RL as a sequence prediction problem.

The paper argues that many policy regularization methods suffer because the policy classes they use (e.g., Gaussian policies) are not expressive enough to accurately model complex, potentially multimodal, behavior policies often found in offline datasets. This can lead to improper regularization and suboptimal solutions. Diffusion models, known for their ability to model complex distributions, offer a promising alternative.

### Core Idea: Diffusion Q-Learning (Diffusion-QL)

The central contribution of the referenced paper is the **Diffusion Q-Learning (Diffusion-QL)** algorithm. This algorithm innovatively employs a **Conditional Diffusion Model** to represent the policy $\pi_{\theta}(a|s)$ in reinforcement learning. Given the current state $s$, this model generates an action $a$ through an Iterative Denoising Process.

The primary features and advantages of Diffusion-QL include:

1.  **Expressive Policy Representation**: The algorithm utilizes a Denoising Diffusion Probabilistic Model (DDPM), typically based on a Multi-Layer Perceptron (MLP), to construct the policy network. This powerful generative model enables the policy to capture and learn highly complex action distributions, including **Multimodality**, which is often present in offline datasets composed of diverse behavioral patterns (e.g., demonstrations from different experts). This overcomes the limitations of traditional policy classes (like Gaussian policies) in representing complex behaviors.

2.   **Integrated Policy Regularization and Improvement**: The training objective for the diffusion policy in Diffusion-QL skillfully combines two crucial components:
    **Behavior Cloning Loss $\mathcal{L}_d(\theta)$**: This loss term (Equation 2 in the paper) drives the diffusion model to learn and mimic the behavior policy $\pi_b$ present in the offline dataset $\mathcal{D}$. This constitutes a powerful, sample-based implicit regularization mechanism, encouraging the learned policy to generate actions similar to those in the training data, thereby effectively constraining the policy to be "In-Distribution" and avoiding blind exploration in unknown regions. This is achieved by training a noise prediction model $\epsilon_{\theta}(a^i, s, i)$ to accurately predict the noise added to clean actions from the dataset during the diffusion process.
    **Q-Learning Guidance / Policy Improvement Term $\mathcal{L}_q(\theta)$**: This loss term injects guidance signals, derived from an independently learned action-value function (Q-function) $Q_{\phi}(s,a)$, into the diffusion model's training process. It encourages the policy network $\pi_{\theta}$ to generate actions $a^0$ (sampled from $\pi_{\theta}$) that maximize the learned Q-function values (i.e., maximizing $ \mathbb{E}_{s \sim \mathcal{D}, a^0 \sim \pi_{\theta}} [Q_{\phi}(s, a^0)] $). Crucially, the gradient from the Q-value function can be backpropagated through the reparameterized sampling process of the diffusion model, directly optimizing the policy parameters.

* **Offline Model-Free Approach**:
    Diffusion-QL is a model-free reinforcement learning method. It does not rely on learning a dynamics model of the environment but instead learns the policy and value functions directly from a static, pre-collected dataset.

* **Standard Q-Learning Components**:
    The action-value function $Q_{\phi}(s,a)$ is learned in a conventional manner by minimizing the Bellman error. To enhance training stability and mitigate Q-value overestimation, the algorithm incorporates established techniques such as **Double Q-Learning** and **Target Networks**.

The experimental results in the paper robustly demonstrate that the outstanding expressiveness of the diffusion policy, coupled with the effective integration of behavior cloning (as an implicit regularization) and Q-learning guidance, are the key contributors to Diffusion-QL's state-of-the-art performance across numerous D4RL benchmark tasks.

## Project Objectives 1: Implement the algorithm described in the original paper

Our primary goal for this project is to **faithfully implement the Diffusion-QL algorithm** as detailed in the paper. This involves:

1.  **Implementing the Conditional Diffusion Policy**:
    * Develop the MLP-based noise prediction network $\epsilon_{\theta}(a^i, s, i)$.
    * Implement the forward noising process (for training) and the reverse denoising sampling process.
    * Incorporate the specified noise schedule.
2.  **Implementing the Q-Networks**:
    * Set up two Q-networks ($Q_{\phi_1}, Q_{\phi_2}$) and their corresponding target Q-networks ($Q_{\phi_1'}, Q_{\phi_2'}$).
    * Implement a target policy network ($\pi_{\theta'}$ or $\epsilon_{\theta'}$).
3.  **Implementing the Training Loop (Algorithm 1 in the paper)**:
    * Implement the Q-value function learning step, including sampling next actions from the target policy and using the minimum of target Q-values (Equation 4).
    * Implement the policy learning step, minimizing the combined loss $\mathcal{L}(\theta) = \mathcal{L}_d(\theta) - \alpha \cdot \mathbb{E}[Q_{\phi}(s,a^0)]$ (Equation 3), including the Q-value normalization for $\alpha$.
    * Implement soft updates for all target networks.
4.  **Evaluation**:
    * Set up experiments on selected D4RL benchmark tasks.
    * Reproduce or compare against some of the baseline results reported in the paper for these tasks.
    * Utilize the D4RL normalized score for performance comparison.
5.  **Analysis (Time Permitting)**:
    * Investigate the effect of the number of diffusion steps <span class="math-inline">N</span> on performance and computational cost, as discussed in the paper (Section 4, Figure 2; Section 5, Figure 3).
    * Analyze the interplay between the behavior cloning term <span class="math-inline">\\mathcal\{L\}\_d</span> and the Q-learning term <span class="math-inline">\\mathcal\{L\}\_q</span>.

## Project Objective 2: Innovations / Further Exploration

